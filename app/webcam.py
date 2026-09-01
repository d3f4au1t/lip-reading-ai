from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.camera import discover_macos_cameras, open_camera, resolve_camera
from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
from app.model import WordCertainty
from app.pipeline import PipelineResult, VisualSpeechPipeline


@dataclass
class CaptionEntry:
    entry_id: int
    started_at: float
    text: str = ""
    word_certainties: tuple[WordCertainty, ...] = tuple()
    pending: bool = True


class CaptionHistory:
    """Keep the three most recent recognition windows in display order."""

    def __init__(self, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("Caption history must contain at least one row.")
        self.limit = limit
        self.entries: list[CaptionEntry] = []
        self._next_entry_id = 0

    def start(self, started_at: float | None = None) -> int:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        self.entries.append(
            CaptionEntry(
                entry_id=entry_id,
                started_at=time.monotonic() if started_at is None else started_at,
            )
        )
        if len(self.entries) > self.limit:
            self.entries.pop(0)
        return entry_id

    def complete(
        self,
        entry_id: int,
        text: str,
        word_certainties: tuple[WordCertainty, ...] = tuple(),
    ) -> None:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.text = text
                entry.word_certainties = word_certainties
                entry.pending = False
                return
        raise KeyError(f"Caption entry {entry_id} is no longer visible.")


def _processing_placeholder(started_at: float, now: float) -> str:
    phase = int(max(0.0, now - started_at) * 2) % 3
    return "." * (phase + 1)


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _certainty_color(certainty: float) -> tuple[int, int, int]:
    """Return an OpenCV BGR color from red (0%) through yellow to green (100%)."""
    certainty = min(1.0, max(0.0, certainty))
    red = round(255 * min(1.0, 2.0 * (1.0 - certainty)))
    green = round(255 * min(1.0, 2.0 * certainty))
    return (0, green, red)


def _draw_word_certainties(
    frame: np.ndarray,
    word_certainties: tuple[WordCertainty, ...],
    left: int,
    word_baseline: int,
    percentage_baseline: int,
    available_width: int,
    preferred_word_scale: float = 0.68,
    thickness: int = 2,
) -> None:
    """Draw large words with smaller, aligned certainty percentages underneath."""
    if not word_certainties:
        return

    def measure(
        word_scale: float,
    ) -> tuple[float, list[tuple[int, int]], int, int]:
        percentage_scale = word_scale * 0.52
        widths: list[tuple[int, int]] = []
        for item in word_certainties:
            word_width = cv2.getTextSize(
                item.word, cv2.FONT_HERSHEY_SIMPLEX, word_scale, thickness
            )[0][0]
            percentage_width = cv2.getTextSize(
                f"{item.certainty:.0%}",
                cv2.FONT_HERSHEY_SIMPLEX,
                percentage_scale,
                1,
            )[0][0]
            widths.append((word_width, percentage_width))
        gap = max(5, round(12 * word_scale / preferred_word_scale))
        total_width = sum(max(pair) for pair in widths) + gap * (len(widths) - 1)
        return percentage_scale, widths, gap, total_width

    word_scale = preferred_word_scale
    percentage_scale, widths, gap, total_width = measure(word_scale)
    if total_width > available_width:
        word_scale *= available_width / total_width
        percentage_scale, widths, gap, total_width = measure(word_scale)

    x = left + max(0, (available_width - total_width) // 2)
    for item, (word_width, percentage_width) in zip(word_certainties, widths):
        column_width = max(word_width, percentage_width)
        word_x = x + (column_width - word_width) // 2
        percentage_x = x + (column_width - percentage_width) // 2
        _draw_text(
            frame,
            item.word,
            (word_x, word_baseline),
            word_scale,
            (255, 255, 255),
            thickness,
        )
        _draw_text(
            frame,
            f"{item.certainty:.0%}",
            (percentage_x, percentage_baseline),
            percentage_scale,
            _certainty_color(item.certainty),
            1,
        )
        x += column_width + gap


def _fit_text_scale(
    text: str, available_width: int, preferred: float = 0.62, minimum: float = 0.3
) -> float:
    text_width = cv2.getTextSize(
        " ".join(text.split()), cv2.FONT_HERSHEY_SIMPLEX, preferred, 2
    )[0][0]
    if text_width <= available_width or text_width == 0:
        return preferred
    return max(minimum, preferred * available_width / text_width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short-window visual-only webcam captions")
    parser.add_argument(
        "--camera",
        default="built-in",
        help="Camera selector: 'built-in' (default) or a numeric index",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="List macOS cameras and exit"
    )
    parser.add_argument("--window-seconds", type=float, default=4.0, help="Frames per recognition window")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cameras:
        devices = discover_macos_cameras()
        if not devices:
            print("No cameras reported by macOS.")
            return 0
        for device in devices:
            label = "built-in Mac camera" if device.is_builtin_mac_camera else "external/Continuity"
            print(f"{device.index}: {device.name} ({label})")
        return 0
    if args.window_seconds < 1.0 or args.window_seconds > 16.0:
        raise ValueError("Window length must be between 1 and 16 seconds.")

    device = resolve_camera(args.camera)
    print(f"Selected camera {device.index}: {device.name}")
    print("Loading visual speech model...")
    pipeline = VisualSpeechPipeline(args.checkpoint, args.device, args.beam_size)
    camera, first_frame = open_camera(device, TARGET_FPS)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-speech")
    future: Future[PipelineResult] | None = None
    future_entry_id: int | None = None
    capture_entry_id: int | None = None
    buffered_frames: list[np.ndarray] = []
    target_frames = int(args.window_seconds * TARGET_FPS)
    caption_history = CaptionHistory(limit=3)
    status = "CAPTURING"
    face_visible = False
    latency: float | None = None
    frame_number = 0

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
            else:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError(f"{device.name} stopped returning frames.")
            frame_number += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffered_frames.append(rgb)
            if capture_entry_id is None:
                capture_entry_id = caption_history.start()

            if frame_number % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(80, 80))
                face_visible = len(faces) > 0

            if future is not None and future.done():
                if future_entry_id is None:
                    raise RuntimeError("A recognition task finished without a caption row.")
                try:
                    result = future.result()
                    caption_history.complete(
                        future_entry_id,
                        result.transcription or "[No words decoded — try again]",
                        result.recognition.word_certainties,
                    )
                    latency = result.preprocessing_seconds + result.recognition.inference_seconds
                except Exception as exc:
                    caption_history.complete(
                        future_entry_id, f"Could not transcribe: {exc}"
                    )
                future = None
                future_entry_id = None
                status = "CAPTURING"

            if future is None and len(buffered_frames) >= target_frames:
                segment = buffered_frames[-target_frames:]
                buffered_frames.clear()
                if capture_entry_id is None:
                    raise RuntimeError("A captured window has no caption row.")
                future_entry_id = capture_entry_id
                capture_entry_id = None
                future = executor.submit(pipeline.transcribe_frames, segment)
                status = "PROCESSING VISUAL SPEECH"
            elif len(buffered_frames) > target_frames * 2:
                buffered_frames = buffered_frames[-target_frames:]

            display = frame.copy()
            height, width = display.shape[:2]
            overlay = display.copy()
            panel_top = max(80, height - 270)
            cv2.rectangle(overlay, (0, 0), (width, 78), (15, 15, 15), -1)
            cv2.rectangle(overlay, (0, panel_top), (width, height), (15, 15, 15), -1)
            display = cv2.addWeighted(overlay, 0.78, display, 0.22, 0)
            face_color = (80, 220, 100) if face_visible else (80, 180, 255)
            _draw_text(display, "ASSISTIVE CAPTIONING PROTOTYPE", (18, 30), 0.65, (255, 255, 255))
            _draw_text(display, status, (18, 62), 0.72, (80, 220, 255))
            _draw_text(
                display,
                "FACE VISIBLE" if face_visible else "POSITION FACE TOWARD CAMERA",
                (max(18, width - 360), 31),
                0.55,
                face_color,
            )
            if latency is not None:
                _draw_text(display, f"Last latency: {latency:.1f}s", (max(18, width - 270), 62), 0.5, (220, 220, 220), 1)

            row_area_top = panel_top + 12
            row_area_bottom = height - 34
            row_height = max(38, (row_area_bottom - row_area_top) // 3)
            now = time.monotonic()
            for row_index in range(3):
                row_top = row_area_top + row_index * row_height
                baseline = row_top + row_height // 2 + 8
                if row_index >= len(caption_history.entries):
                    _draw_text(
                        display, "—", (22, baseline), 0.55, (105, 105, 105), 1
                    )
                else:
                    entry = caption_history.entries[row_index]
                    if entry.pending:
                        placeholder = _processing_placeholder(entry.started_at, now)
                        _draw_text(
                            display,
                            placeholder,
                            (22, baseline),
                            0.78,
                            (80, 220, 255),
                        )
                    else:
                        if entry.word_certainties:
                            _draw_word_certainties(
                                display,
                                entry.word_certainties,
                                left=22,
                                word_baseline=row_top + row_height // 2,
                                percentage_baseline=row_top + row_height - 9,
                                available_width=width - 44,
                            )
                        else:
                            scale = _fit_text_scale(entry.text, width - 44)
                            _draw_text(
                                display,
                                entry.text,
                                (22, baseline),
                                scale,
                                (255, 255, 255),
                                2,
                            )
                if row_index < 2:
                    divider_y = row_area_top + (row_index + 1) * row_height
                    cv2.line(
                        display,
                        (18, divider_y),
                        (width - 18, divider_y),
                        (70, 70, 70),
                        1,
                        cv2.LINE_AA,
                    )
            _draw_text(display, "Q: quit", (22, height - 12), 0.42, (180, 180, 180), 1)
            _draw_text(
                display,
                "WORD CERTAINTY: decoder estimate, not calibrated",
                (max(18, width - 390), height - 12),
                0.42,
                (120, 190, 255),
                1,
            )

            cv2.imshow("Visual-Only Assistive Captions", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
