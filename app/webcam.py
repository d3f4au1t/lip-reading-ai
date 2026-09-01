from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.camera import discover_macos_cameras, open_camera, resolve_camera
from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
from app.model import WordCertainty
from app.pipeline import PipelineResult, VisualSpeechPipeline


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


def _draw_certainty_line(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    thickness: int = 2,
) -> None:
    """Draw caption words in white and percentage tokens in certainty colors."""
    x, y = origin
    space_width = cv2.getTextSize(
        " ", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0]
    for token in text.split():
        color = (255, 255, 255)
        if token.endswith("%"):
            try:
                color = _certainty_color(float(token[:-1]) / 100.0)
            except ValueError:
                pass
        _draw_text(frame, token, (x, y), scale, color, thickness)
        token_width = cv2.getTextSize(
            token, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0]
        x += token_width + space_width


def _wrap_caption(text: str, max_chars: int = 54, max_lines: int = 4) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[-max_lines:] or ["Waiting for visible speech..."]


def _certainty_caption(
    caption: str, word_certainties: tuple[WordCertainty, ...]
) -> str:
    if not word_certainties:
        return caption
    return "  ".join(
        f"{item.word} {item.certainty:.0%}" for item in word_certainties
    )


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
    buffered_frames: list[np.ndarray] = []
    target_frames = int(args.window_seconds * TARGET_FPS)
    caption = "Waiting for visible speech..."
    word_certainties: tuple[WordCertainty, ...] = tuple()
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

            if frame_number % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(80, 80))
                face_visible = len(faces) > 0

            if future is not None and future.done():
                try:
                    result = future.result()
                    caption = result.transcription or "[No words decoded — try again]"
                    word_certainties = result.recognition.word_certainties
                    latency = result.preprocessing_seconds + result.recognition.inference_seconds
                except Exception as exc:
                    caption = f"Could not transcribe: {exc}"
                    word_certainties = tuple()
                future = None
                status = "CAPTURING"

            if future is None and len(buffered_frames) >= target_frames:
                segment = buffered_frames[-target_frames:]
                buffered_frames.clear()
                future = executor.submit(pipeline.transcribe_frames, segment)
                status = "PROCESSING VISUAL SPEECH"
            elif len(buffered_frames) > target_frames * 2:
                buffered_frames = buffered_frames[-target_frames:]

            display = frame.copy()
            height, width = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (width, 78), (15, 15, 15), -1)
            cv2.rectangle(overlay, (0, max(0, height - 230)), (width, height), (15, 15, 15), -1)
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

            certainty_text = _certainty_caption(caption, word_certainties)
            caption_lines = _wrap_caption(
                certainty_text, max_chars=max(36, width // 16)
            )
            line_height = 38
            base_y = height - 42 - (len(caption_lines) - 1) * line_height
            for index, line in enumerate(caption_lines):
                origin = (22, base_y + index * line_height)
                if word_certainties:
                    _draw_certainty_line(display, line, origin, 0.72)
                else:
                    _draw_text(display, line, origin, 0.9, (255, 255, 255), 2)
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
