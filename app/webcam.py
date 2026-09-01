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

from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
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


def _wrap_caption(text: str, max_chars: int = 54) -> list[str]:
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
    return lines[-2:] or ["Waiting for visible speech..."]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short-window visual-only webcam captions")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--window-seconds", type=float, default=4.0, help="Frames per recognition window")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_seconds < 1.0 or args.window_seconds > 16.0:
        raise ValueError("Window length must be between 1 and 16 seconds.")

    print("Loading visual speech model...")
    pipeline = VisualSpeechPipeline(args.checkpoint, args.device, args.beam_size)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check macOS Camera privacy permission for your terminal."
        )
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-speech")
    future: Future[PipelineResult] | None = None
    buffered_frames: list[np.ndarray] = []
    target_frames = int(args.window_seconds * TARGET_FPS)
    caption = "Waiting for visible speech..."
    status = "CAPTURING"
    face_visible = False
    latency: float | None = None
    frame_number = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
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
                    latency = result.preprocessing_seconds + result.recognition.inference_seconds
                except Exception as exc:
                    caption = f"Could not transcribe: {exc}"
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
            cv2.rectangle(overlay, (0, max(0, height - 150)), (width, height), (15, 15, 15), -1)
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

            caption_lines = _wrap_caption(caption)
            base_y = height - 92 if len(caption_lines) == 2 else height - 58
            for index, line in enumerate(caption_lines):
                _draw_text(display, line, (22, base_y + index * 45), 0.9, (255, 255, 255), 2)
            _draw_text(display, "Q: quit", (22, height - 12), 0.42, (180, 180, 180), 1)
            _draw_text(
                display,
                "UNCERTAINTY: predictions are not calibrated",
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
