from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party" / "auto_avsr"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "vsr_trlrs2lrs3vox2avsp_base.pth"
DEFAULT_SAMPLE = PROJECT_ROOT / "samples" / "test.mp4"
TARGET_FPS = 25.0
MAX_RECOMMENDED_SECONDS = 16.0


def validate_model_source() -> None:
    required = (
        THIRD_PARTY_ROOT / "lightning.py",
        THIRD_PARTY_ROOT / "preparation" / "detectors" / "mediapipe" / "detector.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Auto-AVSR source is missing. Initialize it with "
            "`git submodule update --init --recursive`. Missing: " + ", ".join(missing)
        )

