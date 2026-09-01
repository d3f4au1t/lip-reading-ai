from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import psutil
import torch

from app.config import DEFAULT_CHECKPOINT, THIRD_PARTY_ROOT, validate_model_source


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    device: str
    model_load_seconds: float
    inference_seconds: float
    video_seconds: float
    real_time_factor: float
    decoding_score_per_token: float | None
    average_process_cpu_percent: float
    memory_rss_mb: float

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot access Apple Metal.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, mps, cuda.")
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class AutoAVSRRecognizer:
    """Thin inference wrapper around the pinned official Auto-AVSR model."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
        beam_size: int = 10,
    ) -> None:
        validate_model_source()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run `python scripts/download_assets.py`."
            )
        if beam_size < 1:
            raise ValueError("Beam size must be at least 1.")

        source = str(THIRD_PARTY_ROOT)
        if source not in sys.path:
            sys.path.insert(0, source)
        from lightning import ModelModule, get_beam_search_decoder

        self.encoder_device = choose_device(device)
        # Auto-AVSR's legacy ESPnet CTC prefix scorer creates CPU index tensors.
        # Keep decoding on CPU for MPS while accelerating the expensive visual
        # frontend and Conformer encoder on Metal.
        self.decoder_device = (
            torch.device("cpu") if self.encoder_device.type == "mps" else self.encoder_device
        )
        self.beam_size = beam_size
        started = time.perf_counter()
        args = Namespace(modality="video", ctc_weight=0.1)
        module = ModelModule(args)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        module.model.load_state_dict(state_dict, strict=True)
        self.model = module.model.to(self.encoder_device).eval()
        if self.decoder_device != self.encoder_device:
            self.model.decoder.to(self.decoder_device)
            self.model.ctc.to(self.decoder_device)
        self.text_transform = module.text_transform
        self.beam_search = get_beam_search_decoder(
            self.model,
            module.token_list,
            beam_size=beam_size,
            ctc_weight=0.1,
        )
        _synchronize(self.encoder_device)
        self.model_load_seconds = time.perf_counter() - started

    def transcribe(self, video: torch.Tensor, fps: float = 25.0) -> RecognitionResult:
        if video.ndim != 4 or video.shape[1:] != (1, 88, 88):
            raise ValueError(
                f"Expected preprocessed video [T, 1, 88, 88], received {tuple(video.shape)}."
            )
        if len(video) < 3:
            raise ValueError("At least three preprocessed frames are required.")

        process = psutil.Process()
        cpu_before = process.cpu_times()
        started = time.perf_counter()
        sample = video.unsqueeze(0).to(self.encoder_device)
        with torch.inference_mode():
            encoded = self.model.frontend(sample)
            encoded = self.model.proj_encoder(encoded)
            encoded, _ = self.model.encoder(encoded, None)
            encoded = encoded.squeeze(0).to(self.decoder_device)
            hypotheses = self.beam_search(encoded)
        _synchronize(self.encoder_device)
        inference_seconds = time.perf_counter() - started
        cpu_after = process.cpu_times()
        cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)

        if not hypotheses:
            raise RuntimeError("The decoder returned no transcription hypotheses.")
        best = hypotheses[0].asdict()
        token_ids = torch.tensor([int(token) for token in best["yseq"][1:]])
        text = self.text_transform.post_process(token_ids).replace("<eos>", "").strip()
        score = best.get("score")
        if isinstance(score, torch.Tensor):
            score = float(score.detach().cpu())
        token_count = max(1, len(token_ids) - 1)
        score_per_token = float(score) / token_count if score is not None else None
        video_seconds = len(video) / fps

        return RecognitionResult(
            text=text,
            device=(
                str(self.encoder_device)
                if self.decoder_device == self.encoder_device
                else f"{self.encoder_device} encoder + {self.decoder_device} decoder"
            ),
            model_load_seconds=self.model_load_seconds,
            inference_seconds=inference_seconds,
            video_seconds=video_seconds,
            real_time_factor=inference_seconds / video_seconds,
            decoding_score_per_token=score_per_token,
            average_process_cpu_percent=100 * cpu_seconds / inference_seconds,
            memory_rss_mb=process.memory_info().rss / (1024 * 1024),
        )
