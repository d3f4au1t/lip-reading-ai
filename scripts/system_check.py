from __future__ import annotations

import json
import platform
import shutil
import sys

import psutil
import torch


def main() -> int:
    details = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "disk_free_gb": round(shutil.disk_usage(".").free / 1024**3, 1),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

