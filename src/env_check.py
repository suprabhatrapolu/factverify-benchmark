"""Phase 0 environment verification.

Smoke-tests every required import, writes resolved package versions to
output/tables/package_versions.csv, and records hardware + platform facts
into output/results/run_metadata.json (brief §3, §10.4).
"""

import csv
import importlib.metadata
import os
import platform
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import utils

# Distribution names as installed by pip -> import names for the smoke test.
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "matplotlib": "matplotlib",
    "datasets": "datasets",
    "transformers": "transformers",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "gensim": "gensim",
    "nltk": "nltk",
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "sec-edgar-downloader": "sec_edgar_downloader",
    "tqdm": "tqdm",
    "pyarrow": "pyarrow",
}


def get_ram_gb() -> float:
    """Total physical RAM in GB via the Win32 API (no extra dependencies)."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return round(status.ullTotalPhys / 1024**3, 1)


def main() -> None:
    logger = utils.setup_logging(0)

    # 1. Smoke-test imports and collect resolved versions.
    rows = []
    for dist_name, import_name in PACKAGES.items():
        importlib.import_module(import_name)
        rows.append({"package": dist_name,
                     "version": importlib.metadata.version(dist_name)})
        logger.info("import ok: %s %s", dist_name, rows[-1]["version"])

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    versions_path = config.TABLES_DIR / "package_versions.csv"
    with open(versions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["package", "version"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", versions_path)

    # 2. Hardware + platform facts.
    import torch

    gpu_available = torch.cuda.is_available()
    metadata = {
        "cpu": platform.processor(),
        "cores": os.cpu_count(),
        "ram_gb": get_ram_gb(),
        "gpu": torch.cuda.get_device_name(0) if gpu_available else "none",
        "torch_cuda_available": gpu_available,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "key_package_versions": {r["package"]: r["version"] for r in rows},
        "seed": config.SEED,
        "date": date.today().isoformat(),
        "phase_wall_times_s": {},
        "bilstm_train_size": None,      # filled in Phase 4
        "distilbert_train_size": None,  # filled in Phase 4
    }
    existing = utils.read_json(config.RUN_METADATA_JSON)
    existing.update(metadata)
    utils.write_json(config.RUN_METADATA_JSON, existing)
    logger.info("wrote %s (gpu=%s)", config.RUN_METADATA_JSON, metadata["gpu"])


if __name__ == "__main__":
    main()
