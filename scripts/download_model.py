"""Automated model asset downloader for fast-intent-classifier.

Fetches the ONNX model graph and HuggingFace fast-tokenizer files
from the official sentence-transformers repository.
"""

import sys
import urllib.request
from pathlib import Path

MODEL_REPO_BASE = (
    "https://huggingface.co/sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
)

TARGET_DIR = Path(__file__).resolve().parents[1] / "models"

ASSETS = {
    "model.onnx": f"{MODEL_REPO_BASE}/onnx/model.onnx",
    "tokenizer.json": f"{MODEL_REPO_BASE}/tokenizer.json",
    "tokenizer_config.json": f"{MODEL_REPO_BASE}/tokenizer_config.json",
    "special_tokens_map.json": f"{MODEL_REPO_BASE}/special_tokens_map.json",
    "config.json": f"{MODEL_REPO_BASE}/config.json",
}

def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r  -> Progress: {percent:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)"
        )
        sys.stdout.flush()


def download_assets() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Ensuring model artifacts exist in: {TARGET_DIR}")

    for filename, url in ASSETS.items():
        destination = TARGET_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[+] Found cached: {filename}")
            continue

        print(f"[↓] Fetching: {filename}")
        try:
            urllib.request.urlretrieve(
                url, destination, reporthook=_report_progress
            )
            print()
        except Exception as err:
            if destination.exists():
                destination.unlink()
            print(f"\n[!] Failed to download {filename}: {err}")
            sys.exit(1)

    print("[✓] All model artifacts are ready.")


if __name__ == "__main__":
    download_assets()
