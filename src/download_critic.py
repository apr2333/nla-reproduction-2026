"""Download the released NLA Critic checkpoint from Hugging Face Hub.

The Critic is the AR (Activation Reconstructor) trained by Fraser-Taliente et al.
on Qwen2.5-7B-Instruct layer 20. It is a 21-layer truncated Qwen backbone
plus a Linear(d, d) value-head, distributed under Apache-2.0 by the paper's
official codebase: https://github.com/kitft/natural_language_autoencoders

The full checkpoint is ~10.9 GB across 3 sharded safetensors plus a
`value_head.safetensors`, plus an `nla_meta.yaml` sidecar that contains
prompt template, mse_scale, and other invariants the Critic was trained with.

Usage:
    python src/download_critic.py --output data/nla_critic_qwen7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "kitft/nla-qwen2.5-7b-L20-ar"
PATTERNS = ["*.json", "*.safetensors", "*.yaml", "tokenizer*", "*.txt"]


def main(output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} → {output} (~10.9 GB, ~2 min)...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(output),
        allow_patterns=PATTERNS,
    )

    # Verify the critical files arrived. snapshot_download has occasionally
    # produced truncated shard files on Colab Drive mounts; this catches that
    # before downstream code tries to load a 0-byte safetensors.
    required = [
        "nla_meta.yaml",
        "config.json",
        "tokenizer.json",
        "value_head.safetensors",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    missing = []
    for fname in required:
        path = output / fname
        if not path.exists() or path.stat().st_size < 1024:
            missing.append(fname)

    if missing:
        raise FileNotFoundError(
            f"Critic download incomplete. Missing or truncated: {missing}. "
            f"Re-run this script; if the issue persists, download "
            f"the missing shard with `huggingface_hub.hf_hub_download` directly."
        )

    total_mb = sum(p.stat().st_size for p in output.iterdir() if p.is_file()) / 1e6
    print(f"\n✅ Critic ready ({total_mb:.0f} MB across {len(list(output.iterdir()))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/nla_critic_qwen7b",
                        help="Local directory to download the Critic into")
    args = parser.parse_args()
    main(args.output)
