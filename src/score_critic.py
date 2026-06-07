"""Score 191 (z, h_l) pairs through the released NLA Critic and report FVE.

Round-trip: gold raw activation h_l + paper-format z (from verbalizer_claude.py)
            → NLACritic.score(z, h_l) → (mse, cos)

Reports:
    - per-pair MSE and cosine distributions (mean, std, median, min, max)
    - aggregate FVE = 1 − mean(MSE) / 2.0

The denominator 2.0 is the expected MSE of two random unit vectors after
mse_scale normalization (orthogonal cos = 0 → MSE = 2(1−0) = 2). So
FVE = 0 means "no better than random" and FVE = 1 means "perfect reconstruction".

Usage:
    python src/score_critic.py \\
        --critic-dir data/nla_critic_qwen7b/ \\
        --activations-pt data/qwen7b_layer20_4bit_n200.pt \\
        --zs-pt data/qwen7b_paper_zs_n200.pt \\
        --output results/path_x_fve.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nla_critic import NLACritic


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "score_critic.py requires a GPU for the Critic forward pass"

    # ---- Load Critic (~11 GB on GPU, ~2 min) ----
    print("Loading NLA Critic (~11 GB on GPU, ~2 min)...")
    critic = NLACritic(args.critic_dir, device=device, dtype=torch.bfloat16)

    # ---- Load activations + z's ----
    act_data = torch.load(args.activations_pt, weights_only=False)
    z_data = torch.load(args.zs_pt, weights_only=False)
    gold_activations = act_data["raw_activations"]
    paper_zs = z_data["paper_zs"]

    assert len(gold_activations) == len(paper_zs), \
        f"Mismatched lengths: {len(gold_activations)} activations vs {len(paper_zs)} z's. " \
        f"They must come from the same extract_activations.py + verbalizer_claude.py run."

    # Drop pairs where z generation failed (empty string)
    valid_pairs = [(z, g) for z, g in zip(paper_zs, gold_activations) if z]
    print(f"\n✅ {len(valid_pairs)}/{len(paper_zs)} valid pairs ready (skipping empty z's)")
    print(f"   gold magnitude: mean={gold_activations.norm(dim=-1).mean():.1f}")

    # ---- Score ----
    print(f"\nScoring {len(valid_pairs)} pairs (~2-3 sec each)...")
    mses: list[float] = []
    coses: list[float] = []
    for z, gold in tqdm(valid_pairs, desc="Scoring"):
        mse, cos = critic.score(z, gold)
        mses.append(mse)
        coses.append(cos)

    mses_arr = np.array(mses)
    coses_arr = np.array(coses)
    fve = 1.0 - mses_arr.mean() / 2.0

    # ---- Report ----
    print("\n=== RESULTS ===")
    print(f"N pairs: {len(mses)}")
    print(f"\nMSE distribution:")
    print(f"  mean:   {mses_arr.mean():.4f}")
    print(f"  median: {np.median(mses_arr):.4f}")
    print(f"  std:    {mses_arr.std():.4f}")
    print(f"  min:    {mses_arr.min():.4f}")
    print(f"  max:    {mses_arr.max():.4f}")
    print(f"\ncos distribution:")
    print(f"  mean:   {coses_arr.mean():.4f}")
    print(f"  median: {np.median(coses_arr):.4f}")
    print(f"  std:    {coses_arr.std():.4f}")
    print(f"  min:    {coses_arr.min():.4f}")
    print(f"  max:    {coses_arr.max():.4f}")
    print(f"\n{'='*60}")
    print(f"📊 FVE = 1 − mean(MSE) / 2.0 = {fve:.4f}")
    print(f"{'='*60}")
    if fve > 0.5:
        print("   🎉 paper RL-level reproduction (paper RL: ~0.75)")
    elif fve > 0.2:
        print("   ✅ paper warmup-level (paper SFT: ~0.3-0.4)")
    elif fve > 0:
        print("   ✅ POSITIVE — validates the path X approach")
    else:
        print("   ❌ Still negative — check prompt format and Critic load")

    # ---- Save ----
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mses": mses_arr,
        "coses": coses_arr,
        "fve": fve,
        "n_pairs": len(mses),
        "config": {
            "critic_dir": str(args.critic_dir),
            "activations_pt": str(args.activations_pt),
            "zs_pt": str(args.zs_pt),
        },
    }, str(output))
    print(f"\n✅ Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--critic-dir", required=True,
                        help="Directory of the released NLA Critic checkpoint (output of download_critic.py)")
    parser.add_argument("--activations-pt", required=True,
                        help="Output of extract_activations.py")
    parser.add_argument("--zs-pt", required=True,
                        help="Output of verbalizer_claude.py")
    parser.add_argument("--output", default="results/path_x_fve.pt")
    args = parser.parse_args()
    main(args)
