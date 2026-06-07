"""Extract Qwen-7B layer-20 activations from WikiText-103 prefixes.

Loads Qwen2.5-7B-Instruct in 4-bit nf4 quantization (so it fits on a 16 GB T4),
runs a forward pass on each text segment up to length 128 tokens, and grabs
the residual-stream activation at a fixed position (default 60) of layer 20.

Why position 60: paper's `_MIN_POSITION = 50` requires enough left context that
the residual stream has accumulated meaningful semantic state. Earlier
positions decode to noise.

Why 4-bit: paper uses fp16 (~14 GB). T4 with 16 GB cannot hold fp16 Qwen-7B
plus a forward pass. nf4 with double-quant fits in ~5.6 GB and leaves 10 GB
of inference headroom. See README §5.4 for the quantization caveat.

Usage:
    python src/extract_activations.py \\
        --model Qwen/Qwen2.5-7B-Instruct --layer 20 --position 60 \\
        --num-segments 200 --max-length 128 \\
        --output data/qwen7b_layer20_4bit_n200.pt
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def filter_wikitext(text: str, min_chars: int = 200) -> bool:
    """Drop short paragraphs and section headers, matching paper's data-gen filtering."""
    text = text.strip()
    return len(text) > min_chars and not text.startswith("=")


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "extract_activations.py requires a GPU"

    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---- Load Qwen 4-bit ----
    print(f"\nLoading {args.model} in 4-bit nf4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"✅ Model ready. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---- Load WikiText-103 and filter ----
    print("\nLoading WikiText-103...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                       split=f"train[:{args.num_segments * 5}]")
    texts = [x["text"] for x in raw if filter_wikitext(x["text"])][: args.num_segments]
    print(f"✅ {len(texts)} segments after filtering")

    # ---- Extract activations ----
    print(f"\nExtracting layer {args.layer} @ position {args.position} for {len(texts)} segments...")
    raw_activations = []
    extracted_token_strs = []
    prefix_for_claude = []
    skipped = 0

    with torch.no_grad():
        for i, text in enumerate(tqdm(texts, desc="Extracting")):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=args.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]

            if seq_len < args.position + 5:
                skipped += 1
                continue

            outputs = model(**inputs, output_hidden_states=True)

            # hidden_states is a tuple of length (num_layers + 1).
            # Index [0] is the post-embedding output; [k+1] is the output of
            # transformer block k. So "layer 20 output" = hidden_states[21].
            h_l = outputs.hidden_states[args.layer + 1][0, args.position, :]
            raw_activations.append(h_l.float().cpu())

            token_id = inputs["input_ids"][0, args.position].item()
            extracted_token_strs.append(tokenizer.decode([token_id]))
            prefix_text = tokenizer.decode(inputs["input_ids"][0, : args.position + 1])
            prefix_for_claude.append(prefix_text)

    raw_activations = torch.stack(raw_activations)

    # ---- Report ----
    mag = raw_activations.norm(dim=-1)
    print(f"\n✅ Extracted {raw_activations.shape[0]} (skipped {skipped} segments shorter than {args.position + 5} tokens)")
    print(f"   Shape: {tuple(raw_activations.shape)}")
    print(f"   Magnitude: mean={mag.mean():.1f}, std={mag.std():.1f}")
    print(f"   (paper Qwen layer 20 typical: 100-170)")
    print(f"   Sample tokens (first 10): {extracted_token_strs[:10]}")

    # ---- Save ----
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "raw_activations": raw_activations,
            "extracted_token_strs": extracted_token_strs,
            "prefix_for_claude": prefix_for_claude,
            "config": {
                "model": args.model,
                "layer": args.layer,
                "position": args.position,
                "n_segments": raw_activations.shape[0],
                "d_model": raw_activations.shape[1],
                "quantization": "4bit-nf4",
                "max_length": args.max_length,
            },
        },
        str(output),
    )
    print(f"\n✅ Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=20,
                        help="Transformer layer to extract from (0-indexed). Default 20 = paper's choice.")
    parser.add_argument("--position", type=int, default=60,
                        help="Token position to extract activation at. Must be >= 50 per paper.")
    parser.add_argument("--num-segments", type=int, default=200,
                        help="Number of WikiText segments to attempt. ~5%% will be too short.")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Tokenizer truncation length")
    parser.add_argument("--output", default="data/qwen7b_layer20_4bit_n200.pt")
    args = parser.parse_args()
    main(args)
