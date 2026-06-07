"""Generate paper-format Claude z's for each WikiText prefix.

This script substitutes a Claude-3.5-Haiku call for the released NLA AV.
The released AV reads an injected activation vector; my Claude-as-AV reads
the *text prefix* up to the same token position. The substitution preserves
the AV→z→AR direction the released Critic was trained to handle, but it does
not test AV training itself. See README §5.3 for the limit of this claim.

The prompt is copied verbatim from paper's open-source data-generation
pipeline (`nla/datagen/stage2_api_explain.py`'s `_DEFAULT_INSTRUCTION`):
asks Claude to identify "2-3 features the LM is using to predict the next
token", in `<analysis>...</analysis>` tags. This is the format the released
Critic was trained against. Substituting the original "summarize in one
sentence" wording drops FVE to ≈0 (paper's Critic refuses the off-distribution
z's). See README §3.2 misalignment #2.

Usage (requires ANTHROPIC_API_KEY env var):
    python src/verbalizer_claude.py \\
        --activations-pt data/qwen7b_layer20_4bit_n200.pt \\
        --output data/qwen7b_paper_zs_n200.pt
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
import torch
from tqdm import tqdm


# Endpoint = OpenAI-compatible gateway used during my run. Swap for the
# direct Anthropic API by using https://api.anthropic.com/v1 + the official
# `anthropic` SDK if you prefer.
DEFAULT_BASE_URL = "https://api.penguinsaichat.dpdns.org/v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

PAPER_PROMPT = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"


def gen_claude_paper_z(prefix_text: str, *, api_key: str, base_url: str,
                       model: str, max_tokens: int = 300, max_retries: int = 3,
                       timeout: int = 60) -> str | None:
    """Single-shot Claude call. Returns None if the response is malformed
    (truncated before closing </analysis> or no <analysis> at all). Caller
    should drop None rows."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": PAPER_PROMPT.format(text=prefix_text)}
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{base_url}/chat/completions",
                                  headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            m = re.search(RESPONSE_PATTERN, raw, flags=re.DOTALL)
            return m.group(1).strip() if m is not None else None
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(2 ** attempt)
        except requests.HTTPError as e:
            last_err = e
            if e.response.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError(f"All retries failed: {last_err}")


def main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key is not None, "Set ANTHROPIC_API_KEY env var (or compatible gateway key)"

    # Load prefixes
    data = torch.load(args.activations_pt, weights_only=False)
    prefixes = data["prefix_for_claude"]
    print(f"✅ {len(prefixes)} prefixes ready")

    # Generate
    print(f"\nGenerating paper-format Claude z for {len(prefixes)} prefixes")
    print(f"  ~5–12 min, ~$0.20 with Haiku")
    paper_zs: list[str] = []
    failed = 0
    for prefix in tqdm(prefixes, desc="Claude paper-format"):
        z = gen_claude_paper_z(prefix, api_key=api_key, base_url=args.base_url,
                                model=args.model)
        if z is None:
            failed += 1
            paper_zs.append("")
        else:
            paper_zs.append(z)

    valid = sum(1 for z in paper_zs if z)
    print(f"\n✅ {valid}/{len(paper_zs)} valid (failed: {failed})")

    # Sanity preview
    for idx in (0, 10):
        if idx < len(paper_zs) and paper_zs[idx]:
            print(f"\nSample paper z [{idx}]:")
            print(f"  Prefix: {prefixes[idx][:120]}...")
            print(f"  z: {paper_zs[idx][:400]}")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "paper_zs": paper_zs,
        "failed_count": failed,
        "valid_count": valid,
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "prompt_source": "paper stage2_api_explain._DEFAULT_INSTRUCTION (verbatim)",
        },
    }, str(output))
    print(f"\n✅ Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-pt", required=True,
                        help="Output of extract_activations.py (contains prefix_for_claude)")
    parser.add_argument("--output", default="data/qwen7b_paper_zs_n200.pt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible gateway. For direct Anthropic, use https://api.anthropic.com/v1 with anthropic SDK")
    args = parser.parse_args()
    main(args)
