"""NLACritic — load the released Activation Reconstructor (AR) and score
text-to-vector reconstructions against gold activations.

Architecture (set by paper authors at training time, see `nla_meta.yaml`):
    - K+1 = 21 transformer layers truncated from Qwen2.5-7B-Instruct
    - lm_head replaced by Identity (Critic never emits logits)
    - final LayerNorm replaced by Identity (value-head reads raw residual)
    - Linear(d, d) value-head projects last-token residual to predicted activation

Loading invariants taken from the released `nla_meta.yaml`:
    - mse_scale = √d_model (≈ 59.87 for d = 3584). Both pred and gold are
      normalized to L2 = mse_scale before MSE, so per-element MSE = 2(1 − cos)
    - prompt template: "Summary of the following text: <text>{explanation}</text> <summary>"
    - extraction at tokens[-1] (the suffix " <summary>" anchors extraction)
    - tokenizer is added_special_tokens=True (Qwen has bos_token=None so this is a no-op)

The two MISSING-key warnings on load (`model.norm.weight`, `lm_head.weight`)
are expected — they are intentionally replaced with Identity above.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# Final LayerNorm attribute name varies by architecture. Qwen2/Llama/Mistral
# use "norm". GPT-2 style uses "ln_f". Some HF arches use "final_layernorm".
_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class NLACritic:
    """Loads the released NLA AR (Critic) checkpoint and scores reconstructions.

    Parameters
    ----------
    ckpt_dir : Path or str
        Local directory containing the downloaded Critic checkpoint.
        Must include nla_meta.yaml, value_head.safetensors, config.json,
        tokenizer files, and the model-*.safetensors shards.
    device : str
        cuda or cpu.
    dtype : torch.dtype
        bfloat16 fits in ~11 GB on T4.
    """

    def __init__(self, ckpt_dir: str | Path, *, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        ckpt_dir = Path(ckpt_dir)
        meta = yaml.safe_load((ckpt_dir / "nla_meta.yaml").read_text())
        assert meta.get("role") in ("ar", "critic"), \
            f"Expected role=ar/critic in sidecar, got {meta.get('role')!r}. " \
            f"Did you point this at the AV (actor) checkpoint by mistake?"

        ms = meta["extraction"]["mse_scale"]
        assert ms is not None, \
            "sidecar mse_scale is None — this Critic was not trained with normalization"
        self.mse_scale: float = float(ms)

        # The AR template ends with ' <summary>' — extraction happens at tokens[-1]
        # of `template.format(explanation=z)`.
        self.template: str = meta["prompt_templates"]["ar"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)

        # Load truncated backbone. config.json already says num_hidden_layers=21
        # (the K+1 truncation was done at training time), so HF loads only those layers.
        backbone = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=dtype, trust_remote_code=True,
        )
        # Strip lm_head and final LN — the Critic outputs raw residual + value_head
        backbone.lm_head = nn.Identity()
        for attr in _FINAL_LN_ATTRS:
            if hasattr(backbone.model, attr):
                setattr(backbone.model, attr, nn.Identity())
                break
        else:
            raise AssertionError(
                f"No final-LN attribute on {type(backbone.model).__name__} — "
                f"tried {_FINAL_LN_ATTRS!r}. Add this arch's attribute name."
            )

        # value_head is a separate safetensors file, not in the main shard
        d = backbone.config.hidden_size
        self.value_head = nn.Linear(d, d, bias=False, dtype=dtype)
        self.value_head.load_state_dict(load_file(str(ckpt_dir / "value_head.safetensors")))

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        self.d_model = d

        print(f"[NLACritic] {backbone.config.num_hidden_layers} layers  "
              f"d={d}  mse_scale={self.mse_scale:.2f}")

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """text z → predicted activation vector (raw, unnormalized).

        Returns a 1-D fp32 tensor on CPU, length = d_model.
        """
        prompt = self.template.format(explanation=explanation)
        ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        out = self.backbone(
            input_ids=ids["input_ids"].to(self.device),
            attention_mask=ids["attention_mask"].to(self.device),
        )
        # `out.logits` is the post-Identity output of the truncated backbone
        # (because we replaced lm_head with Identity), so it equals the
        # last-layer residual stream. We extract at tokens[-1] (the " <summary>" anchor).
        last_hidden = out.logits[0, -1, :]
        return self.value_head(last_hidden.unsqueeze(0)).squeeze(0).float().cpu()

    @torch.inference_mode()
    def score(self, explanation: str, gold_activation) -> tuple[float, float]:
        """text z + gold raw activation → (mse, cosine) on the unit-mse_scale sphere.

        Both pred and gold are L2-normalized to mse_scale, so MSE = 2(1 − cos).
        cosine ≈ 0 → orthogonal (random); cosine = 1 → perfect reconstruction.
        """
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(gold_activation, dtype=torch.float32).flatten()
        pred_n = pred / pred.norm() * self.mse_scale
        gold_n = gold / gold.norm() * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return mse, cos
