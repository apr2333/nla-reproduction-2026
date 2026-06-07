# Reproducing Natural Language Autoencoders on a Small Open Model

A reproduction of Anthropic's *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* (Fraser-Taliente et al., Transformer Circuits, 2026), submitted for the [KTH ASSERT Lab PhD recruitment task (AI4Code)](https://github.com/ASSERT-KTH/phd-recruitment-2026-ai4code).

- Source paper: <https://transformer-circuits.pub/2026/nla/index.html>
- Author: Mian Qin (email: mianqin24@gmail.com)
- Compute: Google Colab Free, single Tesla T4 (16 GB)

## 1. Introduction

A Natural Language Autoencoder (NLA) is a pair of language-model components mapping a residual-stream activation `h_l` to a short text description `z` and back to a reconstructed vector `ĥ_l`. The Activation Verbalizer (AV) writes `z` from `h_l`; the Activation Reconstructor (AR) predicts `ĥ_l` from `z`. The round-trip squared error normalised by activation variance gives the Fraction of Variance Explained:

```
FVE = 1 − E‖h_l − ĥ_l‖² / E‖h_l − h̄_l‖²
```

Anthropic report FVE 0.6–0.8 on Claude-class models after joint reinforcement-learning training, and release four trained checkpoint pairs under [`kitft/nla-models`](https://huggingface.co/collections/kitft/nla-models). This work reproduces the round-trip evaluation on the smallest of those checkpoints, `Qwen/Qwen2.5-7B-Instruct` at layer 20. On 191 held-out positions of WikiText-103, the pipeline reaches FVE = 0.6437 (mean cosine = 0.64), within 0.11 of the paper's 0.75 figure on the same model with a fully RL-trained AV.

The rest of this README is organized as follows: 
**Section 2** documents the choice of target model. 
**Section 3** describes the working pipeline. 
**Section 4** reports the round-trip FVE results, sanity checks, and two qualitative cases. 
**Section 5** gives the reproducibility protocol. 
**Section 6** returns to the earlier training attempts and the discrepancies that reading the released source code revealed. 
**Section 7** discusses why the final pipeline works, what it can and cannot tell us, and threats to validity. 
**Section 8** collects what I learned beyond the core task.
**Section 9** sketches future work.

## 2. Choice of target model

The final pipeline targets `Qwen/Qwen2.5-7B-Instruct`, layer 20 of 28. Two considerations drove this choice. First, the released NLA Critic for Qwen-7B layer 20 ([`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar)) is the smallest open NLA-trained checkpoint available, providing a comparable reference number. Second, at 4-bit nf4 quantisation Qwen-7B occupies approximately 5.6 GB on T4, leaving 10 GB for inference; fp16 (the paper's setup) requires about 14 GB and triggers OOM.

I attempted Pythia-70M and SmolLM2-135M first with the goal of training NLA from scratch. Section 6 reports why neither produced positive FVE under decoupled supervised training, which motivated the change from "train" to "obtain"—a path the task description allows.

## 3. Method

For each WikiText-103 segment I run a forward pass through Qwen-7B (4-bit nf4, `bnb_4bit_compute_dtype=bfloat16`) and extract the layer-20 residual activation at token position 60, stored raw without normalisation. Position 60 satisfies the paper's `_MIN_POSITION = 50` requirement. Of 200 candidate segments, 191 are long enough at position 60.

For each retained activation I form the token prefix up to position 60 and submit it to Claude 3.5 Haiku with the verbatim prompt template from `nla/datagen/stage2_api_explain.py` in the released repository. The prompt asks Claude to identify "the 2–3 most important features the language model would use to predict what comes next" and to enclose its answer in `<analysis>...</analysis>`, with the final feature describing the role and immediate constraints of the last token. All 191 calls returned a parseable block.

The AR is the released Critic checkpoint loaded as-is: a 21-layer truncation of Qwen-7B with `lm_head` and final LayerNorm replaced by `Identity`, plus a `Linear(3584, 3584)` value-head loaded from `value_head.safetensors`. The released `nla_meta.yaml` specifies the AR prompt template (`"Summary of the following text: <text>{explanation}</text> <summary>"`) and `mse_scale = √3584 ≈ 59.87`. Reconstruction takes the last-token residual of the truncated backbone, applies the value-head, and returns a 3584-dimensional vector. Both predicted and gold activations are L2-normalised to magnitude `mse_scale`, after which their MSE equals `2(1 − cos)`. FVE uses the orthogonal random pair as variance reference: `FVE = 1 − E[MSE] / 2`.

The full pipeline lives in [`src/`](src/): [`download_critic.py`](src/download_critic.py), [`extract_activations.py`](src/extract_activations.py), [`verbalizer_claude.py`](src/verbalizer_claude.py), [`nla_critic.py`](src/nla_critic.py), [`score_critic.py`](src/score_critic.py).

Two components of original paper's pipeline are deliberately not reproduced. The AV is not retrained, since joint AV–AR RL with GRPO on 2×8×H100 is infeasible on Colab Free; the substitution by a Claude-prompted verbaliser is the closest within-budget proxy that still draws z's from the distribution the released Critic was trained against, and a single small-scale REINFORCE attempt (Section 6) is reported as evidence that the gap to in-budget RL is not just a tuning question. 
The Critic is not retrained either, since the released checkpoint already incorporates the joint-RL reward signal that supervised training cannot recover (Section 6.1); using it as-is preserves the paper's training distribution and makes the resulting FVE numbers directly comparable. 

## 4. Results

### 4.1 Round-trip evaluation on Qwen-7B layer 20

Table 1 summarises the round-trip evaluation on N = 191 held-out WikiText-103 positions. The mean cosine of 0.6437 corresponds to FVE = 0.6437. The distribution is approximately Gaussian around the mean with std/mean = 0.36; cosines range from 0.34 to 0.91, so no single sample dominates the average. The full distribution is in [`figures/path_x_distribution.png`](figures/path_x_distribution.png).

**Table 1.** Round-trip evaluation. Vectors L2-normalised to `mse_scale` before comparison.

| Statistic | Value |
|---|---|
| N | 191 |
| Mean / median MSE | 0.7127 / 0.6960 |
| Mean / median cosine | 0.6437 / 0.6520 |
| Cosine min / max | 0.3382 / 0.9059 |
| **FVE** | **0.6437** |

### 4.2 Sanity checks on the released Critic

Three checks confirm correct loading. Scoring a paper-format z against a random gold vector returns cos = −0.022 and MSE = 2.044, within sampling noise of the orthogonal-pair expectation. Scoring a z against the AR's own reconstruction of that z returns cos = 1.0000 exactly. The bf16-loaded Critic occupies 10.96 GB, consistent with the 21-layer Qwen-7B truncation plus value-head.

### 4.3 Two qualitative cases

[`figures/qualitative_examples.txt`](figures/qualitative_examples.txt) contains the highest- and lowest-cosine cases. The highest (cos = 0.91) is the token "early" in a Cicely Mary Barker biography, where the prefix ends "She admitted a fondness for the early"; the Claude z explicitly identifies "early" as an adjective requiring an art-historical noun, and the Critic recovers a residual consistent with that constraint. A medium case (cos = 0.65) is the token "work" after "Development", where the Claude z reads it as an incomplete predicate but the Critic recovers a vector also encoding domain priors (game-development) that the syntactic-frame description leaves out. Across the 191 cases, high-cosine z's name a concrete syntactic-semantic constraint at the final token; lower-cosine z's stop at a generic syntactic frame.

## 5. Reproducibility

### 5.1 Environment

Python 3.12, CUDA 12.x, Tesla T4 (16 GB). Dependencies in [`requirements.txt`](requirements.txt).

### 5.2 Pipeline

```bash
# 1. Download released Critic (~11 GB)
python src/download_critic.py

# 2. Extract Qwen-7B 4-bit layer-20 activations on 200 prefixes (~5 min)
python src/extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct --layer 20 --position 60 \
    --num-segments 200 \
    --output data/qwen7b_layer20_4bit_n200.pt

# 3. Generate paper-format Claude z's (~12 min, ~$0.20)
python src/verbalizer_claude.py \
    --activations-pt data/qwen7b_layer20_4bit_n200.pt \
    --output data/qwen7b_paper_zs_n200.pt

# 4. Score and report FVE (~3 min)
python src/score_critic.py \
    --critic-dir data/nla_critic_qwen7b/ \
    --activations-pt data/qwen7b_layer20_4bit_n200.pt \
    --zs-pt data/qwen7b_paper_zs_n200.pt \
    --output results/path_x_fve.pt
```

End-to-end wall-clock on T4: approximately 25 minutes. Expected `mean_cos ≈ 0.64`, `fve ≈ 0.64`.

### 5.3 Repository layout

```
nla-reproduction-2026/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── download_critic.py
│   ├── extract_activations.py
│   ├── verbalizer_claude.py
│   ├── nla_critic.py
│   └── score_critic.py
└── figures/
    ├── path_x_distribution.png
    └── qualitative_examples.txt
```

`.pt` files generated by the pipeline are excluded via `.gitignore`; running the four commands above regenerates them.

## 6. Earlier attempts and what they revealed

### 6.1 Thirteen training attempts on smaller models

I attempted to train NLA from scratch on Pythia-70M and SmolLM2-135M before turning to the released checkpoint. None of the thirteen runs reached positive FVE (Table 2). The pattern is consistent across two base models, three AR architectures, and two training paradigms: FVE plateaus between −0.02 and +0.01. Changing model size, training data scale, pooling, encoder choice, fine-tuning depth, and AR architecture did not move it.

**Table 2.** Thirteen training attempts that preceded the final pipeline.

| Day | Configuration | val FVE | Failure mode |
|---|---|---|---|
| 2 | Pythia-70M AV + Pythia layer 3 + LinearAR | −0.0700 | Pythia-70M as AV outputs degenerate "X is a Y" summaries |
| 2 | Claude AV (one-sentence) + Pythia + LinearAR | −0.0025 | Pythia-encoded z's collapse (cos = 0.978) |
| 2 | Claude AV + Pythia layer 1 mean + LinearAR | +0.0100 | Layer/pooling sweep winner; FVE still ≈ 0 |
| 2 | Claude AV + sbert encoder + LinearAR | −0.0910 | sbert separates z's well (cos = 0.047); cross-space mapping fails on 800 samples |
| 3 | SmolLM2-135M truncated AR (no output norm) | −192.94 | Output magnitude blow-up |
| 3 | SmolLM2 truncated AR (unit-norm output) | −0.1914 | Forcing unit-norm makes FVE structurally negative |
| 3 | SmolLM2 5k LinearProbe | ≈ −0.03 | Mode-collapse to WikiText priors |
| 3 | SmolLM2 5k Fine-tune-last-1 | −0.0166 | Marginal |
| 4 | Mini-NLA AV SFT v1 (134M trainable) | val loss 3.50; 5/5 generations off-topic | "Aston Villa" mode collapse; AV regressed to Wikipedia priors |
| 4 | Mini-NLA AV SFT v2 (last 4 layers, 42M) | val loss 3.78 | Same collapse, less overfit |
| 5 | AR initial SFT, 5 epochs | −0.0006 | Closest pre-final attempt to zero |
| 5 | AR initial SFT, 10 epochs + cosine LR | −0.0002 | Approaches but does not cross zero |
| 5 | REINFORCE (250 steps, interrupted) | −0.0157 | Reward signal too weak; AV mode-collapsed |

The shared failure mode is that supervised cross-entropy against external ground-truth z's does not penalise an AV that bypasses the activation and outputs the most probable z under the training distribution. The AV minimises loss most efficiently by learning the prior over z, not by reading h. The Day 4 "Aston Villa" generations are concrete evidence: held-out activations from Wikipedia articles on World War II battles produced AV outputs about English football, because football is high-frequency in the Wikipedia training distribution. A separate layer × pooling sweep over the Pythia-70M z encoder confirmed that the bottleneck is not encoding capacity (mean pairwise cosine across 1000 z embeddings did not drop below 0.892 at any layer or pooling choice); it is the loss objective itself.

This is the structural problem the paper's RL stage is designed to fix, with reward `−mse_nrm(AR(AV(h)), h)` rewarding only z's that the AR can decode back to h. Without that reward, AV faithfulness is not enforced.

### 6.2 Six discrepancies surfaced by reading the released code

After thirteen consecutive non-positive results, I cloned the released codebase and walked through `nla_inference.py`, `nla/reward.py`, `nla/datagen/stage2_api_explain.py`, and the published sidecar files. Six discrepancies between my implementation and the actual NLA recipe surfaced (Table 3).

**Table 3.** Discrepancies between my Day 1–5 implementation and the released NLA recipe.

| # | My implementation | Released recipe |
|---|---|---|
| 1 | AV reads source text, writes summary | AV reads injected activation vector as a single-token embedding via `<INJECT>` placeholder |
| 2 | "Summarise in one sentence" | "Identify 2–3 features the LM uses to predict the next token", in `<analysis>` tags |
| 3 | Activations pre-normalised to unit L2 at extraction | Inputs raw; normalisation at injection (`injection_scale=150`) and at loss (`mse_scale=√d`) only |
| 4 | "z encoder + MLP head" | K+1 truncated LM with `Linear(d,d)` value-head, extracted at the last token after a `<summary>` suffix |
| 5 | Paragraph-level `h_l` | Per-token `h_l` from positions ≥ 50 |
| 6 | `injection_scale` set by magnitude-matching | Per-model learned constant from sidecar (Qwen=150); wrong value silently produces CJK output |

About half of these discrepancies are determinable from the paper text; the other half are only visible in code—the sidecar invariants, the `Identity` replacement of `lm_head`, the suffix-anchored extraction at the last token, the magnitude of `injection_scale`. Discrepancies 2–6 are corrected in the Section 3 pipeline. Discrepancy 1 is not: my AV substitutes a text-conditioned Claude call for the trained vector-conditioned NLA AV.

## 7. Discussion

### 7.1 Why the final pipeline works

In a supervised setup, an AV trained against external z's minimises cross-entropy most efficiently by approximating the marginal distribution of z, not by reading h (Section 6.1). The paper's RL stage closes this gap by making the loss depend on h through both AV and AR. The final pipeline sidesteps the problem by using the released Critic, whose training already incorporated this reward, and by drawing z's from a Claude prompt designed in the released codebase to match the Critic's training distribution.

The pipeline tests one specific question: given a z drawn from the distribution the Critic was trained against, does the released Critic reconstruct the corresponding Qwen-7B activation? The answer is yes, with FVE = 0.6437. The pipeline does not test whether an in-house AV can learn to read activations from scratch on a small model; for that, the thirteen runs in Section 6 collectively answer "no" for the supervised-only setup. They do not answer "no" for joint RL, because joint RL was not run within the time budget.

### 7.2 Threats to validity

**External.** Numbers are obtained on a single base model, single layer, single Claude version. Generalisation to other model families, layers, or AVs is not tested.

**Sample size.** N = 191 is small; the paper's eval set is at least an order of magnitude larger. Confidence intervals on FVE are wide.

**Quantisation.** 4-bit nf4 is a precision regime the released Critic was not trained against; the fp16-vs-4-bit ablation could not be run on T4. The 0.07–0.11 gap to the paper's 0.75 plausibly decomposes as roughly 0.05 from quantisation, 0.05 from the smaller eval set, and the residual from AV substitution; this decomposition is unverified.

**AV substitution.** The most consequential threat. My round-trip uses Claude (text → text) where the released NLA AV is vector-conditional. The FVE number reflects "Critic reconstruction quality given a properly distributed z", not "AV+AR round-trip given a vector".

**"Small model".** The KTH brief specifies a small open model; Qwen-7B is at the small end of the paper's checkpoints but is not small in absolute terms. The smallest model with a released NLA Critic is 7B; below 7B the thirteen runs in Section 6 indicate that training one from scratch under supervised-only conditions does not converge.

## 8. What I learned

Joint training in NLA appears to be a structural requirement, not an optimisation convenience. Supervised training does not penalise the AV for ignoring its input; whatever objective is used has to make the loss a function of the activation through both AV and AR. The paper's reward signal is the simplest construction that does this.

The released code is part of the specification. About half of the discrepancies in Section 6.2 are unrecoverable from the paper text alone. The `injection_scale` constant for Qwen (150) is stated nowhere in the paper but is mandatory in the sidecar; getting it wrong silently produces CJK-only output. The truncated-LM AR with stripped final norm and a separate value-head is specified by the released `config.json`, not the paper.

The "obtain" path in the task brief is a deliberate design choice. After Day 5 the supervised-only training was clearly stuck and the budget for RL was not there. Switching to the released Critic with a Claude AV substitution recovered a paper-comparable FVE in roughly half a day. The honest framing is that I obtained the AR and substituted the AV; the substitution is a real limitation, but the alternative within the time budget was a non-positive FVE on a smaller model.

## 9. Future work

A fp16 vs 4-bit baseline on a single A100-hour would isolate the quantisation component of the gap. A layer ablation (5/10/15/20/25) at fixed Critic would describe how Critic generalisation decays for off-distribution layers. A replacement of the Claude verbaliser with a small in-house AV trained via REINFORCE-with-baseline plus KL anchoring against a frozen reference would test whether the supervised-training failure mode of Section 6 is reachable with limited RL compute.

## Citation

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

NLA codebase and released checkpoints by Kit Fraser-Taliente, Apache-2.0: <https://github.com/kitft/natural_language_autoencoders>.
