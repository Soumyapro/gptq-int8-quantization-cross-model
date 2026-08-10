# GPTQ INT8 Quantization: Cross Model Comparison

Applying the GPTQ quantization technique, via the GPTQModel library, to two different LLM architectures and measuring whether INT8 quantization preserves model quality consistently across them.

The GPTQ algorithm itself (Hessian based, column by column, error compensated quantization) was not implemented from scratch for this project, since it involves linear algebra (inverse Hessian computation, Cholesky based numerics) beyond current scope. Instead, this project focuses on careful experimental design: proper calibration and evaluation data separation, consistent methodology across two architecturally different models, and honest, verified perplexity comparisons.

---

## Project Overview

Most quantization write ups report a single model's before and after numbers and stop there. This project asks a narrower but more useful question: **does GPTQ's INT8 quantization behave consistently when applied to two different model architectures, or is quality preservation specific to one model family?**

| | |
|---|---|
| **Models** | Qwen/Qwen2.5 1.5B Instruct and HuggingFaceTB/SmolLM2 1.7B Instruct (FP16 baselines) |
| **Quantization technique** | GPTQ (Hessian guided, column wise, error compensated), applied via the GPTQModel library |
| **Bit width tested** | INT8 only |
| **Eval dataset** | WikiText 2, test split |
| **Environment** | Google Colab, T4 GPU |

---

## What Was Built

### 1. Baseline Sanity Check

Both models were loaded in FP16 and run on an identical test prompt before any quantization was introduced, confirming both produced coherent, on topic output. This established a working baseline and confirmed the environment was correctly configured before adding quantization complexity.

### 2. Calibration Data Preparation

WikiText 2's train split was filtered to remove blank lines and section header artifacts (roughly 13,000 blank rows were present in the raw data). The cleaned text was tokenized separately for each model, since Qwen and SmolLM2 use different vocabularies, then split into 128 non overlapping chunks of 512 tokens each, matching the standard calibration size used in the original GPTQ paper and most library defaults. This produced two separate calibration sets, one per model, reformatted into the `input_ids` and `attention_mask` dictionary structure GPTQModel requires.

### 3. GPTQ Quantization

Each model was quantized to INT8 using GPTQModel, with symmetric quantization (`sym=True`, appropriate for weights, which are roughly zero centered) and a group size of 128. Quantization was run separately for each model using its own calibration set, and each quantized model was saved to disk before moving to the next, with explicit memory cleanup between runs to avoid exhausting GPU memory when working with two 1.5 to 1.7B parameter models in the same session.

### 4. Evaluation Data Preparation

WikiText 2's test split was filtered and tokenized using the same approach as calibration, but kept completely separate from the train split used for calibration to avoid data leakage. This produced 567 chunks of 512 tokens for Qwen's tokenizer and 580 chunks for SmolLM2's tokenizer (the same underlying text, tokenized differently per vocabulary), using the full filtered test split rather than a capped sample, for a stable perplexity estimate.

### 5. Perplexity Evaluation

All four model versions (Qwen FP16, Qwen INT8, SmolLM2 FP16, SmolLM2 INT8) were evaluated one at a time, with each model's cross entropy loss computed across every eval chunk and averaged, then converted to perplexity. Models were loaded, evaluated, and deleted from memory sequentially to keep GPU usage manageable throughout.

---

## Results

| Model | Perplexity | Change from FP16 |
|---|---|---|
| Qwen FP16 (baseline) | 12.3246 | — |
| Qwen INT8 (GPTQ) | 12.3352 | +0.0106 |
| SmolLM2 FP16 (baseline) | 11.4971 | — |
| SmolLM2 INT8 (GPTQ) | 11.4960 | −0.0011 |

GPTQ INT8 quantization produced perplexity changes small enough to be considered noise for both models, with neither architecture showing any measurable quality degradation at this bit width.

---

## Key Takeaways

- GPTQ's calibration driven, error compensated approach to INT8 quantization is essentially lossless for both models tested, consistent with GPTQ's design goal of using real activation statistics (via the Hessian) rather than raw weight magnitude alone to guide quantization decisions.
- This result held consistently across two architecturally different models (Qwen and SmolLM2), not just one, which is the core finding this project set out to check.
- Calibration and evaluation data must be drawn from separate splits and never mixed, otherwise perplexity numbers become an unreliable, overly optimistic measure of quantization quality.
  
---

## Repository Structure

```
gptq_int8_quantization_qwen_vs_smollm2.ipynb
README.md
```

---

## How to Run

Open the notebook in Google Colab with a GPU runtime (T4 is sufficient), run all cells top to bottom. Models and datasets are downloaded automatically via the `transformers`, `datasets`, and `gptqmodel` libraries. Quantized model checkpoints are saved locally during the notebook run and are not included in this repository due to file size.
