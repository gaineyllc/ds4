# DeepSeek V4.1 Flash 1017 UNCENSORED, imatrix Q2 for ds4

The model files this fork's Mac setup runs. They load only in ds4. Use this fork's
`rank/1017` branch: the server tuning and the DSpark runtime it relies on are not in
upstream ds4. llama.cpp and other GGUF runtimes cannot load these files; the tensor
layout is ds4's own.

## Files

| File | Size | What it is |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-UNC-Q2.gguf` | 365,713,686,528 B (340.6 GiB) | Main model. Routed experts IQ2_XXS (gate/up) and Q2_K (down); attention, shared experts and output head Q8_0. |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8,633,942,016 B | DSpark speculative-decoding support model, 3 stages, block 5, native MXFP4 weights. |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 970,555,584 B | Vision encoder for image input. |

Checksums are in `SHA256SUMS`, served next to the files.

## Provenance
- Source checkpoint: [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8), revision `c19ca5ca4eaeb79d29fdf08aea8327da95b1116a`. That is an uncensored release of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
  - The GGUF metadata field `general.source.url` names the deepseek-ai base; the revision is dealignai's.
- Quantization: `gguf-tools/deepseek41_quantize.py --quant q2`, guided by an importance matrix (imatrix) for the routed experts.
  - The imatrix was computed on this model, with the V4.1 calibration prompts rendered through the model's own chat template, run at up to 1M context.
  - The imatrix file is `1017-unc-routed-moe-1m.dat`; its name is recorded in the GGUF metadata.
- DSpark and Vision were converted from the same checkpoint with `gguf-tools/deepseek41_dspark.py` and `gguf-tools/deepseek41_vision.py`.

## Quality and speed
Measured on an M5 Max with 128 GB RAM, Oct 2026, using this fork:
- Decode: about 12 tok/s plain and 17 tok/s with DSpark, on short-context code and prose (warm, temperature 0).
- Prefill: 260–350 tok/s for long prompts.
- With DSpark at temperature 0, output matches plain decoding on most prompts. Batched verification can rarely flip a near-tie token.
- Other Macs are untested; expect lower numbers on M3/M4. `scripts/mac/verify.sh` measures yours.

## License and terms
- MIT, the same as the base model and dealignai's release.
- dealignai asks for credit from anyone who redistributes, quantizes, serves or writes about the model. This file is that credit.
- Their release is gated behind terms (18+, comply with local law, the user is responsible for outputs). The same terms apply to these files.
- This is an uncensored model. It can produce content the base model refuses. No warranty.
