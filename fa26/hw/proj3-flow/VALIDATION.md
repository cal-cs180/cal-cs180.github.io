# Flow Part A validation — 2026-09-21

## Scope and provenance

Implemented all sections of `FM_Proj_A.pdf` using the specified
`MCG-NJU/PixNerd-XXL-P16-T2I` checkpoint. See README for pinned model, encoder,
and source revisions. The official scheduler and training target confirm that
this is linear flow matching, in RGB pixel space, with time running from noise
at 0 to image at 1. The strict EMA checkpoint load completed without missing
or unexpected parameters.

These historical results used bfloat16 model evaluation. The current adapter uses float32
throughout and requires new runtime/memory measurements. In the historical runs,
Euler states, interpolation and guidance
arithmetic remain float32. Native resolution is 512×512. No VAE, DeepFloyd
stages, alpha-cumprod schedule, DDPM variance term, or 64×64 upsampling stage
remains in the revised Part A.

## Execution

All four section runs completed successfully in independent notebook kernels
on physical GPUs 4, 5, 6, 7 of `dgx5` (A100-SXM4-80GB), using PyTorch 2.7.1+cu126.
Each executed 14 code cells without an error output; group guards selected the
relevant expensive examples. The four executed notebooks are in `results/`.
The complete sequential run also passed all 14 code cells, with zero error outputs,
in 219.9 seconds (including model setup). It is saved as
`results/parta_all.executed.ipynb` and copied to `parta_soln.ipynb` for convenient review.
Its peak allocated memory was 5.825 GiB. All figures and checks ran in a single kernel
with `FLOW_GROUP=all`, so this verifies cell-order dependencies in addition to the split runs.

Settings: seeds 180–184 for the five sampling examples; Euler h=0.02; CFG w=7
in the proposal's conditional-plus-difference convention (upstream scale 8).
No timestep shift, guidance clipping, or rescaling was silently added.

The runs cover five ordinary conditional samples and five CFG samples; six
starting times on six image/drawing fixtures; three inpainting examples;
three text-guided editing examples at six times; two anagrams and two hybrids.
Fixtures are staff demonstrations, not student-submitted original work.

Checks that passed:

- Interpolation at 0 equals the supplied noise; at 1 equals the input image.
- Velocity preserves the input tensor's shape, dtype, device and is finite.
- Euler takes 50 updates from 0 and 25 from 0.5 at h=0.02; a 0.93 start
  correctly shortens the last of four steps and reaches 1 exactly.
- Constant-field Euler integration matches the analytic endpoint.
- CFG w=0 equals conditional prediction.
- Every Euler state is finite.
- The unmasked inpainting region has exactly zero maximum error on all three fixtures.
- Applying the 180-degree rotation twice recovers the original tensor exactly.

Peak allocated CUDA memory in section runs was approximately 5.7–5.8 GiB per GPU.
This is measured tensor allocation, not total driver reservation or a guarantee
that an untested hardware/software environment will work.

## Numerical versus visual results

Campanile reconstruction MSE in normalized RGB at t=0.50:

| Method | MSE |
| --- | ---: |
| Best Gaussian blur in the tested sweep | 0.0242174 |
| One-step flow endpoint estimate | 0.00848592 |
| 25 Euler updates | 0.00997104 |

Euler restores visible fine detail, but does **not** improve pixel MSE over the
one-step estimate in this example. The assignment should ask for comparison,
not claim iterative denoising always scores better.

The generic prompt "a high quality photo" generated portraits in all five CFG
examples. This prompt is not semantically neutral for this checkpoint. Text-guided
Campanile-to-rocket editing works visibly, with weaker changes at larger start times.
The old-man/campfire and village/horse anagrams exhibit both orientations, with
imperfections typical of the task.

Inpainting is numerically correct outside the mask, but the baseline Campanile
fill has a conspicuous boundary and incoherent content. Five further seeds with
a top-reaching mask still frequently generate faces or unrelated forms. Additional
guidance/context-prompt checks are saved under `results/quality_checks/inpaint_tuning/`.
These are labeled departures from the proposal's generic prompt/default guidance.
The contextual clock-tower prompt improves subject relevance but still produces a
rectangular pasted-looking region. The tested w=0,3,7 settings do not resolve the seam.
This remains a limitation; it is not presented as a successful context-consistent fill.

The default waterfall/skull and mountain/wolf hybrid examples favor the high-frequency
subject. A further sweep of three prompt pairs at sigma 4, 8, 16 (seed 182, w=7,
h=0.02) often changes which single subject dominates, rather than producing a clear
two-scale illusion. The extra portrait pair also fails to establish a reliable
viewing-distance switch. Do not advertise these as verified successful hybrids.
All sweep images and parameter records are retained in `results/quality_checks/hybrid/`.

## Teaching recommendation

The checkpoint is suitable for demonstrating flow interpolation, Euler sampling,
CFG, and text-guided editing. The inpainting and hybrid deliverables need further
pedagogical tuning before publishing this as a tested assignment: consider a
contextual prompt for inpainting and make hybrids an exploratory exercise with
explicit failure analysis until reliable settings/prompt pairs are found.
The requested defaults remain intact in the notebook and specification.

The upstream neural-field positional embedding emits a complex-to-real cast warning.
This comes from the pinned upstream implementation; it did not cause nonfinite outputs
or execution failures. Its numerical behavior was preserved for checkpoint compatibility.
