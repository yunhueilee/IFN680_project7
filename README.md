# Project 7 — Improving DDPM (Linear vs. Cosine Schedule)

## Files

- **`ddpm_utils.py`** — shared library: dataset wrapper, both beta schedules, the conditional U-Net, the training loop, the DDPM sampler (with accelerated/respaced sampling for the fewer-steps experiment), checkpointing, and the auxiliary evaluation classifier. Imported by both notebooks so the two schedule runs stay strictly comparable.
- **`DDPM_CosineSchedule.ipynb`** — trains the Linear-schedule and Cosine-schedule conditional DDPMs (identical architecture/hyperparameters, schedule is the only difference) and the auxiliary classifier. Saves everything to `checkpoints/`.
- **`main_report.ipynb`** — reproducibility/inference notebook. Loads the checkpoints only (no training), reproduces the required 12×10 sample grids, the fewer-steps hypothesis test, and every number/figure used in the report. Saves figures to `figures/`.
- **`checkpoints/`** — populated by `DDPM_CosineSchedule.ipynb`: `linear_ddpm.pt`, `cosine_ddpm.pt`, `classifier.pt`, `linear_train_log.pkl`, `cosine_train_log.pkl`.
- **`figures/`** — populated by both notebooks; these PNGs are the ones used in `Project7_Report.docx`.

## Submission contents (`project7_code.zip`)

The zip submitted on Canvas contains exactly:

```
DDPM_CosineSchedule.ipynb
main_report.ipynb
ddpm_utils.py
mnist_custom.pt
checkpoints/  (linear_ddpm.pt, cosine_ddpm.pt, classifier.pt, linear_train_log.pkl, cosine_train_log.pkl)
figures/
README.md
```

`mnist_custom.pt` is included even though it's the teacher-provided dataset, because both notebooks load it by relative path (`DATA_PATH = "mnist_custom.pt"`) and the grader's environment won't otherwise have a copy next to the notebooks — per the brief, "failure on any of the cells will result in a 0 mark." The instruction/rubric PDFs and `Week_9_solution.ipynb` are excluded; they aren't part of the deliverable.

## How to run (QUT GPU environment)

1. Upload this whole folder to the QUT Jupyter platform (same environment as the 9.3 Tutorial). `mnist_custom.pt` must sit next to the notebooks.
2. Run `DDPM_CosineSchedule.ipynb` top to bottom. This trains both models — the only thing you may want to adjust for your GPU time budget is `epochs` (Sections 4/5) and `timesteps` (Section 2). It populates `checkpoints/`.
3. Run `main_report.ipynb` top to bottom. It only does inference/evaluation (no training), so it's much faster than step 2. It populates `figures/` and prints every number used in the report.
4. Both notebooks in this submission were run end-to-end already and carry their saved outputs (150 epochs, `timesteps = 1000`); re-running from a fresh kernel should reproduce the same checkpoints/figures up to floating-point/GPU noise, since every random seed is fixed.

## Design notes

- `mnist_custom.pt` stores 20×20 images (not the standard torchvision 28×28 MNIST), pixel values in `[0,1]`; `ddpm_utils.MNISTCustomDataset` rescales to `[-1,1]`.
- Both models share one `UNetCond` implementation and one `train_ddpm`/`sample_ddpm` implementation — only the `NoiseSchedule` passed in differs, so the schedule is the sole experimental variable.
- The fewer-steps experiment in `main_report.ipynb` uses `U.sample_ddpm(..., num_steps=K)`, which builds a "respaced" schedule over `K` of the original `timesteps` (Nichol & Dhariwal, 2021) rather than retraining — this is what lets the notebook directly test the project's hypothesis about sampling-step count.
- The `DigitClassifier` in `ddpm_utils.py` is not part of either diffusion model; it's a small CNN trained once on real digits, used only to quantitatively score how recognisable generated samples are.
