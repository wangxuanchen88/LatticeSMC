# Model checkpoints

The trained checkpoints are not part of the anonymous supplementary material and are released at camera-ready:

- `runs/chunk_dispE/ckpt_adopted.pt`: the 4M-parameter chunked dance denoiser (EDGE representation, latent 192, 4 layers, 4 heads; `configs/chunk_dispE.yaml`, adopted step 12,000 by the lowest held-out monitor loss).
- `runs/chunk_dispE_L/ckpt_adopted.pt`: the 38M-parameter model (latent 512, 6 layers, 8 heads; `configs/chunk_dispE_L.yaml`, adopted step 9,000).
- `data/twist/twist_a002.pt`, `data/twist_rep/twist_a002.pt`, `data/twist_rep_L/twist_a002.pt`: the learned prefix twists of Appendix B (MLP 4638-512-512-1).

Training both dance models from the AIST++ data needs `configs/chunk_dispE.yaml` / `chunk_dispE_L.yaml` and `python -m lattice_smc.train` (0.9 and 5.0 GPU-hours on an A6000); the twists need `python -m lattice_smc.twist.collect` followed by `python -m lattice_smc.twist.train` (about 3 GPU-hours per reward). The music testbed uses the publicly released Stable Audio 3 (medium) weights and the LAION CLAP checkpoint `music_audioset_epoch_15_esc_90.14.pt`, both downloaded by their own tooling; no music model was trained.
