#!/bin/bash
# A6: σ-голова поверх замороженої моделі з y_ref = інтерполяція - діагностика калібрування і прапорець низької достовірності
PY=.venv312/bin/python
$PY scripts/calib_diagnostics.py --root data/raw --seeds 0 1 2 3 4 \
    --uncertainty-model "results/uncertainty_frozen_interp_gpu_256_seed{seed}/model.pt" --unc-ref-mode linear_interp \
    --out results/calib_diagnostics_frozen
for s in 0 1 2 3 4; do
  $PY scripts/validation_extras.py --root data/raw --step2-model results/step2_interp_gpu_256_seed$s/model.pt \
      --uncertainty-model results/uncertainty_frozen_interp_gpu_256_seed$s/model.pt --img-size 256 --seed $s \
      --ref-mode linear_interp --unc-ref-mode linear_interp --ref-cache-dir data/ref_cache \
      --out results/validation_extras/frozen_interp_gpu_seed$s
done
echo "$(date) A6 EVAL DONE"
