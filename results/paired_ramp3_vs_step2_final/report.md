# Пряме парне порівняння: phen ramp3 (A) проти pure L1 (B)

Той самий test-набір (s2_america_test/ROIs1970/132, img-size=128, seed=0) для обох моделей - парний бутстреп по РІЗНИЦІ прогнозів на кожному патчі (n=270).

A = results/phen_grid_best_ramp3/model.pt
B = results/step2_final/model.pt

Δ RMSE (A - B) = +0.0011 [+0.0008, +0.0014] - A значуще гірше за B
Δ SAM  (A - B) = +0.51° [+0.42, +0.59] - A значуще гірше за B
