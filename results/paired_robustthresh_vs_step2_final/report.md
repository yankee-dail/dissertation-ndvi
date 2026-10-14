# Пряме парне порівняння: phen robustthresh (A) проти pure L1 (B)

Той самий test-набір (s2_america_test/ROIs1970/132, img-size=128, seed=0) для обох моделей - парний бутстреп по РІЗНИЦІ прогнозів на кожному патчі (n=270).

A = results/phen_grid_best_robustthresh/model.pt
B = results/step2_final/model.pt

Δ RMSE (A - B) = +0.0014 [+0.0008, +0.0019] - A значуще гірше за B
Δ SAM  (A - B) = +0.17° [+0.09, +0.26] - A значуще гірше за B
