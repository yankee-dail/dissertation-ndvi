# Пряме парне порівняння: phen w=0.1 asym=2 (tuned) (A) проти pure L1 (step2_final) (B)

Той самий test-набір (s2_america_test/ROIs1970/132, img-size=128, seed=0) для обох моделей - парний бутстреп по РІЗНИЦІ прогнозів на кожному патчі (n=270).

A = results/phen_grid_best/model.pt
B = results/step2_final/model.pt

Δ RMSE (A - B) = +0.0013 [+0.0006, +0.0020] - A значуще гірше за B
Δ SAM  (A - B) = -0.10° [-0.19, -0.02] - A значуще краще за B
