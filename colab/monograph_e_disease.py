"""
Експеримент E (монографія, розд. 10.3-10.4): розпізнавання хвороб за фото листка, перенесення з лабораторних
знімків PlantVillage на польові PlantDoc, конформні множини класів. Потрібен GPU (Runtime -> Change runtime
type -> T4 або краще). Час: близько 1-1,5 год на T4 (п'ять запусків).

Одна клітинка, токен не потрібен (репозиторій публічний):
    !curl -sfL -o monograph_e_disease.py \
      https://raw.githubusercontent.com/yankee-dail/dissertation-ndvi/monograph-experiments/colab/monograph_e_disease.py \
      && python monograph_e_disease.py

Після кожного запуску (seed) друкується проміжна таблиця; наприкінці - блок «ПІДСУМОК», його треба скопіювати
повністю. Якщо сесія обірвалась, повторний запуск тієї самої клітинки в ТІЙ САМІЙ сесії продовжить з
наступного seed (дані й готові seed лишаються в /content).
"""
import subprocess
import sys

RAW = "https://raw.githubusercontent.com/yankee-dail/dissertation-ndvi/monograph-experiments/scripts/disease_transfer.py"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"curl -sfL -o /content/disease_transfer.py {RAW}")
sh(f"{sys.executable} -m pip install -q timm tensorflow-datasets")
sh(f"{sys.executable} /content/disease_transfer.py --work /content/work "
   f"--out /content/results/disease_transfer --seeds 0 1 2 3 4")
