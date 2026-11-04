"""
Експеримент D2 (монографія, розд. 5.8 і 6.3): ансамбль п'яти моделей і калібрування в межах регіону
на шести нових тестових ділянках експерименту D. Без навчання.

ОКРЕМА сесія Colab з GPU; дані asiaWest і europa качаються заново (~33 ГБ, 1-1,5 год при ~10 МБ/с).
    1. Клітинка:  import os; from google.colab import userdata
                  os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    2. Клітинка:  !curl -sfL -o monograph_d2_ensemble.py \
                    https://raw.githubusercontent.com/yankee-dail/dissertation-ndvi/monograph-experiments/colab/monograph_d2_ensemble.py \
                    && python monograph_d2_ensemble.py

Ідемпотентний; результати - коміт і push у гілку monograph-d2-ensemble.
"""
import os
import sys
import time

REPO = "/content/dissertation-ndvi"
BRANCH = "monograph-d2-ensemble"
BASE = "monograph-experiments"


def main():
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN відсутній - встанови os.environ['GITHUB_TOKEN'] у клітинці ноутбука")
    remote = f"https://x-access-token:{token}@github.com/yankee-dail/dissertation-ndvi.git"
    if not os.path.isdir(f"{REPO}/.git"):
        if os.system(f"git clone -q -b {BASE} {remote} {REPO}") != 0:
            sys.exit("git clone впав - перевір GITHUB_TOKEN")
    else:  # повторний запуск: допоміжний модуль лежить лише в гілці BASE, не в master
        os.system(f"cd {REPO} && git fetch -q origin && git checkout -q origin/{BASE} -- colab/monograph_d_multiroi.py")
    sys.path.insert(0, f"{REPO}/colab")
    import monograph_d_multiroi as D  # завантаження й розпакування - як в експерименті D

    D.check_token(token)
    D.run("git fetch origin", cwd=REPO)
    if D.run(f"git rev-parse --verify origin/{BRANCH}", cwd=REPO, check=False) == 0:
        D.run(f"git checkout -B {BRANCH} origin/{BRANCH}", cwd=REPO)
        D.run(f"git merge --no-edit origin/{BASE}", cwd=REPO)
    else:
        D.run(f"git checkout -B {BRANCH} origin/{BASE}", cwd=REPO)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    raw = f"{REPO}/data/raw"
    os.makedirs(raw, exist_ok=True)
    for name in ("s2_asiaWest_test", "s2_europa_test"):
        for attempt in range(3):
            try:
                D.stream_extract(name, raw)
                break
            except (OSError, EOFError) as e:
                print(f"{name}: обрив потоку ({e}), спроба {attempt + 2}/3", flush=True)
                time.sleep(30)
        else:
            sys.exit(f"{name}: не вдалося завантажити")
    rois = []
    for name in ("s2_asiaWest_test", "s2_europa_test"):
        with open(f"{raw}/{name}.done") as f:
            rois += [r for r in f.read().split() if r]
    print("Ділянки:", rois, flush=True)

    D.run("python scripts/multi_roi_ensemble.py --root data/raw " + " ".join(f"--roi {r}" for r in rois)
          + " --out results/multi_roi_ens", cwd=REPO)
    D.run("git add -f results/multi_roi_ens", cwd=REPO)
    if D.run("git diff --cached --quiet", cwd=REPO, check=False) != 0:
        D.run('git -c user.email="colab@example.com" -c user.name="Colab GPU run" '
              'commit -m "Experiment D2: ensemble and within-region calibration on new test ROIs"', cwd=REPO)
        D.run(f"git push {remote} {BRANCH}", cwd=REPO, retries=3)
    print("Готово: результати в гілці", BRANCH)


if __name__ == "__main__":
    main()
