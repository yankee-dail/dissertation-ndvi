"""
Експеримент A6 (монографія, розд. 6.6): σ-голова поверх ЗАМОРОЖЕНОЇ залишкової моделі з y_ref = лінійна
інтерполяція (results/step2_interp_gpu_256_seed*). μ не змінюється, навчається лише log σ, тож σ оцінена
саме для того μ, яке отримує користувач. GPU, 256x256, 20 епох (вчиться лише 1x1-голова), 5 seed;
вибір епохи asiaEast/119 (не калібрувальна ділянка), калібрування asiaEast/139, тест america/132.

ВИКОРИСТАННЯ в Colab (GPU-рантайм T4), після monograph_a0_zeroinit.py або в окремій сесії:
    1. Клітинка:  import os; from google.colab import userdata
                  os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    2. Клітинка:  !curl -sfL -H "Authorization: token $GITHUB_TOKEN" -o monograph_a6_frozen_sigma.py \
                    https://raw.githubusercontent.com/yankee-dail/dissertation-ndvi/monograph-experiments/colab/monograph_a6_frozen_sigma.py \
                    && python monograph_a6_frozen_sigma.py

Ідемпотентний; після кожного seed - коміт і push у гілку monograph-a6-frozen-sigma.
"""
import os
import re
import subprocess
import sys
import time

REPO = "/content/dissertation-ndvi"
BRANCH = "monograph-a6-frozen-sigma"
BASE = "monograph-experiments"  # гілка, де є прапорець --init-from-step2
SEEDS = [0, 1, 2, 3, 4]
ARCHIVES = {
    "s2_america_test": "19c7a11d81c1a574f5876ec8e21dfec7f101677704021a6191146e59625670419e8606aa1efb5a9f39c74700a4830875f87025ce4b8a9de9e6fcbf35d233276f",
    "s2_africa_test": "93b2c305541e5e075c7300e7c65e8b173bbe3a937a959ec7f57c0e06b57ddb768f032ea7009a11d5aca0c5d7071f9502182c17269e2c5e6ffe40e201774b553b",
    "s2_asiaEast_test": "0c6265aa1fe2bb88c5a05e4fa0335cf50950efc0da9e4bc62677f40f958d1acbfba4fb039e78b0071df97eb9bd39efa628b427a620716849aa8d99f8dd2f634d",
}


def mask(text):
    return re.sub(r"(ghp_|github_pat_)[A-Za-z0-9_]+", r"\1***", text)


def run(cmd, cwd=None, check=True, retries=1):
    """stdout+stderr об'єднано й друкується явно: при запуску через %run/exec у ядрі
    Colab stderr підпроцесу інакше не потрапляє в клітинку (не видно 'fatal: ...')."""
    for attempt in range(1, retries + 1):
        print(f"$ {mask(cmd)}" + (f"  (спроба {attempt}/{retries})" if retries > 1 else ""), flush=True)
        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(mask(line), end="", flush=True)
        if proc.wait() == 0:
            return 0
        if attempt < retries:
            time.sleep(20)
    if check:
        sys.exit(f"Команда впала (код {proc.returncode}): {mask(cmd)}")
    return proc.returncode


def check_token(token):
    """Пояснює, ЧОМУ git не може авторизуватись, замість туманного 'could not read Username'."""
    import json
    import urllib.error
    import urllib.request
    req = urllib.request.Request("https://api.github.com/repos/yankee-dail/dissertation-ndvi",
                                 headers={"Authorization": f"Bearer {token.strip()}",
                                          "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req) as r:
            perms = json.load(r).get("permissions", {})
    except urllib.error.HTTPError as e:
        hints = {401: "токен недійсний, прострочений або відкликаний - створи новий і онови Colab Secret",
                 403: "токену заборонено доступ (SSO/обмеження організації чи rate limit)",
                 404: "токен валідний, але НЕ має доступу до цього приватного репо - для fine-grained токена "
                      "додай yankee-dail/dissertation-ndvi у 'Repository access' і дай Contents: Read and write"}
        sys.exit(f"Перевірка GITHUB_TOKEN: HTTP {e.code} - {hints.get(e.code, e.reason)}")
    if not perms.get("push"):
        sys.exit("GITHUB_TOKEN читає репо, але без права запису - потрібен Contents: Read and write (або scope 'repo')")
    print("GITHUB_TOKEN: доступ до репо з правом запису - ок", flush=True)


def main():
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN відсутній - встанови os.environ['GITHUB_TOKEN'] у клітинці ноутбука")
    check_token(token)
    remote = f"https://x-access-token:{token}@github.com/yankee-dail/dissertation-ndvi.git"

    if os.path.exists(REPO) and not os.path.isdir(f"{REPO}/.git"):
        aside = f"{REPO}_old_{int(time.time())}"
        print(f"{REPO} існує без .git - переношу в {aside}")
        os.rename(REPO, aside)
    if not os.path.isdir(f"{REPO}/.git"):
        run(f"git clone {remote} {REPO}")
    run("git fetch origin", cwd=REPO)
    if run(f"git rev-parse --verify origin/{BRANCH}", cwd=REPO, check=False) == 0:
        run(f"git checkout -B {BRANCH} origin/{BRANCH}", cwd=REPO)
    else:
        run(f"git checkout -B {BRANCH} origin/{BASE}", cwd=REPO)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("Немає GPU - Runtime -> Change runtime type -> T4 GPU")

    raw = f"{REPO}/data/raw"
    os.makedirs(raw, exist_ok=True)
    for name, sha in ARCHIVES.items():
        if os.path.isdir(f"{raw}/{name}"):
            print(f"{name} вже розпаковано - пропускаю")
            continue
        run(f'curl --fail --silent --show-error -o {name}.tar.gz "ftp://m1659251:m1659251@dataserv.ub.tum.de/{name}.tar.gz"',
            cwd=raw, retries=6)
        run(f'echo "{sha}  {name}.tar.gz" | shasum -a 512 -c -', cwd=raw)
        run(f"tar -xzf {name}.tar.gz && rm {name}.tar.gz", cwd=raw)
    run("df -h /content")

    for seed in SEEDS:
        out = f"results/uncertainty_frozen_interp_gpu_256_seed{seed}"
        if os.path.exists(f"{REPO}/{out}/model.pt"):
            print(f"seed {seed} вже є в гілці - пропускаю", flush=True)
            continue
        run("python scripts/train_step2_uncertainty.py --root data/raw "
            "--train-roi s2_america_test/ROIs1158/106 --train-roi s2_africa_test/ROIs2017/32 "
            "--train-roi s2_africa_test/ROIs2017/140 --train-roi s2_asiaEast_test/ROIs1868/73 "
            "--select-roi s2_asiaEast_test/ROIs1868/119 "
            "--calib-roi s2_asiaEast_test/ROIs1970/139 --test-roi s2_america_test/ROIs1970/132 "
            f"--epochs 20 --img-size 256 --batch-size 8 --coverage 0.9 --seed {seed} "
            "--ref-mode linear_interp --ref-cache-dir data/ref_cache "
            f"--init-from-step2 results/step2_interp_gpu_256_seed{seed}/model.pt "
            f"--out {out}", cwd=REPO)
        run(f"git add -f {out}", cwd=REPO)
        run('git -c user.email="colab@example.com" -c user.name="Colab GPU run" '
            f'commit -m "Sigma head on frozen linear-interp model, GPU 256px, seed {seed}"', cwd=REPO)
        run(f"git push {remote} {BRANCH}", cwd=REPO, retries=3)

    print("Готово: усі seed у гілці", BRANCH)


if __name__ == "__main__":
    main()
