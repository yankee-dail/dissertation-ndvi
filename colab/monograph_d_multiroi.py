"""
Експеримент D (монографія, розд. 5.8 і 6.3): уже навчені моделі (y_ref = інтерполяція, 5 seed) на НОВИХ
тестових ділянках SEN12MS-CR-TS - asiaWest і europa, яких не було ні в навчанні, ні в калібруванні.
Без навчання: лише прогін моделей, метрики й покриття з q, уже відкаліброваним на asiaEast/139.

ЗАПУСКАТИ В ОКРЕМІЙ сесії Colab (або після A6): скрипт видаляє з диска дані america/africa/asiaEast,
щоб вмістити нові регіони. GPU бажаний, але не обов'язковий.
    1. Клітинка:  import os; from google.colab import userdata
                  os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    2. Клітинка:  !curl -sfL -H "Authorization: token $GITHUB_TOKEN" -o monograph_d_multiroi.py \
                    https://raw.githubusercontent.com/yankee-dail/dissertation-ndvi/monograph-experiments/colab/monograph_d_multiroi.py \
                    && python monograph_d_multiroi.py

Ідемпотентний; після кожної ділянки - коміт і push у гілку monograph-d-multiroi.
"""
import os
import re
import subprocess
import sys
import time

REPO = "/content/dissertation-ndvi"
BRANCH = "monograph-d-multiroi"
BASE = "monograph-experiments"  # гілка, де є scripts/multi_roi_eval.py і ваги моделей
SEEDS = [0, 1, 2, 3, 4]
FTP = "ftp://m1659251:m1659251@dataserv.ub.tum.de"
SHA512 = {  # з checksums.sha512 на сервері TUM
    "s2_asiaWest_test": "1dcb80867797d83ded71cd34ab0de2fd0aa3ecbd7f473ae53f7b54720ec2015896862040567e5b5680698524e6875afbefe8cf8c789138000d1312e39622ae8e",
    "s2_europa_test": "6dac51b8091eb1d12f8ec6caab6d3936cf2a7a8af6c444593bd5dfb4d53cbc5d77043452feb54204aad0b62637b13ef64ab5ff9677b8d6cfaa2592fb9daf4201",
}
USED = set()  # asiaWest/100 бралась лише для прототипу кроку 1; у навчанні, виборі епохи й калібруванні фінальних моделей її немає
KEEP_FREE_GB = 25    # нову ділянку не починаємо, якщо на диску менше
MAX_ROIS = {"s2_europa_test": 5}  # europa (~28 ГБ стиснено) дочитувати до кінця - години; 5 ділянок достатньо
STALL_S = 180        # FTP мовчить довше - вважаємо обривом і починаємо архів спочатку


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



def stream_extract(name, raw):
    """Качає архів потоком, рахує SHA512 на льоту й розпаковує цілими ділянками, поки вистачає диска.
    europa (~28 ГБ стиснено, ~80 ГБ розпаковано) повністю на диск Colab не вміщується."""
    import hashlib
    import shutil
    import tarfile
    import urllib.request
    done = f"{raw}/{name}.done"
    if os.path.exists(done):
        print(f"{name}: уже розпаковано", flush=True)
        return
    h = hashlib.sha512()
    t0 = time.time()

    class Hashing:
        """Рахує SHA512 і раз на хвилину друкує прогрес, щоб довге мовчання не плутати із зависанням."""
        def __init__(self, f):
            self.f, self.n, self.last = f, 0, t0

        def read(self, n=-1):
            b = self.f.read(n)
            h.update(b)
            self.n += len(b)
            now = time.time()
            if now - self.last > 60:
                self.last = now
                print(f"    {name}: прочитано {self.n / 1e9:.1f} ГБ стиснутого, "
                      f"{self.n / 1e6 / (now - t0):.0f} МБ/с, вільно {shutil.disk_usage(raw).free / 1e9:.0f} ГБ",
                      flush=True)
            return b

    kept, skipped, cur, extract_cur, partial = [], set(), None, False, False
    limit = MAX_ROIS.get(name)
    with urllib.request.urlopen(f"{FTP}/{name}.tar.gz", timeout=STALL_S) as resp:
        stream = Hashing(resp)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for m in tar:
                parts = m.name.lstrip("./").split("/")
                if len(parts) < 3 or any(x.startswith("._") for x in parts):
                    continue
                roi = "/".join(parts[:3])
                if roi != cur:
                    if limit and len(kept) >= limit and roi not in kept:
                        partial = True  # потрібні ділянки вже повністю розпаковані - далі не читаємо
                        break
                    cur = roi
                    free = shutil.disk_usage(raw).free / 1e9
                    extract_cur = roi in kept or (roi not in USED and roi not in skipped and free > KEEP_FREE_GB)
                    if not extract_cur:
                        skipped.add(roi)
                    elif roi not in kept:
                        kept.append(roi)
                        print(f"  розпаковую {roi} (вільно {free:.0f} ГБ)", flush=True)
                if extract_cur:
                    tar.extract(m, raw, filter="data") if hasattr(tarfile, "data_filter") else tar.extract(m, raw)
        while not partial and stream.read(1 << 20):  # дочитати хвіст, щоб сума була повною
            pass
    if partial:  # SHA512 рахується лише по всьому архіву; цілісність прочитаної частини перевіряє deflate
        print(f"{name}: розпаковано {kept} (перші {limit} ділянок; SHA512 не перевірено - архів не дочитано)",
              flush=True)
    elif h.hexdigest() != SHA512[name]:
        sys.exit(f"{name}: SHA512 не збігається - архів пошкоджений, перезапусти клітинку")
    else:
        print(f"{name}: SHA512 ок; розпаковано {kept}, пропущено {sorted(skipped)}", flush=True)
    with open(done, "w") as f:
        f.write("\n".join(kept))


def main():
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN відсутній - встанови os.environ['GITHUB_TOKEN'] у клітинці ноутбука")
    check_token(token)
    remote = f"https://x-access-token:{token}@github.com/yankee-dail/dissertation-ndvi.git"

    if os.path.exists(REPO) and not os.path.isdir(f"{REPO}/.git"):
        aside = f"{REPO}_old_{int(time.time())}"
        os.rename(REPO, aside)
    if not os.path.isdir(f"{REPO}/.git"):
        run(f"git clone {remote} {REPO}")
    run("git fetch origin", cwd=REPO)
    if run(f"git rev-parse --verify origin/{BRANCH}", cwd=REPO, check=False) == 0:
        run(f"git checkout -B {BRANCH} origin/{BRANCH}", cwd=REPO)
        run(f"git merge --no-edit origin/{BASE}", cwd=REPO)
    else:
        run(f"git checkout -B {BRANCH} origin/{BASE}", cwd=REPO)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)

    raw = f"{REPO}/data/raw"
    os.makedirs(raw, exist_ok=True)
    # калібрувальні й навчальні дані тут не потрібні (q уже пораховано на asiaEast/139) - звільняємо диск
    for old in ("s2_america_test", "s2_africa_test", "s2_asiaEast_test"):
        if os.path.isdir(f"{raw}/{old}"):
            run(f"rm -rf {raw}/{old}")
    run("df -h /content")
    for name in ("s2_asiaWest_test", "s2_europa_test"):
        for attempt in range(3):  # FTP-потік без докачування: при обриві починаємо архів спочатку
            try:
                stream_extract(name, raw)
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
    print("Нові тестові ділянки:", rois, flush=True)

    for roi in rois:
        run(f"python scripts/multi_roi_eval.py --root data/raw --roi {roi} --seeds 0 1 2 3 4 --out results/multi_roi",
            cwd=REPO)
        run("git add -f results/multi_roi", cwd=REPO)
        if run("git diff --cached --quiet", cwd=REPO, check=False) != 0:
            run('git -c user.email="colab@example.com" -c user.name="Colab GPU run" '
                f'commit -m "Evaluate trained models on new test ROI {roi}"', cwd=REPO)
            run(f"git push {remote} {BRANCH}", cwd=REPO, retries=3)

    print("Готово: усі ділянки в гілці", BRANCH)


if __name__ == "__main__":
    main()
