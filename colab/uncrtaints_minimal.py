"""
Мінімальний, малодисковий шлях до UnCRtainTS-порівняння в Colab після скидання сесії
(попередня спроба забила весь диск через S1+S2 для 3 регіонів ≈140+ ГБ - 235 ГБ диска
Colab цього не витримало разом з усім іншим). Тут - лише регіон america (ROIs1158/106 +
ROIs1970/132, той самий test ROI, що й у results/step2_final/uncertainty_final) - цього
достатньо для порівняння на тому самому test ROI, не потрібні всі 7.

ВИКОРИСТАННЯ в Colab (після скидання сесії):
    1. Переконайся, що GPU-рантайм під'єднано (Runtime -> Change runtime type -> T4 GPU).
    2. У новій клітинці:
         %%writefile uncrtaints_minimal.py
         <встав увесь вміст цього файлу>
    3. У наступній клітинці:
         !python uncrtaints_minimal.py
       (GITHUB_TOKEN підхоплюється з Colab Secrets автоматично - має зберегтись між
       сесіями, це налаштування акаунта, не /content).
"""
import os
import subprocess
import sys
import time


def run(cmd, cwd=None, check=True, retries=1):
    """retries>1 - для мережевих команд (curl по FTP зрідка падає на транзієнтних
    збоях з'єднання, напр. errno 115 - це не означає, що сервер недоступний,
    повторна спроба зазвичай проходить)."""
    for attempt in range(1, retries + 1):
        print(f"$ {cmd}" + (f"  (спроба {attempt}/{retries})" if retries > 1 else ""))
        result = subprocess.run(cmd, shell=True, cwd=cwd)
        if result.returncode == 0:
            return 0
        if attempt < retries:
            print(f"  -> код {result.returncode}, чекаю 10с і пробую ще раз")
            time.sleep(10)
    if check:
        sys.exit(f"Команда впала (код {result.returncode}) після {retries} спроб(и): {cmd}")
    return result.returncode


def main():
    # === 1. GITHUB_TOKEN ===
    # google.colab.userdata.get() вимагає живого IPython-ядра (комунікує з ним через
    # get_ipython()) - у ПІДПРОЦЕСІ (`!python script.py`) ядра немає, тому виклик падає
    # з AttributeError. Тому токен отримуємо тут із змінної середовища, яку потрібно
    # встановити В САМІЙ клітинці ноутбука (де userdata.get() ще працює) ПЕРЕД запуском
    # цього скрипта: os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN відсутній у середовищі - встанови його в клітинці ноутбука "
                 "(os.environ['GITHUB_TOKEN'] = userdata.get('GITHUB_TOKEN')) перед запуском цього скрипта")

    # === 2. Клонування основного репозиторію (ідемпотентно - повторний запуск після
    # збою пізнішого кроку не повинен падати на "destination already exists") ===
    if not os.path.isdir("/content/dissertation-ndvi/.git"):
        run(f"git clone https://{token}@github.com/yankee-dail/dissertation-ndvi.git /content/dissertation-ndvi")
    else:
        print("dissertation-ndvi вже клоновано - пропускаю")

    import torch
    print(f"CUDA доступна: {torch.cuda.is_available()}, torch={torch.__version__}")

    # === 3. Завантаження S2+S1 ЛИШЕ для america (не africa/asiaEast - економія ~100 ГБ).
    # Ідемпотентно - якщо дані вже розпаковані (ROI-директорія існує), пропускаємо. ===
    raw_dir = "/content/dissertation-ndvi/data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(raw_dir, "s2_america_test")):
        run('curl --fail --show-error -o s2_america_test.tar.gz '
            '"ftp://m1659251:m1659251@dataserv.ub.tum.de/s2_america_test.tar.gz"', cwd=raw_dir, retries=3)
        run('echo "19c7a11d81c1a574f5876ec8e21dfec7f101677704021a6191146e59625670419e8606aa1efb5a9f39c74700a4830875f87025ce4b8a9de9e6fcbf35d233276f  '
            's2_america_test.tar.gz" | shasum -a 512 -c -', cwd=raw_dir)
        run("tar -xzf s2_america_test.tar.gz && rm s2_america_test.tar.gz", cwd=raw_dir)
    else:
        print("s2_america_test вже розпаковано - пропускаю")

    if not os.path.isdir(os.path.join(raw_dir, "s1_america_test")):
        run('curl --fail --show-error -o s1_america_test.tar.gz '
            '"ftp://m1659251:m1659251@dataserv.ub.tum.de/s1_america_test.tar.gz"', cwd=raw_dir, retries=3)
        run('echo "416d3d94c6b0f8a7b5c729ed754aa5a52e4dc662ac64b779c57956d31b5cd91ceee6dc3dcde6d08531e949ab9bb25a810be807e14d0cc7a2675dcbc12abf7a12  '
            's1_america_test.tar.gz" | shasum -a 512 -c -', cwd=raw_dir)
        run("tar -xzf s1_america_test.tar.gz && rm s1_america_test.tar.gz", cwd=raw_dir)
    else:
        print("s1_america_test вже розпаковано - пропускаю")

    run("df -h /content")

    # === 4. Клон UnCRtainTS + залежності (той самий фікс, що й у головному ноутбуці:
    # НЕ --no-deps для основного списку - torch/torchvision вже виключені з нього) ===
    uncrt_dir = "/content/UnCRtainTS"
    if not os.path.isdir(os.path.join(uncrt_dir, ".git")):
        run(f"git clone --depth 1 https://github.com/PatrickTUM/UnCRtainTS.git {uncrt_dir}")
    else:
        print("UnCRtainTS вже клоновано - пропускаю")

    print(f"ДО встановлення залежностей: CUDA доступна={torch.cuda.is_available()}")
    run("sed -E 's/[~=<>!].*$//' requirements.txt | "
        "grep -vE \"^(torch|torchvision|torch_scatter|torchgeometry)$\" > requirements_colab.txt",
        cwd=uncrt_dir)
    run("pip install -q -r requirements_colab.txt", cwd=uncrt_dir)
    run("python -c \"import torch; print('Підпроцес бачить CUDA:', torch.cuda.is_available())\"", cwd=uncrt_dir)

    cuda_tag = ("+cu" + torch.version.cuda.replace(".", "")) if torch.version.cuda else "+cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}{cuda_tag}.html"
    run(f"pip install -q torch_scatter --no-deps -f {wheel_url}", cwd=uncrt_dir, check=False)
    run("pip install -q torchgeometry --no-deps", cwd=uncrt_dir, check=False)
    run("python -c \"import torch; print('Фінальна перевірка, CUDA:', torch.cuda.is_available())\"", cwd=uncrt_dir)

    # === 5. Об'єднання S1+S2 в очікувану UnCRtainTS структуру (лише america) ===
    src_root = raw_dir
    merged_root = "/content/dissertation-ndvi/data/merged"
    for region_dir in os.listdir(src_root):
        region_path = os.path.join(src_root, region_dir)
        if not os.path.isdir(region_path) or not region_dir.startswith("s2_"):
            continue
        s1_region_path = os.path.join(src_root, "s1_" + region_dir[len("s2_"):])
        for seed_dir in os.listdir(region_path):
            seed_path = os.path.join(region_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            for roi in os.listdir(seed_path):
                dst_roi = os.path.join(merged_root, seed_dir, roi)
                os.makedirs(dst_roi, exist_ok=True)
                s2_dst = os.path.join(dst_roi, "S2")
                if not os.path.exists(s2_dst):
                    os.symlink(os.path.join(seed_path, roi, "S2"), s2_dst)
                s1_src = os.path.join(s1_region_path, seed_dir, roi, "S1")
                s1_dst = os.path.join(dst_roi, "S1")
                if os.path.isdir(s1_src) and not os.path.exists(s1_dst):
                    os.symlink(s1_src, s1_dst)
    print("Об'єднано в", merged_root)
    run(f"find {merged_root} -maxdepth 3")

    # === 6. Завантаження готових ваг noSAR_1 (pCloud public API, без інтерактивного логіну) ===
    import requests
    pcloud_code = "kZsdbk0Z5Y2Y2UEm48XLwOvwSVlL8R2L3daV"
    experiment_name = "noSAR_1"

    listing = requests.get("https://api.pcloud.com/showpublink", params={"code": pcloud_code}).json()
    folder = next(f for f in listing["metadata"]["contents"] if f["name"] == experiment_name)

    weight_dir = f"/content/UnCRtainTS/results/{experiment_name}"
    os.makedirs(weight_dir, exist_ok=True)
    for item in folder["contents"]:
        dl = requests.get("https://api.pcloud.com/getpublinkdownload",
                           params={"code": pcloud_code, "fileid": item["fileid"]}).json()
        url = f"https://{dl['hosts'][0]}{dl['path']}"
        out_path = f"{weight_dir}/{item['name']}"
        run(f'curl -sL -o {out_path} "{url}"')
        print(item["name"], "->", out_path)
    run(f"ls -la {weight_dir}")

    # === 7. Запуск інференсу ===
    run(
        f"python test_reconstruct.py "
        f"--experiment_name {experiment_name} "
        f"--root1 /content/dissertation-ndvi/data/merged "
        f"--root2 /content/dissertation-ndvi/data/merged "
        f"--root3 /content/dissertation-ndvi/data/merged "
        f"--input_t 3 --region all --export_every 1 "
        f"--res_dir /content/dissertation-ndvi/results/uncrtaints_inference "
        f"--weight_folder /content/UnCRtainTS/results",
        cwd=f"{uncrt_dir}/model",
    )


if __name__ == "__main__":
    main()
