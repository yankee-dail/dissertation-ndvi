# Дисертація: оцінювання стану посівів за хмарності

Робочі матеріали до дисертаційного дослідження з реконструкції NDVI/NDRE за тривалої
хмарності (фенологічно та агрометеорологічно обмежені нейромережеві моделі мультимодальних
даних ДЗЗ). Концепція роботи та план - у записах дисертанта; тут - код кроку 1
("чесна база") та експортер даних.

## Структура

- `scripts/eval_protocol.py` - протокол оцінювання: метрики (RMSE, MAE, SSIM, SAM),
  три базові методи (`least_cloudy_raw`, `least_cloudy_strict`, `linear_interp`),
  детекція ризику витоку цілі у вхід, парний bootstrap. Самоперевірка: `--selftest`.
- `scripts/sen12mscrts_to_npz.py` - експортер офіційного датасету SEN12MS-CR-TS
  у формат `.npz`, який читає `eval_protocol.py`.
- `data/`, `results/` - локальні артефакти (не в git; великі й специфічні для машини).

## Встановлення

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-image rasterio natsort tqdm
```

## Крок 1: самоперевірка на синтетичних даних

```bash
python scripts/eval_protocol.py --selftest
```

## Реальні дані у вільному доступі

### SEN12MS-CR-TS (Ebel et al., 2022, IEEE TGRS)

Часові ряди Sentinel-1/2 з хмарними й безхмарними оптичними знімками, глобальний розподіл.
Ліцензія CC BY 4.0.

- Код і README: https://github.com/PatrickTUM/SEN12MS-CR-TS
- Train split (MediaTUM, ~151 ГБ): https://mediatum.ub.tum.de/1639953
- Test split (MediaTUM): https://mediatum.ub.tum.de/1659251
- Завантаження одним скриптом: `util/dl_data.sh` з репозиторію, або вручну:

  ```bash
  wget "ftp://m1639953:m1639953@dataserv.ub.tum.de/s1_africa.tar.gz"
  rsync -chavzP --stats rsync://m1639953@dataserv.ub.tum.de/m1639953/ .   # train
  rsync -chavzP --stats rsync://m1659251@dataserv.ub.tum.de/m1659251/ .  # test
  find . -name '*.tar.gz' -exec tar -xzvf {} \;
  rsync -a */* .   # злиття підкаталогів у структуру, яку очікує код
  ```

- Структура після злиття (підтверджено з коду `data/dataLoader.py`):

  ```
  <root>/ROIs<seed>/<roi>/S1/<tdx>/*.tif   # tdx = 0..29 - часова точка
  <root>/ROIs<seed>/<roi>/S2/<tdx>/*.tif
  ```

- Офіційний холдаут-набір для відтворюваного бенчмарку (фіксовані індекси
  вхід/ціль) - окремий `.npy` через pCloud-посилання в README репозиторію,
  завантажується прапорцем `--import_data_path`.
- Експорт у формат цього проєкту:

  ```bash
  python scripts/sen12mscrts_to_npz.py --root /path/to/SEN12MSCRTS --split test \
      --out data/sen12mscrts_test --max-patches-per-roi 5
  python scripts/eval_protocol.py --data data/sen12mscrts_test --out results/sen12mscrts_test
  ```

### AllClear (NeurIPS 2024, Cornell/Columbia)

4 млн зображень, 23 742 ROI за 2022 рік, Sentinel-1/2 + Landsat 8/9, хмарні маски й
land cover.

- Проєкт: https://allclear.cs.cornell.edu/
- Код і завантаження повного набору: https://github.com/Zhou-Hangyu/allclear

  ```bash
  git clone https://github.com/Zhou-Hangyu/allclear.git
  cd allclear && git submodule update --init --recursive
  python download.py
  ```

- Тестовий піднабір (один семпл на ROI, 23 ГБ) без клонування репозиторію:
  https://allclear.cs.cornell.edu/dataset/allclear_test_proi1_v1.zip
  (+ метадані: https://allclear.cs.cornell.edu/dataset/s2p_tx3_test_3k_1proi_v1.json)

## Відомий момент, який варто перевірити на реальних даних

У офіційному завантажувачі SEN12MS-CR-TS (режим `cloudy_cloudfree`, параметри за
замовчуванням `min_cov=0.0, max_cov=1.0`) ціль обирається як найменш хмарний кадр
серії (`cloudless_idx = argsort(coverage)[0]`), а вхідні кадри беруться як перші
`n_input_samples` кадрів у часовому порядку - без явного виключення `cloudless_idx`.
Це означає, що кадр-ціль може опинитися серед вхідних, якщо він рано в часовому
ряду. `least_cloudy_raw` в `eval_protocol.py` відтворює цей ефект; `least_cloudy_strict`
і `linear_interp` явно виключають цільовий кадр. Різниця RMSE між `raw` і
`leak_free`-підвибіркою - це міра завищення базового методу в літературі.
