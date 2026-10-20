"""
PyTorch Dataset, який читає SEN12MS-CR-TS напряму з розпакованої директорії (без
проміжного експорту в .npz - це важливо при обмеженому дисковому просторі: .npz з
повним 30-кадровим рядом на патч дублює дані, тут читання йде напряму з .tif).

Очікує структуру (справжню, підтверджену на реальному завантаженому зрізі):
    <root>/<roi_dir>/<roi>/S2/<tdx>/*.tif,  tdx в 0..29

Для кожного патча:
    - ціль (target_idx) обирається режимом 'random_clean' (див. sen12mscrts_to_npz.py) -
      випадковий достатньо чистий кадр, НЕЗАЛЕЖНИЙ від глобального мінімуму хмарності,
      щоб уникнути тавтології "ціль = найчистіший кадр".
    - y_ref - найчистіший кадр СЕРЕД ПОПЕРЕДНІХ у часі (причинний базовий рівень:
      "останнє відоме чисте спостереження"), або, якщо такого немає, найчистіший
      серед усіх кадрів окрім цілі.
    - x_near - кадр, найближчий у часі до цілі (окрім ref і самої цілі) - умовний
      вхід, що несе найсвіжішу (можливо, хмарну) інформацію про поточний стан поля.

Повертається (x, y_ref, target), де x = concat[y_ref, x_near, cloud(x_near), day_gap].
Модель передбачає залишок: ŷ = y_ref + Δθ(x).
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from eval_protocol import CLEAN_PIXEL_THRESH, TOL_SAME_DAYS

REFL_MAX = 10000.0
REF_MODES = ("prior_clean", "linear_interp")
S1_DB_MIN, S1_DB_MAX = -25.0, 0.0  # діапазон з офіційного process_SAR (dataLoader.py)


def read_tif(path):
    with rasterio.open(path) as tif:
        return tif.read().astype(np.float32)


def normalize_s1(img):
    """img: (2, H, W) VV/VH у дБ. Кліп у [-25, 0] і масштаб у [0,1] - як в офіційному коді."""
    return ((np.clip(img, S1_DB_MIN, S1_DB_MAX) - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)).astype(np.float32)


def read_tif_decimated(path, out_hw=32):
    """Швидке декантоване читання лише для оцінки хмарності (не для фінального зображення)."""
    with rasterio.open(path) as tif:
        return tif.read(out_shape=(tif.count, out_hw, out_hw)).astype(np.float32)


def simple_cloud_prob(img):
    bright = np.clip(img, 0, REFL_MAX).mean(axis=0) / REFL_MAX
    lo, hi = 0.25, 0.45
    return np.clip((bright - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def linear_interp_ref(frames, clouds, t_days, target_idx, clean_thresh=CLEAN_PIXEL_THRESH):
    """Попіксельна лінійна інтерполяція в часі між найближчими ЧИСТИМИ пікселями до і після цілі.

    frames: (T, C, H, W), clouds: (T, H, W), t_days: (T,). Цільовий кадр і кадри того самого
    дня (±TOL_SAME_DAYS) виключено. Якщо чистий піксель є лише з одного боку - береться він,
    якщо ніде - середнє за рядом (як eval_protocol.linear_interp, але векторизовано)."""
    rel = (t_days - t_days[target_idx]).astype(np.float64)
    keep = np.abs(rel) > TOL_SAME_DAYS
    keep[target_idx] = False
    f, c, d = frames[keep], clouds[keep], rel[keep]
    order = np.argsort(d)
    f, c, d = f[order], c[order], d[order]
    T, C, H, W = f.shape
    clean = (c < clean_thresh).reshape(T, -1)
    ff = f.reshape(T, C, -1)
    ar = np.arange(T)[:, None]
    ib = np.where(clean & (d[:, None] < 0), ar, -1).max(0)
    ia = np.where(clean & (d[:, None] > 0), ar, T).min(0)
    hb, ha = ib >= 0, ia < T
    p = np.arange(ff.shape[2])
    vb = ff[np.clip(ib, 0, T - 1), :, p].T
    va = ff[np.clip(ia, 0, T - 1), :, p].T
    db, da = d[np.clip(ib, 0, T - 1)], d[np.clip(ia, 0, T - 1)]
    w = np.where(hb & ha, -db / np.where(da - db == 0, 1, da - db), 0.0)
    out = np.where(hb & ha, vb + w * (va - vb), np.where(hb, vb, np.where(ha, va, ff.mean(0))))
    return out.reshape(C, H, W).astype(np.float32)


def pick_target_idx(frac, rng, max_cloud, mode="random_clean"):
    if mode == "argmin":
        return int(np.argmin(frac))
    candidates = np.where(frac <= max_cloud)[0]
    if len(candidates) == 0:
        return int(np.argmin(frac))
    return int(rng.choice(candidates))


class SEN12MSCRTSPatchDataset(Dataset):
    def __init__(self, root, roi_dir, roi, patch_indices, n_time_points=30,
                 target_mode="random_clean", target_max_cloud=0.3, seed=0, img_size=None,
                 s1_root=None, era5_path=None, ref_mode="prior_clean", ref_cache_dir=None):
        """s1_root: директорія з тією самою структурою <roi_dir>/<roi>/S1/<tdx>/*.tif -
        якщо задано, до x додається VV/VH (ref і near кадри, 2x2=4 канали). S1 НІКОЛИ не
        йде в ціль - лише як додатковий вхід (радар не замінює оптичну ground truth).
        era5_path: шлях до .nc від fetch_era5.py (один на ROI) - якщо задано, 5-й елемент
        повертаного кортежу - це різниця кумулятивного GDD (а не днів) між y_ref і ціллю,
        точніше масштабує фенологічний штраф (GDD, не календарний час, визначає фенофазу)."""
        self.root = Path(root)
        self.roi_dir, self.roi = roi_dir, roi
        self.patch_indices = list(patch_indices)
        self.n_t = n_time_points
        self.target_mode = target_mode
        self.target_max_cloud = target_max_cloud
        self.seed = seed
        self.img_size = img_size
        self.s1_root = Path(s1_root) if s1_root is not None else None
        if ref_mode not in REF_MODES:
            raise ValueError(f"ref_mode має бути одним із {REF_MODES}")
        self.ref_mode = ref_mode
        self.ref_cache_dir = Path(ref_cache_dir) if ref_cache_dir is not None else None
        if self.ref_cache_dir is not None:
            self.ref_cache_dir.mkdir(parents=True, exist_ok=True)

        self.era5 = None
        if era5_path is not None:
            from era5_lookup import Era5Lookup
            self.era5 = Era5Lookup(era5_path)

        self.series = []  # [tdx] -> sorted list of patch file paths
        for tdx in range(n_time_points):
            d = self.root / roi_dir / roi / "S2" / str(tdx)
            files = sorted(f for f in d.iterdir() if f.suffix == ".tif")
            self.series.append(files)
        lengths = {len(f) for f in self.series}
        if len(lengths) != 1:
            raise ValueError(f"різна кількість патчів між часовими точками: {lengths}")

        if self.s1_root is not None:
            self.s1_series = []
            for tdx in range(n_time_points):
                d = self.s1_root / roi_dir / roi / "S1" / str(tdx)
                files = sorted(f for f in d.iterdir() if f.suffix == ".tif")
                self.s1_series.append(files)
            s1_lengths = {len(f) for f in self.s1_series}
            if s1_lengths != lengths:
                raise ValueError(f"кількість патчів S1 ({s1_lengths}) не збігається з S2 ({lengths})")

        # Одноразовий прохід: обираємо (target_idx, ref_idx, near_idx) на декантованих
        # (швидких) версіях зображень, щоб потім на кожній епосі читати лише 3 повнорозмірні
        # .tif на семпл замість усіх 30 (10x менше дискового І/О на епоху).
        self._picks = {}
        for pdx in self.patch_indices:
            self._picks[pdx] = self._precompute_picks(pdx)

    def _precompute_picks(self, pdx):
        paths = [self.series[tdx][pdx] for tdx in range(self.n_t)]
        small = [read_tif_decimated(p) for p in paths]
        frac = np.array([float(simple_cloud_prob(f).mean()) for f in small])
        t_days = np.array([self._parse_day(p.name) for p in paths], dtype=np.int64)

        rng = np.random.default_rng(self.seed + pdx)
        target_idx = pick_target_idx(frac, rng, self.target_max_cloud, self.target_mode)

        before = [t for t in range(self.n_t) if t_days[t] < t_days[target_idx]]
        if before:
            ref_idx = before[int(np.argmin(frac[before]))]
        else:
            others = [t for t in range(self.n_t) if t != target_idx]
            ref_idx = others[int(np.argmin(frac[others]))]

        rest = [t for t in range(self.n_t) if t not in (target_idx, ref_idx)]
        near_idx = rest[int(np.argmin(np.abs(t_days[rest] - t_days[target_idx])))]

        return {"target_idx": target_idx, "ref_idx": ref_idx, "near_idx": near_idx, "t_days": t_days}

    def __len__(self):
        return len(self.patch_indices)

    def _maybe_resize(self, arr):
        # arr: (C, H, W). Прості strided-кроки замість інтерполяції - досить для прототипу швидкості.
        if self.img_size is None:
            return arr
        h, w = arr.shape[-2:]
        step_h, step_w = h // self.img_size, w // self.img_size
        return arr[:, ::step_h, ::step_w][:, : self.img_size, : self.img_size]

    def full_series(self, i):
        """Увесь 30-кадровий ряд фрагмента: (T, C, H, W) відбиття [0,1] і (T, H, W) хмарність."""
        pdx = self.patch_indices[i]
        frames, clouds = [], []
        for t in range(self.n_t):
            raw = read_tif(self.series[t][pdx])
            frames.append(self._maybe_resize((np.clip(raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32)))
            clouds.append(self._maybe_resize(simple_cloud_prob(raw)[None])[0])
        return np.stack(frames), np.stack(clouds)

    def _interp_ref(self, i, pick):
        pdx = self.patch_indices[i]
        cache = None
        if self.ref_cache_dir is not None:
            cache = self.ref_cache_dir / (f"{self.roi_dir}_{self.roi}_s{self.seed}_i{self.img_size}_"
                                          f"t{self.target_mode}{self.target_max_cloud}_p{pdx}.npy")
            if cache.exists():
                return np.load(cache)
        frames, clouds = self.full_series(i)
        ref = linear_interp_ref(frames, clouds, pick["t_days"], pick["target_idx"])
        if cache is not None:
            np.save(cache, ref)
        return ref

    def __getitem__(self, i):
        pdx = self.patch_indices[i]
        pick = self._picks[pdx]
        target_idx, ref_idx, near_idx, t_days = pick["target_idx"], pick["ref_idx"], pick["near_idx"], pick["t_days"]

        x_near_raw = read_tif(self.series[near_idx][pdx])
        target_raw = read_tif(self.series[target_idx][pdx])

        if self.ref_mode == "linear_interp":
            y_ref = self._interp_ref(i, pick)
        else:
            y_ref = self._maybe_resize((np.clip(read_tif(self.series[ref_idx][pdx]), 0, REFL_MAX) / REFL_MAX).astype(np.float32))
        x_near_norm = (np.clip(x_near_raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32)
        x_near = self._maybe_resize(x_near_norm)
        cloud_near = self._maybe_resize(simple_cloud_prob(x_near_raw)[None].astype(np.float32))
        target = self._maybe_resize((np.clip(target_raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32))

        day_gap = np.full((1,) + y_ref.shape[1:], (t_days[near_idx] - t_days[target_idx]) / 365.0,
                           dtype=np.float32)
        parts = [y_ref, x_near, cloud_near, day_gap]

        if self.s1_root is not None:
            s1_ref = normalize_s1(read_tif(self.s1_series[ref_idx][pdx]))
            s1_near = normalize_s1(read_tif(self.s1_series[near_idx][pdx]))
            parts += [self._maybe_resize(s1_ref), self._maybe_resize(s1_near)]

        x = np.concatenate(parts, axis=0)

        # Хмарність ЦІЛІ (а не входу): target_max_cloud - поріг на СЕРЕДНЮ хмарність усього
        # патча, тож окремі пікселі цілі можуть лишатись хмарними. Повертаємо карту, щоб
        # метрики могли виключити ці пікселі замість порівнювати прогноз із хмарою.
        target_cloud = self._maybe_resize(simple_cloud_prob(target_raw)[None].astype(np.float32))[0]

        # Інтервал y_ref -> ціль - потрібен фенологічному члену втрат для масштабування
        # допустимої швидкості зміни NDVI. У GDD (точніше, бо фенофаза залежить від суми
        # ефективних температур, не календарного часу), якщо є ERA5; інакше - у днях
        # (не плутати з day_gap у x, який кодує near->ціль, інший інтервал).
        if self.era5 is not None:
            gdd_target, _ = self.era5(self._day_to_date(t_days[target_idx]))
            gdd_ref, _ = self.era5(self._day_to_date(t_days[ref_idx]))
            ref_target_scale = np.float32(abs(gdd_target - gdd_ref))
        else:
            ref_target_scale = np.float32(abs(t_days[target_idx] - t_days[ref_idx]))

        return (torch.from_numpy(x), torch.from_numpy(y_ref), torch.from_numpy(target),
                torch.from_numpy(target_cloud), torch.tensor(ref_target_scale))

    def provenance(self, i):
        """Метадані походження для інформаційної моделі (§11 REPORT.md) - НЕ впливає на
        __getitem__/тренування, лише для трасування "виміряне -> відновлене" поза
        тензорами. Повертає файли й дати ref/target кадрів, обраних для цього патча."""
        pdx = self.patch_indices[i]
        pick = self._picks[pdx]
        ref_idx, target_idx = pick["ref_idx"], pick["target_idx"]
        return {
            "roi": f"{self.roi_dir}/{self.roi}", "patch_idx": pdx,
            "ref_file": self.series[ref_idx][pdx].name,
            "ref_date": str(self._day_to_date(pick["t_days"][ref_idx])),
            "target_file": self.series[target_idx][pdx].name,
            "target_date": str(self._day_to_date(pick["t_days"][target_idx])),
        }

    @staticmethod
    def _day_to_date(t_day):
        from datetime import datetime, timedelta
        return (datetime(2014, 4, 3) + timedelta(days=int(t_day))).date()

    @staticmethod
    def _parse_day(filename):
        from datetime import datetime
        token = filename.split("_")[5]
        d = datetime.strptime(token, "%Y-%m-%d")
        return (d - datetime(2014, 4, 3)).days


def n_bands():
    return 13


def n_input_channels(include_s1=False):
    base = 13 + 13 + 1 + 1  # y_ref + x_near + cloud_near + day_gap
    return base + (2 + 2 if include_s1 else 0)  # + VV/VH для ref і near
