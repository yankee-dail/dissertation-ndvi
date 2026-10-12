"""Завантажує .nc від fetch_era5.py і будує по-денну таблицю GDD/опадів для ROI.

GDD (growing degree days) - добова сума ефективних температур: max(0, T_mean - T_base).
T_base=10 C - типове спрощення (без знання конкретної культури на полі, що неможливо
визначити з самих лише супутникових знімків); документується як спрощення, не факт.
Кумулятивний GDD рахується від 1 січня року спостережень (спрощення "початку сезону" -
справжня дата сівби невідома).
"""
from datetime import datetime, date
from pathlib import Path

import numpy as np
import xarray as xr

T_BASE_C = 10.0


def resolve_era5_path(spec, era5_root):
    """'s2_america_test/ROIs1158/106' -> data/era5/america_106.nc"""
    parts = spec.strip("/").split("/")
    region = parts[0]
    if region.startswith("s2_"):
        region = region[3:]
    if region.endswith("_test"):
        region = region[: -len("_test")]
    roi = parts[-1]
    return Path(era5_root) / f"{region}_{roi}.nc"


class Era5Lookup:
    """По даті (date) повертає (cum_gdd, precip_7d_mm) - кумулятивний GDD від 1 січня
    того ж року і суму опадів за попередні 7 днів (включно), в мм."""

    def __init__(self, nc_path, t_base=T_BASE_C):
        ds = xr.open_dataset(nc_path)
        t2m_c = ds["t2m"].mean(dim=["latitude", "longitude"]) - 273.15  # K -> C, просторове усереднення
        tp_mm = ds["tp"].mean(dim=["latitude", "longitude"]) * 1000.0  # м -> мм

        times = ds["valid_time"].values
        dates = np.array([np.datetime64(t, "D") for t in times])
        day_index = np.unique(dates)

        daily_t = np.array([float(t2m_c.values[dates == d].mean()) for d in day_index])
        daily_p = np.array([float(tp_mm.values[dates == d].sum()) for d in day_index])

        gdd_daily = np.clip(daily_t - t_base, 0, None)

        self._dates = [d.astype("datetime64[D]").astype(date) for d in day_index]
        self._date_to_idx = {d: i for i, d in enumerate(self._dates)}
        self._cum_gdd = np.zeros(len(self._dates))
        year_start_idx = 0
        for i, d in enumerate(self._dates):
            if d.month == 1 and d.day == 1:
                year_start_idx = i
            self._cum_gdd[i] = gdd_daily[year_start_idx : i + 1].sum()
        self._precip_7d = np.array([
            daily_p[max(0, i - 6) : i + 1].sum() for i in range(len(self._dates))
        ])

    def __call__(self, d):
        """d: datetime.date або datetime.datetime. Повертає (cum_gdd, precip_7d_mm);
        найближча доступна дата, якщо точного збігу немає (на випадок дірок у ряді)."""
        if isinstance(d, datetime):
            d = d.date()
        idx = self._date_to_idx.get(d)
        if idx is None:
            diffs = [abs((dd - d).days) for dd in self._dates]
            idx = int(np.argmin(diffs))
        return float(self._cum_gdd[idx]), float(self._precip_7d[idx])
