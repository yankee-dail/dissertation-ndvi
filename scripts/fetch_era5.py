#!/usr/bin/env python3
"""Завантажує добові ERA5-Land (температура 2м, опади) для ROI через Copernicus CDS API.

Потребує ~/.cdsapirc з url/key (див. README проєкту). Запит у чергу CDS може тривати
від хвилин до годин залежно від завантаженості сервера - це нормально, не помилка.

Використання:
    python fetch_era5.py --name america_106 --bbox 34.522 -116.783 34.545 -116.755 \
        --year 2018 --out data/era5
"""
import argparse
from pathlib import Path

import cdsapi


def fetch(name, south, west, north, east, year, out_dir, pad=0.15):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.nc"
    if out_path.exists():
        print(f"{name}: вже завантажено ({out_path}), пропускаю")
        return

    c = cdsapi.Client()
    area = [north + pad, west - pad, south - pad, east + pad]  # [N, W, S, E]
    print(f"{name}: запит до CDS, area={area}, рік={year} ...")
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["2m_temperature", "total_precipitation"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"{name}: збережено -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--out", default="data/era5")
    args = ap.parse_args()
    fetch(args.name, *args.bbox, args.year, args.out)


if __name__ == "__main__":
    main()
