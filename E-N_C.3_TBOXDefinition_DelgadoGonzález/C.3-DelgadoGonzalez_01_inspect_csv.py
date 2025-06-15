#!/usr/bin/env python
"""
inspect_csv.py  ·  Muestra forma, tipos de dato y primeras filas
Uso:
    python src/inspect_csv.py [archivo1.csv archivo2.csv ...]
Sin argumentos inspecciona todos los .csv en /data.
"""
import sys, textwrap
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Archivos a leer: args o todos los *.csv
files = [Path(f) for f in sys.argv[1:]] or sorted(DATA_DIR.glob("*.csv"))

if not files:
    print("No se encontraron CSV en 'data/'.")
    sys.exit(1)

for path in files:
    if not path.exists():
        print(f"  {path.name} no encontrado, se omite.")
        continue

    df = pd.read_csv(path)
    print("\n" + "="*70)
    print(f"{path.name}  —  Shape {df.shape}")
    print("-"*70)
    print(df.dtypes)

    print("\nPrimeras 5 filas:")
    print(df.head(), "\n")

    # Valores únicos para posibles PK/FK
    print("Valores únicos (máx 10) en columnas que parecen IDs o categóricas:")
    for col in df.columns:
        nunq = df[col].nunique(dropna=True)
        if col.lower().endswith("id") or nunq <= 10:
            vals = ", ".join(map(str, df[col].unique()[:10]))
            print(f"  {col:<20} → {nunq:>5}  [{vals}]")
