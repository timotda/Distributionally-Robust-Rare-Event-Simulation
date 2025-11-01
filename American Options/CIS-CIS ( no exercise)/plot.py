#!/usr/bin/env python3
"""
Fill two columns and plot column Y vs column X.

Requires: pandas, matplotlib
pip install pandas matplotlib
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============ EDIT ME ============
# Choose your column names (in the final DataFrame)
X_NAME = "s*"
Y_NAME = "Estimated Proba"

# Option A: type your data (same length!)
X_DATA = [75,80,85,89,89.9999,95,100,110]  # e.g., [1, 2, 3, 4]
R_DATA = [0.3809098135763269, 0.38172028418390086,0.38341019154698663,0.3874360868429277,0.3904159997873934,0.8697680549669892,1.5520836766845567,2.8775243209993597]  # e.g., [0.1, 0.2, 0.3, 0.4]
LSRE_DATA = [3.4,3.7,4.4,5.2,5.4,3.6,3.2,2.8]  # e.g., [2.0, 2.5, 3.2, 4.1]
Y_DATA = [0.178,0.14,0.11,0.08,0.07,0.03,0.01,0.002 ]  # e.g., [0.01, 0.02, 0.03, 0.04]

# Option B: read from CSV instead
USE_CSV = False
CSV_PATH = "data.csv"
CSV_X_COL = "your_x_column_name_in_csv"
CSV_Y_COL = "your_y_column_name_in_csv"
# =================================

def build_dataframe() -> pd.DataFrame:
    if USE_CSV:
        csv_file = Path(CSV_PATH)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file.resolve()}")
        df = pd.read_csv(csv_file, usecols=[CSV_X_COL, CSV_Y_COL]).rename(
            columns={CSV_X_COL: X_NAME, CSV_Y_COL: Y_NAME}
        )
    else:
        if not X_DATA or not Y_DATA:
            raise ValueError("Please fill X_DATA and Y_DATA with values.")
        if len(X_DATA) != len(Y_DATA):
            raise ValueError(f"Lengths differ: len(X_DATA)={len(X_DATA)} vs len(Y_DATA)={len(Y_DATA)}.")
        df = pd.DataFrame({X_NAME: X_DATA, Y_NAME: Y_DATA})

    # Coerce to numeric and drop non-numeric/NaN rows
    before = len(df)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < before:
        print(f"[info] Dropped {before - len(df)} non-numeric/NaN rows.")
    if df.empty:
        raise ValueError("DataFrame ended up empty. Check inputs.")
    return df

def plot_xy(df: pd.DataFrame, outfile: str = "xy_scatter.png") -> None:
    plt.figure()
    plt.plot(df[X_NAME], df[Y_NAME])
    plt.xlabel(X_NAME)
    plt.ylabel(Y_NAME)
    plt.title(f"{Y_NAME} vs {X_NAME}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"[saved] {outfile}")
    plt.show()

def main():
    df = build_dataframe()
    print(df.head())  # quick peek
    plot_xy(df)

if __name__ == "__main__":
    main()
