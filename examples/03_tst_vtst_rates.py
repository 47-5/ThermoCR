"""Calculate TST and VTST rate scans from thermo tables."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ThermoCR.kinetics import calculate_tst_rate_frame, calculate_vtst_rate_frame


EXAMPLE_DIR = ROOT / "example"
OUTPUT_DIR = ROOT / "examples" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reactant = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_CPD.xlsx")
    ts = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_TS.xlsx")
    path1 = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_01_02_path1_1.xlsx")
    path2 = pd.read_excel(EXAMPLE_DIR / "QMthermoScan_01_02_path2_1.xlsx")

    tst = calculate_tst_rate_frame(ts, [reactant, reactant])
    vtst = calculate_vtst_rate_frame(
        [path1, path2],
        [reactant, reactant],
        path_names=["irc_path1", "irc_path2"],
    )

    tst_csv = OUTPUT_DIR / "TST_2CPD_to_DCPD.csv"
    vtst_csv = OUTPUT_DIR / "VTST_2CPD_to_DCPD.csv"
    tst.to_csv(tst_csv, index=False)
    vtst.to_csv(vtst_csv, index=False)

    print(f"wrote: {tst_csv}")
    print(f"wrote: {vtst_csv}")
    print(f"lowest VTST rate at {vtst['temperature'].iloc[0]:.1f} K: {vtst['rate_constant'].iloc[0]:.6e}")


if __name__ == "__main__":
    main()
