"""Command-line interface for ThermoCR."""

import argparse
from pathlib import Path

from ThermoCR.io.gaussian import select_gaussian_output, split_gaussian_link1_output
from ThermoCR.io.orca import read_orca_final_single_point_energy
from ThermoCR.io.qm_output import read_electronic_energy, read_molecule_data
from ThermoCR.thermo import ThermoOptions, scan_thermo


def _positive_or_negative_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    return parsed


def _positive_int(value):
    parsed = _positive_or_negative_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _temperature_grid(t_min, t_max, n_points):
    if n_points == 1:
        return [float(t_min)]
    step = (float(t_max) - float(t_min)) / (n_points - 1)
    return [float(t_min) + index * step for index in range(n_points)]


def _write_dataframe(df, output_path):
    output_path = Path(output_path)
    if output_path.suffix.lower() in {".xls", ".xlsx"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def _cmd_split_link1(args):
    output_paths = split_gaussian_link1_output(
        input_path=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    for output_path in output_paths:
        print(output_path)
    return 0


def _cmd_select_gaussian(args):
    select_gaussian_output(
        input_path=args.input,
        output_path=args.output,
        task_id=args.task_id,
        select_mode=args.mode,
    )
    print(Path(args.output))
    return 0


def _cmd_qm_energy(args):
    energy = read_electronic_energy(
        filepath=args.input,
        energy_index=args.energy_index,
        return_hartree=args.unit == "hartree",
        gaussian_job_index=args.gaussian_job_index,
    )
    print(energy)
    return 0


def _cmd_orca_energy(args):
    energy = read_orca_final_single_point_energy(args.input)
    print(energy)
    return 0


def _cmd_thermo_scan(args):
    molecule = read_molecule_data(
        args.input,
        gaussian_job_index=args.gaussian_job_index,
    )
    options = ThermoOptions(
        pressure=args.pressure,
        ignore_trans_and_rot=args.ignore_trans_and_rot,
    )
    temperatures = _temperature_grid(args.t_min, args.t_max, args.n_points)
    df = scan_thermo(
        molecule,
        temperatures=temperatures,
        pressure=args.pressure,
        options=options,
    )
    output_path = _write_dataframe(df, args.output)
    print(output_path)
    return 0


def _add_thermo_commands(subparsers):
    thermo_parser = subparsers.add_parser(
        "thermo",
        help="structured thermochemistry workflows",
    )
    thermo_subparsers = thermo_parser.add_subparsers(
        dest="thermo_command",
        required=True,
    )

    scan_parser = thermo_subparsers.add_parser(
        "scan",
        help="scan thermochemistry over a temperature grid",
    )
    scan_parser.add_argument("input", help="QM output file to read")
    scan_parser.add_argument("--output", required=True, help="CSV/XLSX output file")
    scan_parser.add_argument(
        "--t-min",
        type=float,
        default=300.0,
        help="minimum temperature in K; default: 300",
    )
    scan_parser.add_argument(
        "--t-max",
        type=float,
        default=3000.0,
        help="maximum temperature in K; default: 3000",
    )
    scan_parser.add_argument(
        "--n-points",
        type=_positive_int,
        default=100,
        help="number of temperature points; default: 100",
    )
    scan_parser.add_argument(
        "--pressure",
        type=float,
        default=101325.0,
        help="pressure in Pa; default: 101325",
    )
    scan_parser.add_argument(
        "--gaussian-job-index",
        type=_positive_or_negative_int,
        help="Gaussian Link1 job index; default: last job when Link1 is detected",
    )
    scan_parser.add_argument(
        "--ignore-trans-and-rot",
        action="store_true",
        help="ignore translational and rotational contributions",
    )
    scan_parser.set_defaults(func=_cmd_thermo_scan)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="thermocr",
        description="ThermoCR command-line utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser(
        "split-link1",
        help="split a Gaussian Link1 output into single-job output files",
    )
    split_parser.add_argument("input", help="Gaussian output file to split")
    split_parser.add_argument("output_dir", help="directory for split job files")
    split_parser.add_argument("--prefix", help="output filename prefix")
    split_parser.set_defaults(func=_cmd_split_link1)

    select_parser = subparsers.add_parser(
        "select-gaussian",
        help="write one selected Gaussian job or prefix to an output file",
    )
    select_parser.add_argument("input", help="Gaussian output file to read")
    select_parser.add_argument("output", help="selected output file to write")
    select_parser.add_argument(
        "--task-id",
        type=_positive_or_negative_int,
        default=2,
        help="one-based Gaussian job index; default: 2",
    )
    select_parser.add_argument(
        "--mode",
        choices=("cut", "select"),
        default="cut",
        help="selection behavior; default: cut",
    )
    select_parser.set_defaults(func=_cmd_select_gaussian)

    qm_energy_parser = subparsers.add_parser(
        "qm-energy",
        help="read electronic energy from a Gaussian/ORCA-compatible QM output",
    )
    qm_energy_parser.add_argument("input", help="QM output file to read")
    qm_energy_parser.add_argument(
        "--energy-index",
        type=int,
        default=-1,
        help="energy index from cclib scfenergies; default: -1",
    )
    qm_energy_parser.add_argument(
        "--gaussian-job-index",
        type=_positive_or_negative_int,
        help="Gaussian Link1 job index; default: last job when Link1 is detected",
    )
    qm_energy_parser.add_argument(
        "--unit",
        choices=("hartree", "ev"),
        default="hartree",
        help="output unit; default: hartree",
    )
    qm_energy_parser.set_defaults(func=_cmd_qm_energy)

    orca_energy_parser = subparsers.add_parser(
        "orca-energy",
        help="read the last FINAL SINGLE POINT ENERGY from an ORCA output",
    )
    orca_energy_parser.add_argument("input", help="ORCA output file to read")
    orca_energy_parser.set_defaults(func=_cmd_orca_energy)

    _add_thermo_commands(subparsers)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
