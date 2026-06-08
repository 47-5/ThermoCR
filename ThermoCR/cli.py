"""Command-line interface for ThermoCR."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ThermoCR.export import format_cantera_yaml_thermo
from ThermoCR.io.gaussian import select_gaussian_output, split_gaussian_link1_output
from ThermoCR.io.orca import read_orca_final_single_point_energy
from ThermoCR.io.qm_output import read_electronic_energy, read_molecule_data
from ThermoCR.kinetics import calculate_tst_rate_frame
from ThermoCR.thermo import ThermoOptions, fit_thermo_frame, scan_thermo


_THERMO_MODELS = ("NASA7", "NASA9", "Shomate")


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


def _thermo_model_type(value):
    for model_type in _THERMO_MODELS:
        if str(value).lower() == model_type.lower():
            return model_type
    choices = ", ".join(_THERMO_MODELS)
    raise argparse.ArgumentTypeError(f"expected one of: {choices}")


def _float_sequence(value):
    try:
        return [float(item) for item in str(value).replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma- or space-separated floating-point values"
        ) from exc


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


def _read_dataframe(input_path):
    input_path = Path(input_path)
    if input_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path)


def _write_thermo_fit_result(result, output_path):
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        text = format_cantera_yaml_thermo(
            result.model_type,
            result.temperature_range,
            result.parameters,
        )
        output_path.write_text(text, encoding="utf-8")
    elif suffix == ".json":
        text = json.dumps(result.as_dict(), indent=2)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        raise ValueError("thermo fit output must use .json, .yaml, or .yml")
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


def _cmd_thermo_fit(args):
    df = _read_dataframe(args.input)
    result = fit_thermo_frame(
        df,
        model_type=args.model,
        start_index=args.start_index,
        end_index=args.end_index,
        weight_strategy=args.weight_strategy,
        T_range=args.t_range,
        guess=args.guess,
        maxfev=args.maxfev,
    )
    output_path = _write_thermo_fit_result(result, args.output)
    print(output_path)
    return 0


def _cmd_kinetics_tst(args):
    transition_state = _read_dataframe(args.input)
    reactants = [_read_dataframe(path) for path in args.reactant]
    products = None
    if args.product is not None:
        products = [_read_dataframe(path) for path in args.product]
    df = calculate_tst_rate_frame(
        transition_state,
        reactants,
        product_frames=products,
        delta_n=args.delta_n,
        liquid=args.liquid,
        tunnelling_effect=args.tunnelling_effect,
        imaginary_freq=args.imaginary_freq,
        sigma=args.sigma,
        reference_pressure=args.reference_pressure,
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

    fit_parser = thermo_subparsers.add_parser(
        "fit",
        help="fit NASA/Shomate thermo parameters from scan data",
    )
    fit_parser.add_argument("input", help="CSV/XLSX thermo scan data")
    fit_parser.add_argument("--output", required=True, help="JSON/YAML output file")
    fit_parser.add_argument(
        "--model",
        type=_thermo_model_type,
        default="NASA7",
        help="thermo model to fit: NASA7, NASA9, or Shomate; default: NASA7",
    )
    fit_parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="first row index included in the fit; default: 0",
    )
    fit_parser.add_argument(
        "--end-index",
        type=int,
        help="row index after the last row included in the fit",
    )
    fit_parser.add_argument(
        "--weight-strategy",
        choices=("inverse_mean_abs", "uniform"),
        default="inverse_mean_abs",
        help="fit weighting strategy; default: inverse_mean_abs",
    )
    fit_parser.add_argument(
        "--t-range",
        nargs=2,
        type=float,
        metavar=("T_MIN", "T_MAX"),
        help="temperature range written to the fitted model",
    )
    fit_parser.add_argument(
        "--guess",
        type=_float_sequence,
        help="comma-separated initial model parameters for scipy curve_fit",
    )
    fit_parser.add_argument(
        "--maxfev",
        type=_positive_int,
        default=100000,
        help="maximum function evaluations for fitting; default: 100000",
    )
    fit_parser.set_defaults(func=_cmd_thermo_fit)


def _add_kinetics_commands(subparsers):
    kinetics_parser = subparsers.add_parser(
        "kinetics",
        help="structured kinetics workflows",
    )
    kinetics_subparsers = kinetics_parser.add_subparsers(
        dest="kinetics_command",
        required=True,
    )

    tst_parser = kinetics_subparsers.add_parser(
        "tst",
        help="calculate a TST rate scan from thermo tables",
    )
    tst_parser.add_argument("input", help="CSV/XLSX transition-state thermo data")
    tst_parser.add_argument("--output", required=True, help="CSV/XLSX output file")
    tst_parser.add_argument(
        "--reactant",
        action="append",
        required=True,
        help="CSV/XLSX reactant thermo data; repeat for multiple reactants",
    )
    tst_parser.add_argument(
        "--product",
        action="append",
        help="CSV/XLSX product thermo data; repeat for multiple products",
    )
    tst_parser.add_argument(
        "--delta-n",
        type=int,
        help="gas molecule count change; default: number of reactants minus one",
    )
    tst_parser.add_argument(
        "--liquid",
        action="store_true",
        help="use liquid-phase TST units",
    )
    tst_parser.add_argument(
        "--tunnelling-effect",
        choices=("wigner", "eckart", "skodje_truhlar"),
        help="optional tunnelling correction",
    )
    tst_parser.add_argument(
        "--imaginary-freq",
        type=float,
        help="transition-state imaginary frequency in cm^-1",
    )
    tst_parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="reaction path degeneracy; default: 1",
    )
    tst_parser.add_argument(
        "--reference-pressure",
        type=float,
        default=100000.0,
        help="reference pressure in Pa; default: 100000",
    )
    tst_parser.set_defaults(func=_cmd_kinetics_tst)


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
    _add_kinetics_commands(subparsers)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
