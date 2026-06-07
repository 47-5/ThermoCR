"""Backward-compatible temperature-program simulation namespace."""

from ThermoCR.simulation.temperature_program import (
    export_t_y_T,
    load_config,
    parser_T_program,
    plot_t_y_T,
    run_temperature_simulation,
)

__all__ = [
    "export_t_y_T",
    "load_config",
    "parser_T_program",
    "plot_t_y_T",
    "run_temperature_simulation",
]
