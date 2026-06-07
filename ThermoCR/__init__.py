"""ThermoCR public API."""

from importlib import import_module as _import_module

from ThermoCR._version import __version__

_EXPORT_MODULES = (
    "ThermoCR.constants",
    "ThermoCR.elements",
    "ThermoCR.export",
    "ThermoCR.io",
    "ThermoCR.kinetics",
    "ThermoCR.pointgroup",
    "ThermoCR.simulation",
    "ThermoCR.symmetry",
    "ThermoCR.thermo",
    "ThermoCR.tools.about_gaussian",
    "ThermoCR.tools.about_orca",
)

__all__ = ["__version__"]


def _export_public_api(module_name):
    module = _import_module(module_name)
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in vars(module) if not name.startswith("_")]

    for name in names:
        globals()[name] = getattr(module, name)
        if name not in __all__:
            __all__.append(name)


for _module_name in _EXPORT_MODULES:
    _export_public_api(_module_name)

# Backward-compatible aliases for old top-level utility imports.
get_point_group = detect_point_group
get_I = principal_moments
check_linear = is_linear
get_rotational_symmetry_number = rotational_symmetry_number

for _alias_name in (
    "get_point_group",
    "get_I",
    "check_linear",
    "get_rotational_symmetry_number",
):
    if _alias_name not in __all__:
        __all__.append(_alias_name)

del _EXPORT_MODULES, _alias_name, _export_public_api, _import_module, _module_name
