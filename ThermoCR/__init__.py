"""ThermoCR public API."""

from ThermoCR._version import __version__
from ThermoCR.constants import *
from ThermoCR.elements import *
from ThermoCR.export import *
from ThermoCR.io import *
from ThermoCR.kinetics import *
from ThermoCR.pointgroup import *
from ThermoCR.simulation import *
from ThermoCR.symmetry import *
from ThermoCR.thermo import *

# Backward-compatible aliases for old top-level utility imports.
get_point_group = detect_point_group
get_I = principal_moments
check_linear = is_linear
get_rotational_symmetry_number = rotational_symmetry_number

# Legacy helper names retained at the top level for compatibility.
from ThermoCR.tools.about_gaussian import *
from ThermoCR.tools.about_orca import *