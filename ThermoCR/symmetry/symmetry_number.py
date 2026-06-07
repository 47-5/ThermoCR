"""Rotational symmetry number helpers."""


def rotational_symmetry_number(point_group):
    """Return the rotational symmetry number for a point-group symbol."""
    if point_group == "Dinfh":
        return 2
    if point_group == "Cinfv":
        return 1

    cubic_groups = {
        "T": 12,
        "Th": 12,
        "Td": 12,
        "O": 24,
        "Oh": 24,
        "I": 60,
        "Ih": 60,
    }
    if point_group in cubic_groups:
        return cubic_groups[point_group]

    if len(point_group) >= 2 and point_group[1].isdigit():
        n = int("".join(filter(str.isdigit, point_group)))
        if point_group.startswith("C") or point_group.startswith("S"):
            return n
        if point_group.startswith("D"):
            return 2 * n

    special_groups = {
        "C1": 1,
        "Cs": 1,
        "Ci": 1,
    }
    if point_group in special_groups:
        return special_groups[point_group]

    raise ValueError(f"unknown: {point_group}")
