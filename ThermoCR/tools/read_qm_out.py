import cclib
from ase.units import Hartree

from ThermoCR.tools.about_gaussian.link1 import (
    is_gaussian_link1_output,
    read_gaussian_link1_job,
)


def read_qm_out(filepath, gaussian_job_index=None, prefer_link1_split=True):
    """
    Reads and parses a quantum mechanics output file.

    Summary:
    This function takes a filepath to a quantum mechanics (QM) software output
    file, reads it, and returns the parsed data. It utilizes the cclib library
    to handle various QM software output formats, making the function versatile
    for different types of QM calculations.

    Parameters:
    - filepath: (str)
            The path to the QM software output file to be read and parsed.
    - gaussian_job_index: (int, optional)
            Gaussian Link1 job section to read. Positive values are 1-based;
            negative values follow Python indexing. By default, Link1 outputs
            use the last normally terminated job.
    - prefer_link1_split: (bool, default True)
            If True, Gaussian Link1 outputs are split into single-job sections
            before being passed to cclib.

    Returns:
        (cclib.parser.data.ccData)
            An object containing the parsed data from the QM output file.

    Raises:
    - IOError: If there is an issue with reading the file.
    - cclib.parser.ParseError: If there is an error in parsing the file content.
    """
    if prefer_link1_split and is_gaussian_link1_output(filepath):
        job_index = -1 if gaussian_job_index is None else gaussian_job_index
        return read_gaussian_link1_job(filepath, job_index=job_index)

    try:
        data = cclib.io.ccread(filepath)
    except Exception:
        if prefer_link1_split:
            job_index = -1 if gaussian_job_index is None else gaussian_job_index
            return read_gaussian_link1_job(filepath, job_index=job_index)
        raise
    return data


def read_atom_coord(filepath, coord_index=-1, gaussian_job_index=None):
    """

    Reads atomic coordinates from a computational chemistry output file.

    Detailed summary:
    This function reads and returns the atomic numbers and coordinates from
    a given computational chemistry output file. The coordinates can be
    selected by their index in the list of available geometries, with the
    default being the last geometry (usually the optimized structure).

    Parameters:
    - filepath (str): Path to the computational chemistry output file.
    - coord_index (int, optional): Index of the coordinate set to read,
                                 default is -1 which selects the last set.
    - gaussian_job_index (int, optional): Gaussian Link1 job section to read.

    Returns:
    tuple: A tuple containing two elements:
           1. A list of atomic numbers.
           2. A numpy array of the corresponding atomic coordinates.

    Raises:
    - IOError: If the specified file does not exist or cannot be read.
    - ValueError: If the file format is not supported or if the coord_index
                is out of range for the number of geometries in the file.

    """
    data = read_qm_out(filepath, gaussian_job_index=gaussian_job_index)
    atom_numbers = data.atomnos
    coords = data.atomcoords[coord_index]
    return atom_numbers, coords


def read_vib(filepath, gaussian_job_index=None):
    """
    Reads and returns vibrational frequencies from a computational chemistry output file.

    Summary:
    This function reads a specified computational chemistry output file using the cclib library,
    extracts the vibrational frequencies, and returns them as a list. If the number of atoms in the
    file is 1 or less, an empty list is returned instead, as vibrational analysis is not meaningful
    for such systems.

    Parameters:
    - filepath (str): The path to the computational chemistry output file to be read.
    - gaussian_job_index (int, optional): Gaussian Link1 job section to read.

    Returns:
        (List[float]): A list containing the vibrational frequencies extracted from the file.
                     Returns an empty list if the file contains 1 or fewer atoms.

    Raises:
    - IOError: If there is an issue reading the file at the provided filepath.
    - ValueError: If the data does not contain expected attributes like 'vibfreqs' or 'atomnos'.
    """
    data = read_qm_out(filepath, gaussian_job_index=gaussian_job_index)
    if len(data.atomnos) <= 1:
        return []
    vibfreqs = data.vibfreqs
    return vibfreqs


def read_imaginary_vib(filepath, vibfreqs=None, gaussian_job_index=None):
    """
    Reads and processes vibration frequencies from a given file, focusing on identifying the most significant imaginary (negative) frequency.

    The function first reads vibration frequencies from the provided file. If `vibfreqs` is not provided, it calls an
    external `read_vib` function to read them. It then converts all read frequencies into floats and filters out the
    negative ones, which indicate imaginary frequencies. Among these, if there are multiple, it selects the one with
    the largest absolute value as the most significant. If no negative frequencies are found, it returns None.

    Parameters:
    - filepath (str): Path to the file containing vibration frequencies.
    - vibfreqs (Optional[List[Union[int, float, str]]]): A list of vibration frequencies. If not provided, they will be read from `filepath`.
    - gaussian_job_index (int, optional): Gaussian Link1 job section to read.

    Returns:
        Optional[float]: The selected imaginary frequency with the largest absolute value, or None if no imaginary frequencies were found.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - ValueError: If the file contains non-convertible elements when trying to convert to float.
    """
    if vibfreqs is None:
        vibfreqs = read_vib(filepath=filepath, gaussian_job_index=gaussian_job_index)

    # 将所有元素转换为浮点数
    vibfreqs_float = [float(freq) for freq in vibfreqs]

    # 筛选出所有的负数频率
    negative_freqs = [freq for freq in vibfreqs_float if freq < 0.0]

    if len(negative_freqs) > 1:
        # 当存在多个负数频率时，选取绝对值最大的那一个
        selected_freq = min(negative_freqs)
        print("警告：检测到多个虚频，已选择绝对值最大的一个。")
    elif len(negative_freqs) == 1:
        # 只有一个负数频率的情况
        selected_freq = negative_freqs[0]
    else:
        # 没有找到负数频率
        selected_freq = None
        print("注意：没有检测到虚频。")

    return selected_freq


def read_ee(filepath, ee_index=-1, return_Hartree=True, gaussian_job_index=None):
    """
    Reads the electronic energy from a computational chemistry output file.

    The function reads an output file generated by a computational chemistry
    software, extracts the electronic energy, and optionally converts it to Hartree units.
    It supports reading the last or any specific electronic energy from the file.

    Parameters:
    - filepath (str): The path to the computational chemistry output file.
    - ee_index (int, optional): The index of the electronic energy to read. Defaults to -1,
            which corresponds to the last electronic energy in the file.
    - return_Hartree (bool, optional): If True, the electronic energy is returned in Hartree units.
            If False, the energy is returned in the original units of the file. Defaults to True.
    - gaussian_job_index (int, optional): Gaussian Link1 job section to read.

    Returns:
        float: The electronic energy. The unit depends on the `return_Hartree` parameter.

    Raises:
    - FileNotFoundError: If the file at `filepath` does not exist.
    - ValueError: If `ee_index` is out of range for the number of electronic energies in the file.
    """
    data = read_qm_out(filepath, gaussian_job_index=gaussian_job_index)
    ee = data.scfenergies[ee_index]
    if return_Hartree:
        ee /= Hartree
    return ee
