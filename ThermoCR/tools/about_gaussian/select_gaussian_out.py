"""
截取gaussian的输出文件中关于frequency计算的部分，作为KisTheIP的输入
"""
import glob
import os


def select_gaussian_out(input_path, output_path, task_id=2, select_mode='cut'):
    """
    Selects or cuts a specific part of a Gaussian output file based on the task ID and the selected mode.

    Summary:
    This function reads a Gaussian output file, identifies sections based on 'Normal termination' strings, and then
    either selects or cuts a specified section to write into a new file. The selection or cut is determined by the
    provided task ID and the select_mode which can be either 'cut' or 'select'.

    Parameters:
    - input_path (str): Path to the input Gaussian output file.
    - output_path (str): Path where the processed output will be saved.
    - task_id (int, optional): The ID of the task to process. Defaults to 2.
    - select_mode (str, optional): Mode of operation, either 'cut' or 'select'. Defaults to 'cut'.

    Returns:
    - None

    Raises:
    - AssertionError: If select_mode is not 'cut' or 'select'.
    - NotImplementedError: If an unsupported select_mode is provided.
    """
    assert select_mode in ['cut', 'select']
    f = open(input_path, 'r').readlines()
    end_line_index = []
    for line_index, line in enumerate(f):
        if 'Normal termination' in line:
            end_line_index.append(line_index + 1)

    if task_id == 1:
        select_mode = 'cut'
    task_id -= 1
    if select_mode == 'select':
        out_start_line_index = end_line_index[task_id - 1]
        out_end_line_index = end_line_index[task_id]
        out = f[out_start_line_index:out_end_line_index]
    elif select_mode == 'cut':
        out = f[:end_line_index[task_id]]
    else:
        raise NotImplementedError('select_mode must be cut or select')
    with open(output_path, 'w') as f:
        for line in out:
            f.write(line)

    return None