"""
截取gaussian的输出文件中关于frequency计算的部分，作为KisTheIP的输入
"""
import glob
import os


def select_gaussian_out(input_path, output_path, task_id=2, select_mode='cut'):
    """

    :param input_path:
    :param output_path:
    :param task_id:
    :param select_mode: cut or select, cut是指截断到这个任务，即这个任务之前的任务都要, select是只要这个任务的输出
    :return:
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