import glob
import os
import re
import pandas as pd


def read_orca_wB97Mp2_out(orca_out_file_path):
    pattern = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)')

    matches = [pattern.search(line) for line in open(orca_out_file_path, 'r')]
    matches = [float(i.group(1)) for i in matches if i]
    result = matches[-1]
    return result


# 自定义排序键函数
def sort_key(file_name):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'(\d+)_(\d+)_path(\d+)_(\d+)', file_name)
    if match:
        # 提取匹配到的数字部分
        part1, part2, part3, part4 = map(int, match.groups())
        return (part1, part2, part3, part4)
    else:
        # 如果没有匹配到，返回一个较大的值以便排在最后
        return (float('inf'), float('inf'), float('inf'), float('inf'))



# if __name__ == '__main__':
#
#     run_mode = 'batch'
#     sort = True
#
#     if run_mode == 'single':
#         read_orca_wB97Mp2_out(orca_out_file_path='done/01_02_path1_1.out')
#
#     elif run_mode == 'batch':
#         orca_out_file_path_list = glob.glob('species_wB97Mp2_QZVPP/*.out')
#         if sort:
#             orca_out_file_path_list = sorted(orca_out_file_path_list, key=sort_key)
#         print(orca_out_file_path_list)
#
#         # todo name根据实际情况修改
#         name = [os.path.basename(i) for i in orca_out_file_path_list]
#         name = [i.split('.')[0] for i in name]
#         print(name)
#
#         result = []
#         for i in orca_out_file_path_list:
#             result.append(read_orca_wB97Mp2_out(i))
#
#         df = pd.DataFrame({'name': name, 'wB97Mp2': result})
#         df.to_excel('wB97Mp2_QZVPP_out.xlsx', index=False)