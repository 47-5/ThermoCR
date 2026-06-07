import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ThermoCR.QMconcvar.constant_temperature_simulator import ChemicalKineticsSimulator


def run_temperature_simulation(config_file_path):
    config = load_config(config_file=config_file_path)
    T_program = config['system']['T_program']
    print(T_program)
    T_list, t_start_list, t_end_list = parser_T_program(T_program_config=T_program)

    # 当前初始浓度
    current_concentrations = config['system']['initial_concentrations'].copy()
    # run
    all_results = []
    t_offset = 0
    for simulator_index, (T, t_start, t_end) in enumerate(zip(T_list, t_start_list, t_end_list)):
        print(f'simulator_index: {simulator_index} | T: {T} | t_start: {t_start} | t_end: {t_end}')

        simulator = ChemicalKineticsSimulator(config_file=config_file_path,
                                              override_temperature=T,
                                              override_time_span=[0, t_end-t_start],
                                              override_initial_concentrations=current_concentrations
                                              )
        results = simulator.simulate()
        final_concentrations = results['y'][:, -1]
        current_concentrations = {
            species: final_concentrations[j]
            for j, species in enumerate(simulator.species)
        }
        results['t'] += t_offset
        all_results.append(results)
        t_offset += (t_end - t_start)
        results['T'] = [T] * len(results['t'])

    # combine results
    all_t = np.concatenate([results['t'] for results in all_results])
    all_y = np.hstack([results['y'] for results in all_results])
    all_T = np.concatenate([results['T'] for results in all_results])
    return all_results, all_t, all_y, all_T


def load_config(config_file):
    """加载YAML配置文件"""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def parser_T_program(T_program_config):
    T_list = []
    t_start_list = []
    t_end_list = []
    for T_segment_index, T_segment in enumerate(T_program_config):
        if T_segment['type'] == 'constant':
            T_list.append(T_segment['T_start'])
            t_start_list.append(T_segment['t_start'])
            t_end_list.append(T_segment['t_end'])
        if T_segment['type'] == 'linear':
            T_s = list(np.linspace(start=T_segment['T_start'], stop=T_segment['T_end'], num=T_segment['segments']))
            t_start_s = list(np.linspace(start=T_segment['t_start'], stop=T_segment['t_end'], num=T_segment['segments'] + 1)[:-1])
            t_end_s = list(np.linspace(start=T_segment['t_start'], stop=T_segment['t_end'], num=T_segment['segments'] + 1)[1:])
            T_list += T_s
            t_start_list += t_start_s
            t_end_list += t_end_s

    return T_list, t_start_list, t_end_list


def plot_t_y_T(all_t, all_y, all_T, add_T=True, save_path='t_y_T.png'):
    plt.figure(figsize=(10, 6))
    for i, species in enumerate(all_y):
        plt.plot(all_t, all_y[i], label=f"S{i + 1}")
    plt.xlabel("Time / s")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()

    if add_T:
        plt.twinx()
        plt.plot(all_t, all_T, 'r--', label="Temperature", alpha=0.6)
        plt.ylabel("Temperature (K)")
        plt.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=1000)
    else:
        plt.show()
    return None


def export_t_y_T(all_t, all_y, all_T, save_path='t_y_T.csv'):
    results = {'t': all_t, 'T': all_T}
    for index, y in enumerate(all_y):
        results[f'species{index+1}'] = y
    df = pd.DataFrame(results)
    df.to_csv(save_path)
    return df




