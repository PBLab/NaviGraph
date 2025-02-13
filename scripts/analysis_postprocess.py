import os.path
import re
from enum import Enum
from functools import reduce
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Mode(Enum):
    LEARNING = 'learning'
    MEMORY = 'memory'


ANALYSIS_MODE = Mode.LEARNING


# Constants
ORDER = ['apoE3_4m', 'apoE4_4m']  # configure order manually

GENOTYPE_APOE3 = 'apoE3'
GENOTYPE_APOE4 = 'apoE4'
MALE = 'male'
FEMALE = 'female'
NUM_ITERATIONS_IN_LEARNING_SESSION = 4
TREE_HEIGHT = 7
TOTAL_NUM_NODES = 2**TREE_HEIGHT - 1

random_walker_mean_nodes_on_path = 1298.1424
perfect_route = 6
perfect_wandering_index = 0.05511811
random_wandering_index = 0.57125

# Export CSV path
EXPORT = True
EXPORT_CSV_PATH = '/home/elior/'

# Load data
DATA_PKL_PATHS = ['/home/elior/hdd/maze_master/outputs/new_pkl_outputs/analysis_results.pkl']

# SAVE PLOTS
SAVE_PNG = True
SAVE_SVG = True


def get_values_from_list(data: pd.Series, feature: str, num_iterations: int = NUM_ITERATIONS_IN_LEARNING_SESSION):
    out = [np.nan] * num_iterations
    for ind, val in enumerate(data[feature]):
        out[ind] = val
    return out


def lineplot_wrapper(data_df: pd.DataFrame,
                     x: str,
                     y: str,
                     hue: str,
                     title: str,
                     save_as_filename: str = None,
                     show: bool = False,
                     xticks: list = None):

    fig = plt.figure()
    g = sns.lineplot(data=data_df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette='viridis')
    if xticks is not None:
        g.set(xticks=xticks)
    g.set(title=title)

    if save_as_filename is not None:
        if SAVE_PNG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.png'))
        if SAVE_SVG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.svg'))

    if show:
        fig.show()


def plot_explored_tiles(data_df: pd.DataFrame,
                        x: str,
                        hue: str,
                        title: str,
                        y: str = None,
                        save_as_filename: str = None,
                        xticks: list = None,
                        show: bool = False):
    fig = plt.figure()
    g = sns.lineplot(data=data_df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette='viridis')

    g.set(xticks=xticks, title='Time to rewards learning session')

    if save_as_filename is not None:
        if SAVE_PNG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.png'))
        if SAVE_SVG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.svg'))

    if show:
        fig.show()


def boxplot_with_striplot(data_df: pd.DataFrame,
                          x: str,
                          y: str,
                          hue: str,
                          title: str,
                          order: Optional[list[str]] = None,
                          save_as_filename: str = None,
                          show: bool = False,
                          y_lim: tuple = None,
                          axhlines: list = None,):


    fig = plt.figure()
    g = sns.boxplot(data=data_df,
                    x=x,
                    y=y,
                    palette=['#a6bddb', '#1c9099'],
                    order=order,
                    boxprops=dict(alpha=.3, edgecolor="black", linewidth=1),
                    medianprops={"color": "grey"},
                    whiskerprops={"color": "grey"},
                    capprops={"color": "grey"}).set(title=title)

    g = sns.stripplot(data=data_df,
                      x=x,
                      y=y,
                      palette=['grey', 'black'],
                      hue=hue,
                      size=10,
                      order=order,
                      alpha=0.7,
                      edgecolor=".01",
                      linewidth=0.5)

    if y_lim is not None:
        g.set(ylim=y_lim)

    if axhlines is not None:
        for axhline in axhlines:
            g.axhline(**axhline)

    if save_as_filename is not None:
        if SAVE_PNG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.png'))
        if SAVE_SVG:
            fig.savefig(os.path.join(EXPORT_CSV_PATH, save_as_filename + '.svg'))

    if show:
        fig.show()


df_list = [pd.read_pickle(file) for file in DATA_PKL_PATHS]
df = pd.concat(df_list, ignore_index=False)

analysis_data = []
exploration_frame_data = []
for mouse_id, (session_name, df_session) in enumerate(df.iterrows()):
    mouse = int(session_name.split('_')[1])
    genotype = GENOTYPE_APOE3 if GENOTYPE_APOE3 in session_name else GENOTYPE_APOE4
    age = [sub_str for sub_str in session_name.split('_') if re.match('[0-9]m', sub_str) is not None][0]
    gender = FEMALE if FEMALE in session_name else MALE

    group = genotype + '_' + age
    # group = genotype + '_' + gender # uncomment to change group formation to genotype + gender

    # get time to reward [s]
    time_to_reward_s_1, time_to_reward_s_2, time_to_reward_s_3, time_to_reward_s_4 = \
        get_values_from_list(df_session, 'time_to_reward')

    # get time to reward [m]
    time_to_reward_m_1 = time_to_reward_s_1 / 60 if ~np.isnan(time_to_reward_s_1) else np.nan
    time_to_reward_m_2 = time_to_reward_s_2 / 60 if ~np.isnan(time_to_reward_s_2) else np.nan
    time_to_reward_m_3 = time_to_reward_s_3 / 60 if ~np.isnan(time_to_reward_s_3) else np.nan
    time_to_reward_m_4 = time_to_reward_s_4 / 60 if ~np.isnan(time_to_reward_s_4) else np.nan
    time_in_maze = np.nansum([time_to_reward_m_1, time_to_reward_m_2, time_to_reward_m_3, time_to_reward_m_4])

    # velocity to reward [m/s]
    velocity_to_reward_m_s_1, velocity_to_reward_m_s_2, velocity_to_reward_m_s_3, velocity_to_reward_m_s_4 = \
        get_values_from_list(df_session, 'velocity_to_reward')

    # topological distance to reward to reward [num nodes]
    topological_distance_to_reward_s_1, topological_distance_to_reward_s_2, topological_distance_to_reward_s_3, topological_distance_to_reward_s_4 = \
        get_values_from_list(df_session, 'topological_distance_to_reward')

    # wandering index
    wandering_index_1, wandering_index_2, wandering_index_3, wandering_index_4 = \
        get_values_from_list(df_session, 'exploration_to_reward')

    # eureka path length
    eureka_path_length_1, eureka_path_length_2, eureka_path_length_3, eureka_path_length_4 = \
        get_values_from_list(df_session, 'eureka_path_length')

    # exploration frame data
    exploration_frame_data.append(df_session.exploration_percentage.rename(session_name))

    # set data
    new_line = {'id': mouse_id,
                'Mouse': mouse,
                'Group': group,
                'Gender': gender,
                'Genotype': genotype,
                'Age': age,
                'Time to reward 1 [s]': time_to_reward_s_1,
                'Time to reward 2 [s]': time_to_reward_s_2,
                'Time to reward 3 [s]': time_to_reward_s_3,
                'Time to reward 4 [s]': time_to_reward_s_4,
                'Time to reward 1 [min]': time_to_reward_m_1,
                'Time to reward 2 [min]': time_to_reward_m_2,
                'Time to reward 3 [min]': time_to_reward_m_3,
                'Time to reward 4 [min]': time_to_reward_m_4,
                'Time in maze [min]': time_in_maze,
                'Velocity 1 [m/s]': velocity_to_reward_m_s_1,
                'Velocity 2 [m/s]': velocity_to_reward_m_s_2,
                'Velocity 3 [m/s]': velocity_to_reward_m_s_3,
                'Velocity 4 [m/s]': velocity_to_reward_m_s_4,
                'Topological distance 1 [# nodes in path]': topological_distance_to_reward_s_1,
                'Topological distance 2 [# nodes in path]': topological_distance_to_reward_s_2,
                'Topological distance 3 [# nodes in path]': topological_distance_to_reward_s_3,
                'Topological distance 4 [# nodes in path]': topological_distance_to_reward_s_4,
                'Wandering index 1 [# novel nodes in path/total nodes]': wandering_index_1,
                'Wandering index 2 [# novel nodes in path/total nodes]': wandering_index_2,
                'Wandering index 3 [# novel nodes in path/total nodes]': wandering_index_3,
                'Wandering index 4 [# novel nodes in path/total nodes]': wandering_index_4,
                '# nodes on direct path to reward 1': eureka_path_length_1,
                '# nodes on direct path to reward 2': eureka_path_length_2,
                '# nodes on direct path to reward 3': eureka_path_length_3,
                '# nodes on direct path to reward 4': eureka_path_length_4,}

    analysis_data.append(new_line)

analysis_df = pd.DataFrame(analysis_data)
exploration_df = pd.DataFrame(exploration_frame_data).T
# exploration_df.plot()

# prepare long format data for plotting
id_vars = ["id", "Mouse", "Group", "Gender", "Genotype", "Age", "Time in maze [min]"]

sets = {'Time to rewards': ['Time to reward 1 [min]', 'Time to reward 2 [min]', 'Time to reward 3 [min]', 'Time to reward 4 [min]'],
        'Velocity to rewards': ['Velocity 1 [m/s]', 'Velocity 2 [m/s]', 'Velocity 3 [m/s]', 'Velocity 4 [m/s]'],
        'Topological distance to rewards': ['Topological distance 1 [# nodes in path]', 'Topological distance 2 [# nodes in path]', 'Topological distance 3 [# nodes in path]', 'Topological distance 4 [# nodes in path]'],
        'Wandering index to rewards': ['Wandering index 1 [# novel nodes in path/total nodes]', 'Wandering index 2 [# novel nodes in path/total nodes]', 'Wandering index 3 [# novel nodes in path/total nodes]', 'Wandering index 4 [# novel nodes in path/total nodes]'],
        '# nodes on direct path to rewards': ['# nodes on direct path to reward 1', '# nodes on direct path to reward 2', '# nodes on direct path to reward 3', '# nodes on direct path to reward 4']
        }
df_long_list = []

for key, val in sets.items():
    df_temp = analysis_df[id_vars + val]
    df_temp = df_temp.rename(columns={val[0]: 1, val[1]: 2, val[2]: 3, val[3]: 4})
    df_long_list.append(pd.melt(df_temp, id_vars=id_vars, value_vars=[1, 2, 3, 4], value_name=key, var_name='Reward #'))

df_long = reduce(lambda left, right: pd.merge(left, right), df_long_list)


if EXPORT:
    analysis_df.to_csv(os.path.join(EXPORT_CSV_PATH, 'analysis.csv'))
    exploration_df.to_csv(os.path.join(EXPORT_CSV_PATH, 'exploration.csv'))
    df_long.to_csv(os.path.join(EXPORT_CSV_PATH, 'analysis_long_format.csv'))

# Plots
if ANALYSIS_MODE == Mode.LEARNING:
    rewards = [1, 2, 3, 4]
    # Time to reward line plot
    lineplot_wrapper(df_long,
                     x='Reward #',
                     y='Time to rewards',
                     hue='Group',
                     title='Time to rewards learning session',
                     save_as_filename='time_to_rewards_learning_session',
                     )

    # Velocity to reward line plot
    lineplot_wrapper(df_long,
                     x='Reward #',
                     y='Velocity to rewards',
                     hue='Group',
                     title='Velocity to rewards learning session',
                     save_as_filename='velocity_to_rewards_learning_session',
                     )

    # Topological distance to reward line plot
    lineplot_wrapper(df_long,
                     x='Reward #',
                     y='Topological distance to rewards',
                     hue='Group',
                     title='Topological distance to rewards learning session',
                     save_as_filename='topological_distance_to_rewards_learning_session',
                     )

    # Wandering index to reward line plot
    lineplot_wrapper(df_long,
                     x='Reward #',
                     y='Wandering index to rewards',
                     hue='Group',
                     title='Wandering index to rewards learning session',
                     save_as_filename='wandering_index_to_rewards_learning_session',
                     )

    # Eureka direct path length to reward line plot
    lineplot_wrapper(df_long,
                     x='Reward #',
                     y='# nodes on direct path to rewards',
                     hue='Group',
                     title='# nodes on direct path to rewards learning session',
                     save_as_filename='eureka_direct_path_length_to_rewards_learning_session',
                     )

    # Time in maze
    boxplot_with_striplot(df_long,
                          x='Group',
                          y='Time in maze [min]',
                          hue='Gender',
                          order=ORDER,
                          title='Time in maze learning session',
                          save_as_filename='time_in_maze_learning')

    for i in range(NUM_ITERATIONS_IN_LEARNING_SESSION):
        _id = i+1
        # Time to reward
        boxplot_with_striplot(analysis_df,
                              x='Group',
                              y=f'Time to reward {_id} [min]',
                              hue='Genotype',
                              order=ORDER,
                              title=f'Time to reward {_id} learning session',
                              save_as_filename=f'time_to_reward_learning_{_id}',
                              # show=True
                              )
        # Velocity to reward
        boxplot_with_striplot(analysis_df,
                              x='Group',
                              y=f'Velocity {_id} [m/s]',
                              hue='Genotype',
                              order=ORDER,
                              title=f'Velocity {_id} to reward learning session',
                              save_as_filename=f'velocity_to_reward_learning_{_id}')
        # Topological to reward
        boxplot_with_striplot(analysis_df,
                              x='Group',
                              y=f'Topological distance {_id} [# nodes in path]',
                              hue='Genotype',
                              order=ORDER,
                              title=f'Topological distance to reward {_id} learning session',
                              save_as_filename=f'topological_distance_to_reward_learning_{_id}',
                              axhlines=[dict(y=random_walker_mean_nodes_on_path, color='Blue', linestyle='--'),
                                        dict(y=perfect_route, color='Green', linestyle='--')])

        # Wandering index
        boxplot_with_striplot(analysis_df,
                              x='Group',
                              y=f'Wandering index {_id} [# novel nodes in path/total nodes]',
                              hue='Genotype',
                              order=ORDER,
                              title=f'Wandering index {_id} learning session',
                              save_as_filename=f'wandering_index_learning_{_id}',
                              y_lim=(0, 1.1),
                              axhlines=[dict(y=random_wandering_index, color='Blue', linestyle='--'),
                                        dict(y=perfect_wandering_index, color='Green', linestyle='--')])

        # Eureka path length (add filter here)
        boxplot_with_striplot(analysis_df,
                              x='Group',
                              y=f'# nodes on direct path to reward {_id}',
                              hue='Genotype',
                              order=ORDER,
                              title=f'Length of direct path to reward {_id} learning session',
                              save_as_filename=f'eureka_path_length_learning_{_id}')


if ANALYSIS_MODE == Mode.MEMORY:
    # Time to reward
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='Time to reward 1 [min]',
                          hue='Genotype',
                          order=ORDER,
                          title='Time to reward memory session',
                          save_as_filename='time_to_reward_memory',
                          # show=True
                          )
    # Time in maze
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='Time in maze [min]',
                          hue='Gender',
                          order=ORDER,
                          title='Time in maze memory session',
                          save_as_filename='time_in_maze_memory')

    # Velocity to reward
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='Velocity 1 [m/s]',
                          hue='Genotype',
                          order=ORDER,
                          title='Velocity to reward memory session',
                          save_as_filename='velocity_to_reward_memory')

    # Topological to reward
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='Topological distance 1 [# nodes in path]',
                          hue='Genotype',
                          order=ORDER,
                          title='Topological distance to reward memory session',
                          save_as_filename='topological_distance_to_reward_memory',
                          axhlines=[dict(y=random_walker_mean_nodes_on_path, color='Blue', linestyle='--'),
                                    dict(y=perfect_route, color='Green', linestyle='--')])

    # Wandering index
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='Wandering index 1 [# novel nodes in path/total nodes]',
                          hue='Genotype',
                          order=ORDER,
                          title='Wandering index memory session',
                          save_as_filename='wandering_index_memory',
                          y_lim=(0, 1.1),
                          axhlines=[dict(y=random_wandering_index, color='Blue', linestyle='--'),
                                    dict(y=perfect_wandering_index, color='Green', linestyle='--')])

    # Eureka path length (add filter here)
    boxplot_with_striplot(analysis_df,
                          x='Group',
                          y='# nodes on direct path to reward 1',
                          hue='Genotype',
                          order=ORDER,
                          title='Length of direct path to reward memory session',
                          save_as_filename='eureka_path_length_memory')

    # Plot exploration percentage
    plot_explored_tiles(exploration_df,
                        x=exploration_df.index,
                        hue='Genotype',
                        title='Eureka path length',
                        save_as_filename='eureka_path_length_memory')




