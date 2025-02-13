import os
import numpy as np
from omegaconf import DictConfig
import logging as lg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


log = lg.getLogger(__name__)

MATPLOTLIB_BACKEND = 'matplotlib'

PLOT_GENERATOR_CFG_KEY = 'plots'
PLOT_GENERATOR_COMPONENT_CFG_KEY = 'plot_generator'
FUNC_NAME = 'func_name'
FUNC_ARGS = 'args'


def generate_plot(func):
    def wrapper(*args, **kwargs):
        func(*args, data=args[0].data, **kwargs)
    return wrapper


class PlotGenerator(object):

    def __init__(self, cfg: DictConfig, data: pd.DataFrame):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self.data = data

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def save_fig(self, fig, path, backend=MATPLOTLIB_BACKEND):
        if backend == MATPLOTLIB_BACKEND:
            fig.savefig(path)
        else:
            raise NotImplementedError('Only matplotlib backend is supported at the moment')

    def show_fig(self, fig, backend=MATPLOTLIB_BACKEND):
        if backend == MATPLOTLIB_BACKEND:
            fig.show()
        else:
            raise NotImplementedError('Only matplotlib backend is supported at the moment')

    @generate_plot
    def topological_distance_to_reward(self, data: pd.DataFrame, show=False, save=False, path=None):
        plt.ioff()
        fig = plt.figure()
        ax = sns.boxplot(data=data,
                         x='Group',
                         y='Topological distance to reward (# of nodes)',
                         palette=['#a6bddb', '#1c9099'],
                         boxprops=dict(alpha=.3, edgecolor="black", linewidth=1),
                         order=["apoE3_female", "apoE4_female", "apoE3_male", "apoE4_male"],
                         medianprops={"color": "grey"},
                         whiskerprops={"color": "grey"}, capprops={"color": "grey"})

        ax.set(title='Topological distance 72h memory session')

        sns.stripplot(data=data,
                      x='Group',
                      y='Topological distance to reward (# of nodes)',
                      palette=['grey', 'black'],
                      hue='Gender',
                      size=10,
                      alpha=0.7,
                      order=["apoE3_female", "apoE4_female", "apoE3_male", "apoE4_male"],
                      edgecolor=".01",
                      linewidth=0.5,
                      ax=ax)

        if save:
            self.save_fig(fig, path)

        if show:
            self.show_fig(fig)

    def generate(self):
        feature_dict = self.cfg.get(PLOT_GENERATOR_COMPONENT_CFG_KEY, {}).get(PLOT_GENERATOR_CFG_KEY, {})
        for feature_name, func_dict in feature_dict.items():
            func_name = func_dict[FUNC_NAME]
            func_args = func_dict[FUNC_ARGS] if FUNC_ARGS in func_dict else {}
            getattr(self, func_name)(**func_args)

