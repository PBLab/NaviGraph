import random
import time

import pandas as pd
from omegaconf import DictConfig
from typing import List, Tuple, Union
import hydra
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .graph_tile_dictionary import graph_dict


plt.rcParams["figure.figsize"] = (35, 20)
plt.tight_layout(pad=0)
plt.axis('off')
plt.margins(0)

log = lg.getLogger(__name__)


class Graph(object):
    def __init__(self, cfg: DictConfig):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self.tree = self._build_binary_tree(self.cfg.graph.height)
        self._ax = self.set_axes()
        self.draw_base_tree()

    def set_axes(self):
        _ax = plt.axes()
        _ax.axis('off')
        _ax.figure.tight_layout(pad=0)
        _ax.margins(0)
        return _ax

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def _build_binary_tree(self, height: int, weight=lambda p_node, c_node: 1):
        """
        This function builds a binary tree as a graph where node name consists of 2 numbers the first indicate the level
        starting from 0, and the second the node id withing that level, starting from 0. The weight parameter is a
        function the gets the parent and current node and return a weight int. default is set to constant 1.

        """
        g = nx.Graph()
        x_pos = range(2**height)
        get_x_pos = lambda x: np.convolve(x, np.ones(2), 'valid')/2
        for level in range(height)[::-1]:
            num_nodes_at_level = 2**level
            # Set x position per level
            if level == height - 1:
                x_pos = np.arange(2 ** height)
            else:
                x_pos = get_x_pos(x_pos)[::2]  # rolling mean with step size
            # add nodes and edges with child nodes
            for level_node_id in range(num_nodes_at_level):
                current_node = int(str(level) + str(level_node_id))
                g.add_node(current_node, pos=(x_pos[level_node_id], height - level))
                if level == height - 1:
                    continue

                left_most_child_id, right_most_child_id = level_node_id * 2, (level_node_id * 2) + 1

                left_most_child = int(str(level + 1) + str(left_most_child_id))
                right_most_child = int(str(level + 1) + str(right_most_child_id))
                # parent_node = int(str(level-1) + str(level_node_id//2))
                g.add_edge(current_node, left_most_child, weight=weight(current_node, left_most_child))
                g.add_edge(current_node, right_most_child, weight=weight(current_node, right_most_child))

        return g

    def draw_base_tree(self):
        node_color = [self.cfg.graph.options.static_node_color]*len(self.tree.nodes)
        edge_color = [self.cfg.graph.options.static_edge_color]*len(self.tree.edges)
        width = [1]*len(self.tree.edges)

        nx.draw(self.tree,
                pos=nx.get_node_attributes(self.tree, 'pos'),
                ax=self._ax,
                node_color=node_color,
                edge_color=edge_color,
                width=width,
                **self.cfg.graph.draw)

    def draw_tree(self, node_list: List[int] = None, edge_list: List[tuple] = None, color_mode='current',
                  unique_path: List = None):

        def set_unique_path_color(itr: List, color_list: List):
            if unique_path is not None:
                for ind, item in enumerate(itr):
                    if item in unique_path:
                        if color_mode == 'current':
                            if isinstance(item, tuple):
                                selected_color = self.cfg.graph.options.dynamic_reward_edge_color
                            elif isinstance(item, int):
                                selected_color = self.cfg.graph.options.dynamic_reward_node_color
                            else:
                                raise ValueError('color mode not supported')

                            color_list[ind] = selected_color

                        elif color_mode == 'history':
                            if isinstance(item, tuple):
                                selected_color = self.cfg.graph.options.history_reward_edge_color
                            elif isinstance(item, int):
                                selected_color = self.cfg.graph.options.history_reward_node_color
                            else:
                                raise ValueError('color mode not supported')

                            color_list[ind] = selected_color
                        else:
                            raise ValueError('mode not supported')

            return color_list

        if node_list is not None:
            if color_mode == 'current':
                color = self.cfg.graph.options.dynamic_node_color
            elif color_mode == 'history':
                color = self.cfg.graph.options.history_node_color
            else:
                raise ValueError('color mode not supported')

            node_color = [color] * len(node_list)
            node_color = set_unique_path_color(node_list, node_color)

            nx.draw_networkx_nodes(self.tree,
                                   pos=nx.get_node_attributes(self.tree, 'pos'),
                                   nodelist=node_list,
                                   ax=self._ax,
                                   node_color=node_color,
                                   node_size=self.cfg.graph.draw.node_size)
        if edge_list is not None:
            if color_mode == 'current':
                color = self.cfg.graph.options.dynamic_edge_color
            elif color_mode == 'history':
                color = self.cfg.graph.options.history_edge_color
            else:
                raise ValueError('color mode not supported')

            edge_color = [color] * len(edge_list)
            edge_color = set_unique_path_color(edge_list, edge_color)

            nx.draw_networkx_edges(self.tree,
                                   pos=nx.get_node_attributes(self.tree, 'pos'),
                                   edgelist=edge_list,
                                   ax=self._ax,
                                   edge_color=edge_color,
                                   width=self.cfg.graph.options.edge_width)

    def show_tree(self):
        self._ax.figure.show()

    def tree_fig_to_img(self):
        self._ax.figure.canvas.draw()
        plot_to_image = np.frombuffer(self._ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
        return plot_to_image.reshape(self._ax.figure.canvas.get_width_height()[::-1] + (3,))

    def get_tree_location(self, key: int) -> Union[Tuple, int]:
        return graph_dict.get(key, None)

    def get_shortest_path(self, source=None, target=None, weight=None, method='dijkstra'):
        return nx.shortest_path(self.tree, source=source, target=target, weight=weight, method=method)

    def get_random_walk(self, source, target, disable_backtrack=False):
        """
        return path of a random walker
        :param source:
        :param target:
        :param disable_backtrack:
        :return: list of node walked
        """

        path = [source]
        current_parent = None
        while source != target:
            current_neighbors = list(self.tree.adj[source])

            if disable_backtrack and current_parent in current_neighbors and len(current_neighbors) > 1:
                current_neighbors.remove(current_parent)

            current_parent = source
            source = random.choice(current_neighbors)
            path.append(source)

        return path


@hydra.main(config_path="../configs", config_name="maze_master_basic")
def main(cfg: DictConfig):
    g = Graph(cfg)

    # Random walk usage example
    l = []
    for i in range(5000):
        rand_walk = g.get_random_walk(0, 637, disable_backtrack=False)
        l.append(len(rand_walk))
    print(np.mean(l))
    print(np.std(l))

    # g.show_tree()
    g.draw_tree(node_list=[0], edge_list=[(0, 11)])
    im = g.tree_fig_to_img()
    g.draw_tree(node_list=[10])
    im = g.tree_fig_to_img()

    pass


if __name__ == '__main__':
    main()






