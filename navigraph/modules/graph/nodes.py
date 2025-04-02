# graph_node.py
from typing import Optional, Tuple, Any


class Node:
    """
    A basic graph node containing only an id and a name.
    """

    def __init__(self, node_id: int, name: Optional[str] = None) -> None:
        self.node_id: int = node_id
        self.name: str = name if name is not None else str(node_id)

    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, name={self.name})"


class TileNode(Node):
    """
    A node associated with a physical map tile.

    Attributes:
        tile_id (int): The associated tile id (required).
        virtual_edge (bool): Whether the associated edge is virtual.
        virtual_edge_data (Optional[dict]): Additional data for drawing or processing the virtual edge.
        pos (Optional[Tuple[float, float]]): Optional position.
    """

    def __init__(self,
                 node_id: int,
                 tile_id: int,
                 pos: Optional[Tuple[float, float]] = None,
                 virtual_edge: bool = False,
                 virtual_edge_data: Optional[dict] = None,
                 name: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(node_id, name)
        self.tile_id: int = tile_id  # now required
        self.virtual_edge: bool = virtual_edge
        self.virtual_edge_data: Optional[dict] = virtual_edge_data
        self.pos: Optional[Tuple[float, float]] = pos

    def __repr__(self) -> str:
        return (f"TileNode(id={self.node_id}, tile_id={self.tile_id}, name={self.name}, "
                f"virtual_edge={self.virtual_edge}, virtual_edge_data={self.virtual_edge_data}, pos={self.pos})")
