"""
Network infrastructure for DGD+LOCAL implementation.
"""

from .topology import TopologyGenerator
from .consensus import ConsensusWeights
from .node import DGDNode

__all__ = ['TopologyGenerator', 'ConsensusWeights', 'DGDNode']