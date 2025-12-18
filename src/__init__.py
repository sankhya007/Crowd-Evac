"""
Crowd Simulation & Multi-Hazard Evacuation System
A comprehensive agent-based simulation platform for crowd dynamics and emergency evacuation
"""

__version__ = "1.0.0"
__author__ = "Crowd Simulation Team"

from .agent import Agent
from .environment import Environment, Exit, Obstacle, Grid
from .motion_models import SocialForceModel, RVO, AStarPathfinder, MotionController
from .hazard_manager import HazardManager
from .analytics import AnalyticsCollector
from .visualizer import Visualizer
from .simulation_engine import SimulationEngine
from .floorplan_parser import DXFParser, ImageParser, MapMeta, load_floorplan

__all__ = [
    'Agent',
    'Environment',
    'Exit',
    'Obstacle',
    'Grid',
    'SocialForceModel',
    'RVO',
    'AStarPathfinder',
    'MotionController',
    'HazardManager',
    'AnalyticsCollector',
    'Visualizer',
    'SimulationEngine',
    'DXFParser',
    'ImageParser',
    'MapMeta',
    'load_floorplan'
]
