"""
Environment and Grid System
Manages spatial representation, obstacles, exits, and spatial indexing
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import KDTree


class Exit:
    """Represents an exit point."""
    
    def __init__(self, exit_id: int, position: np.ndarray, width: float, capacity: int):
        self.id = exit_id
        self.position = np.array(position, dtype=float)
        self.width = width
        self.capacity = capacity
        self.status = 'open'  # 'open', 'blocked', 'closed'
        self.agent_count = 0
        self.total_evacuated = 0
        
    def is_agent_at_exit(self, agent_position: np.ndarray) -> bool:
        """Check if agent has reached this exit."""
        dist = np.linalg.norm(agent_position - self.position)
        return dist < (self.width / 2 + 1.0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for agent perception."""
        return {
            'id': self.id,
            'position': self.position,
            'width': self.width,
            'status': self.status,
            'agent_count': self.agent_count
        }


class Obstacle:
    """Represents a rectangular obstacle."""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bounds = (x, y, x + width, y + height)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        x, y = point
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle boundary."""
        x, y = point
        dx = max(self.x - x, 0, x - (self.x + self.width))
        dy = max(self.y - y, 0, y - (self.y + self.height))
        return np.sqrt(dx**2 + dy**2)
    
    def get_repulsion_force(self, point: np.ndarray, strength: float, range_param: float) -> np.ndarray:
        """Calculate repulsion force from obstacle."""
        dist = self.distance_to_point(point)
        if dist < 0.01:
            dist = 0.01
        
        if dist > range_param * 3:
            return np.zeros(2)
        
        # Direction away from closest point on obstacle
        x, y = point
        cx = np.clip(x, self.x, self.x + self.width)
        cy = np.clip(y, self.y, self.y + self.height)
        direction = point - np.array([cx, cy])
        
        if np.linalg.norm(direction) < 0.01:
            direction = np.random.randn(2)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Exponential decay
        force_magnitude = strength * np.exp(-dist / range_param)
        return force_magnitude * direction


class Grid:
    """
    Spatial grid for efficient neighbor queries and hazard representation.
    """
    
    def __init__(self, width: float, height: float, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.nx = int(np.ceil(width / resolution))
        self.ny = int(np.ceil(height / resolution))
        
        # Grid for spatial hashing
        self.cells: Dict[Tuple[int, int], List[int]] = {}
        
        # Hazard fields
        self.fire_intensity = np.zeros((self.nx, self.ny), dtype=float)
        self.smoke_density = np.zeros((self.nx, self.ny), dtype=float)
        self.walkable = np.ones((self.nx, self.ny), dtype=bool)
        
        # Density tracking
        self.agent_density = np.zeros((self.nx, self.ny), dtype=float)
        
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        ix = int(position[0] / self.resolution)
        iy = int(position[1] / self.resolution)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1)
    
    def grid_to_world(self, ix: int, iy: int) -> np.ndarray:
        """Convert grid indices to world coordinates (cell center)."""
        x = (ix + 0.5) * self.resolution
        y = (iy + 0.5) * self.resolution
        return np.array([x, y])
    
    def is_valid(self, ix: int, iy: int) -> bool:
        """Check if grid indices are valid."""
        return 0 <= ix < self.nx and 0 <= iy < self.ny
    
    def is_walkable(self, position: np.ndarray) -> bool:
        """Check if position is walkable."""
        ix, iy = self.world_to_grid(position)
        return self.walkable[ix, iy]
    
    def add_obstacle_to_grid(self, obstacle: Obstacle):
        """Mark obstacle cells as non-walkable."""
        x_start = int(obstacle.x / self.resolution)
        y_start = int(obstacle.y / self.resolution)
        x_end = int((obstacle.x + obstacle.width) / self.resolution) + 1
        y_end = int((obstacle.y + obstacle.height) / self.resolution) + 1
        
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(self.nx, x_end)
        y_end = min(self.ny, y_end)
        
        self.walkable[x_start:x_end, y_start:y_end] = False
    
    def update_agent_positions(self, agents: List):
        """Update spatial hash for agents."""
        self.cells.clear()
        for agent in agents:
            if agent.alive and not agent.evacuated:
                ix, iy = self.world_to_grid(agent.position)
                cell_key = (ix, iy)
                if cell_key not in self.cells:
                    self.cells[cell_key] = []
                self.cells[cell_key].append(agent.id)
    
    def get_neighbors(self, position: np.ndarray, radius: float, all_agents: List) -> List:
        """Get agents within radius using spatial hashing."""
        neighbors = []
        ix, iy = self.world_to_grid(position)
        cell_radius = int(np.ceil(radius / self.resolution))
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (ix + dx, iy + dy)
                if cell_key in self.cells:
                    for agent_id in self.cells[cell_key]:
                        agent = all_agents[agent_id]
                        if agent.alive and not agent.evacuated:
                            dist = np.linalg.norm(agent.position - position)
                            if dist < radius and dist > 0:
                                neighbors.append(agent)
        return neighbors


class Environment:
    """
    Main environment class managing spatial representation.
    """
    
    def __init__(self, width: float, height: float, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        self.grid = Grid(width, height, resolution)
        self.obstacles: List[Obstacle] = []
        self.exits: List[Exit] = []
        
        # Boundaries as obstacles
        self._create_boundaries()
        
    def _create_boundaries(self):
        """Create walls around environment."""
        wall_thickness = 0.5
        # Bottom wall
        self.add_obstacle(Obstacle(-wall_thickness, -wall_thickness, 
                                   self.width + 2*wall_thickness, wall_thickness))
        # Top wall
        self.add_obstacle(Obstacle(-wall_thickness, self.height, 
                                   self.width + 2*wall_thickness, wall_thickness))
        # Left wall
        self.add_obstacle(Obstacle(-wall_thickness, 0, 
                                   wall_thickness, self.height))
        # Right wall
        self.add_obstacle(Obstacle(self.width, 0, 
                                   wall_thickness, self.height))
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to environment."""
        self.obstacles.append(obstacle)
        self.grid.add_obstacle_to_grid(obstacle)
    
    def add_exit(self, exit_obj: Exit):
        """Add exit to environment."""
        self.exits.append(exit_obj)
    
    def is_position_valid(self, position: np.ndarray, radius: float = 0.0) -> bool:
        """Check if position is valid (within bounds and not in obstacle)."""
        if not (radius <= position[0] <= self.width - radius and 
                radius <= position[1] <= self.height - radius):
            return False
        
        for obstacle in self.obstacles:
            if obstacle.distance_to_point(position) < radius:
                return False
        
        return self.grid.is_walkable(position)
    
    def get_random_valid_position(self, radius: float = 0.3, max_attempts: int = 100) -> Optional[np.ndarray]:
        """Generate random valid position."""
        for _ in range(max_attempts):
            x = np.random.uniform(radius + 1, self.width - radius - 1)
            y = np.random.uniform(radius + 1, self.height - radius - 1)
            pos = np.array([x, y])
            if self.is_position_valid(pos, radius):
                return pos
        return None
    
    def get_wall_repulsion_force(self, position: np.ndarray, strength: float, range_param: float) -> np.ndarray:
        """Calculate total repulsion force from walls and obstacles."""
        total_force = np.zeros(2)
        for obstacle in self.obstacles:
            force = obstacle.get_repulsion_force(position, strength, range_param)
            total_force += force
        return total_force
    
    def check_exit_reached(self, agent) -> Optional[Exit]:
        """Check if agent has reached any exit."""
        for exit_obj in self.exits:
            if exit_obj.status == 'open' and exit_obj.is_agent_at_exit(agent.position):
                return exit_obj
        return None
    
    def update_exit_counts(self, agents: List):
        """Update agent counts at each exit."""
        for exit_obj in self.exits:
            exit_obj.agent_count = 0
        
        for agent in agents:
            if agent.alive and not agent.evacuated:
                closest_exit = None
                min_dist = float('inf')
                for exit_obj in self.exits:
                    if exit_obj.status == 'open':
                        dist = np.linalg.norm(agent.position - exit_obj.position)
                        if dist < min_dist:
                            min_dist = dist
                            closest_exit = exit_obj
                
                if closest_exit and min_dist < 5.0:
                    closest_exit.agent_count += 1
    
    def get_exits_info(self) -> List[Dict]:
        """Get list of exit information for agent perception."""
        return [exit_obj.to_dict() for exit_obj in self.exits]
