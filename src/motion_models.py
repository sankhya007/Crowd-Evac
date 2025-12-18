"""
Motion Models: Social Force Model, RVO, and A* Pathfinding
Implements three motion strategies for agent navigation
"""

import numpy as np
from typing import List, Tuple, Optional, Set
import heapq
from collections import defaultdict


class SocialForceModel:
    """
    Social Force Model (Helbing et al. 1995)
    Simulates pedestrian dynamics using attractive and repulsive forces.
    """
    
    def __init__(self, config: dict):
        self.tau = config.get('relaxation_time', 0.5)
        self.agent_strength = config.get('agent_strength', 2000.0)
        self.agent_range = config.get('agent_range', 0.08)
        self.wall_strength = config.get('wall_strength', 2000.0)
        self.wall_range = config.get('wall_range', 0.08)
        self.noise_factor = config.get('noise_factor', 0.1)
    
    def compute_velocity(self, agent, neighbors: List, environment, dt: float) -> np.ndarray:
        """
        Compute desired velocity using social force model.
        
        Args:
            agent: The agent to compute velocity for
            neighbors: List of nearby agents
            environment: Environment object
            dt: Time step
            
        Returns:
            New velocity vector
        """
        if not agent.alive or agent.evacuated:
            return np.zeros(2)
        
        # Get target waypoint
        target = agent.get_next_waypoint()
        if target is None:
            return agent.velocity
        
        # 1. Driving force toward goal
        desired_direction = target - agent.position
        dist_to_target = np.linalg.norm(desired_direction)
        
        if dist_to_target < 0.1:
            desired_direction = np.zeros(2)
        else:
            desired_direction = desired_direction / dist_to_target
        
        desired_speed = agent.get_actual_speed()
        desired_velocity = desired_speed * desired_direction
        driving_force = (desired_velocity - agent.velocity) / self.tau
        
        # 2. Repulsive forces from other agents
        agent_repulsion = np.zeros(2)
        for neighbor in neighbors:
            if not neighbor.alive or neighbor.evacuated:
                continue
                
            diff = agent.position - neighbor.position
            dist = np.linalg.norm(diff)
            
            if dist < 0.01:
                dist = 0.01
                diff = np.random.randn(2)
            
            direction = diff / dist
            
            # Exponential repulsion
            combined_radius = agent.radius + neighbor.radius
            force_magnitude = self.agent_strength * np.exp(-(dist - combined_radius) / self.agent_range)
            
            # Stronger force when panic
            if agent.panic_level > 0.5:
                force_magnitude *= (1.0 + agent.panic_level)
            
            agent_repulsion += force_magnitude * direction
        
        # 3. Repulsive forces from walls
        wall_repulsion = environment.get_wall_repulsion_force(
            agent.position, self.wall_strength, self.wall_range
        )
        
        # 4. Random noise for natural variation
        noise = np.random.randn(2) * self.noise_factor * desired_speed
        
        # Total force
        total_force = driving_force + agent_repulsion + wall_repulsion + noise
        
        # Update velocity
        new_velocity = agent.velocity + total_force * dt
        
        # Speed limit
        speed = np.linalg.norm(new_velocity)
        max_speed = agent.max_speed if agent.panic_level > 0.5 else desired_speed * 1.2
        if speed > max_speed:
            new_velocity = new_velocity * (max_speed / speed)
        
        return new_velocity


class RVO:
    """
    Reciprocal Velocity Obstacles (van den Berg et al. 2008)
    Geometric collision avoidance using velocity space.
    """
    
    def __init__(self, config: dict):
        self.time_horizon = config.get('time_horizon', 2.0)
        self.neighbor_dist = config.get('neighbor_dist', 5.0)
        self.max_neighbors = config.get('max_neighbors', 10)
    
    def compute_velocity(self, agent, neighbors: List, environment, dt: float) -> np.ndarray:
        """
        Compute collision-free velocity using RVO.
        
        Args:
            agent: The agent to compute velocity for
            neighbors: List of nearby agents
            environment: Environment object
            dt: Time step
            
        Returns:
            New velocity vector
        """
        if not agent.alive or agent.evacuated:
            return np.zeros(2)
        
        # Get preferred velocity (toward goal)
        target = agent.get_next_waypoint()
        if target is None:
            return agent.velocity
        
        direction = target - agent.position
        dist = np.linalg.norm(direction)
        
        if dist < 0.1:
            return np.zeros(2)
        
        direction = direction / dist
        pref_velocity = direction * agent.get_actual_speed()
        
        # Simple RVO: adjust velocity to avoid collisions
        avoidance = np.zeros(2)
        
        for neighbor in neighbors[:self.max_neighbors]:
            if not neighbor.alive or neighbor.evacuated:
                continue
            
            relative_pos = neighbor.position - agent.position
            relative_vel = neighbor.velocity - agent.velocity
            dist = np.linalg.norm(relative_pos)
            
            if dist < 0.01:
                continue
            
            # Time to collision
            if np.dot(relative_pos, relative_vel) < 0:  # Approaching
                combined_radius = agent.radius + neighbor.radius
                
                if dist < combined_radius * 3:
                    # Avoid by moving perpendicular
                    perp = np.array([-relative_pos[1], relative_pos[0]])
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm > 0:
                        perp = perp / perp_norm
                        avoidance += perp * (1.0 - dist / (combined_radius * 3))
        
        new_velocity = pref_velocity + avoidance * 2.0
        
        # Speed limit
        speed = np.linalg.norm(new_velocity)
        max_speed = agent.get_actual_speed() * 1.2
        if speed > max_speed:
            new_velocity = new_velocity * (max_speed / speed)
        
        return new_velocity


class AStarPathfinder:
    """
    A* Pathfinding on grid with dynamic weights.
    """
    
    def __init__(self, grid):
        self.grid = grid
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbor cells (8-connected)."""
        x, y = node
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if self.grid.is_valid(nx, ny) and self.grid.walkable[nx, ny]:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def get_edge_cost(self, from_node: Tuple[int, int], to_node: Tuple[int, int], 
                      congestion_weight: float, hazard_weight: float) -> float:
        """
        Calculate edge cost considering distance, congestion, and hazards.
        
        Args:
            from_node: Start cell
            to_node: End cell
            congestion_weight: Weight for agent density
            hazard_weight: Weight for hazards
            
        Returns:
            Edge cost
        """
        # Base distance cost
        dx = abs(to_node[0] - from_node[0])
        dy = abs(to_node[1] - from_node[1])
        distance = np.sqrt(dx**2 + dy**2)
        
        # Congestion penalty
        congestion = self.grid.agent_density[to_node[0], to_node[1]]
        congestion_cost = congestion * congestion_weight
        
        # Hazard penalty
        fire = self.grid.fire_intensity[to_node[0], to_node[1]]
        smoke = self.grid.smoke_density[to_node[0], to_node[1]]
        hazard_cost = (fire * 10 + smoke * 5) * hazard_weight
        
        return distance + congestion_cost + hazard_cost
    
    def find_path(self, start: np.ndarray, goal: np.ndarray, 
                  congestion_weight: float = 0.3, hazard_weight: float = 0.5) -> List[np.ndarray]:
        """
        Find path from start to goal using A*.
        
        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            congestion_weight: Weight for congestion
            hazard_weight: Weight for hazards
            
        Returns:
            List of waypoints in world coordinates
        """
        start_cell = self.grid.world_to_grid(start)
        goal_cell = self.grid.world_to_grid(goal)
        
        if not self.grid.is_valid(*start_cell) or not self.grid.is_valid(*goal_cell):
            return [goal]
        
        if not self.grid.walkable[goal_cell[0], goal_cell[1]]:
            # Find nearest walkable cell to goal
            goal_cell = self._find_nearest_walkable(goal_cell)
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_cell] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start_cell] = self.heuristic(start_cell, goal_cell)
        
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal_cell:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid.grid_to_world(*current))
                    current = came_from[current]
                path.reverse()
                path.append(goal)
                return self._simplify_path(path)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + self.get_edge_cost(
                    current, neighbor, congestion_weight, hazard_weight
                )
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return direct line to goal
        return [goal]
    
    def _find_nearest_walkable(self, cell: Tuple[int, int], max_radius: int = 10) -> Tuple[int, int]:
        """Find nearest walkable cell."""
        x, y = cell
        for r in range(1, max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = x + dx, y + dy
                        if self.grid.is_valid(nx, ny) and self.grid.walkable[nx, ny]:
                            return (nx, ny)
        return cell
    
    def _simplify_path(self, path: List[np.ndarray], tolerance: float = 2.0) -> List[np.ndarray]:
        """Simplify path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._is_line_walkable(path[i], path[j]):
                    simplified.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                i += 1
                if i < len(path):
                    simplified.append(path[i])
        
        return simplified
    
    def _is_line_walkable(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if straight line between points is walkable."""
        steps = int(np.linalg.norm(end - start) / self.grid.resolution) + 1
        for i in range(steps):
            t = i / max(steps - 1, 1)
            pos = start + t * (end - start)
            if not self.grid.is_walkable(pos):
                return False
        return True


class MotionController:
    """
    Unified motion controller that combines different motion models.
    """
    
    def __init__(self, config: dict, grid):
        self.mode = config.get('model', 'hybrid')
        
        self.sfm = SocialForceModel(config.get('sfm', {}))
        self.rvo = RVO(config.get('rvo', {}))
        self.pathfinder = AStarPathfinder(grid)
        
        self.replan_interval = config.get('pathfinding', {}).get('replan_interval', 1.0)
        self.congestion_weight = config.get('pathfinding', {}).get('congestion_weight', 0.3)
        self.hazard_weight = config.get('pathfinding', {}).get('hazard_weight', 0.5)
        
        self.last_replan_time = {}
    
    def update_agent_velocity(self, agent, neighbors: List, environment, 
                             current_time: float, dt: float):
        """
        Update agent velocity based on selected motion model.
        
        Args:
            agent: Agent to update
            neighbors: Nearby agents
            environment: Environment object
            current_time: Current simulation time
            dt: Time step
        """
        if not agent.alive or agent.evacuated:
            return
        
        # Replan path periodically
        if (agent.id not in self.last_replan_time or 
            current_time - self.last_replan_time[agent.id] >= self.replan_interval):
            self._replan_path(agent, environment)
            self.last_replan_time[agent.id] = current_time
        
        # Compute velocity based on mode
        if self.mode == 'sfm':
            agent.velocity = self.sfm.compute_velocity(agent, neighbors, environment, dt)
        elif self.mode == 'rvo':
            agent.velocity = self.rvo.compute_velocity(agent, neighbors, environment, dt)
        elif self.mode == 'pathfinding':
            agent.velocity = self.sfm.compute_velocity(agent, neighbors, environment, dt)
        else:  # hybrid
            # Use pathfinding for high-level navigation, SFM for local collision avoidance
            agent.velocity = self.sfm.compute_velocity(agent, neighbors, environment, dt)
    
    def _replan_path(self, agent, environment):
        """Replan path for agent."""
        if agent.goal is None:
            # Select target exit
            exit_id = agent.select_target_exit()
            if exit_id is not None:
                for exit_obj in environment.exits:
                    if exit_obj.id == exit_id:
                        agent.set_goal(exit_obj.position)
                        agent.target_exit = exit_id
                        break
        
        if agent.goal is not None:
            path = self.pathfinder.find_path(
                agent.position, agent.goal,
                self.congestion_weight, self.hazard_weight
            )
            agent.set_path(path)
