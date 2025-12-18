"""
Core Agent Class for Crowd Simulation
Each agent represents an individual person with autonomous behavior
"""

import numpy as np
from typing import Tuple, List, Optional


class Agent:
    """
    Represents an autonomous agent (person) in the simulation.
    
    Attributes:
        id: Unique identifier
        position: Current [x, y] position in meters
        velocity: Current [vx, vy] velocity in m/s
        desired_speed: Preferred walking speed (m/s)
        radius: Body radius for collision detection (m)
        visibility_range: Maximum perception distance (m)
        panic_level: Current panic state [0-1]
        goal: Target position [x, y] or exit ID
        alive: Whether agent is still alive
        evacuated: Whether agent has reached exit
        path: Planned path as list of waypoints
    """
    
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        desired_speed: float,
        radius: float,
        visibility_range: float,
        panic_threshold: float = 0.3
    ):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.desired_speed = desired_speed
        self.max_speed = desired_speed * 1.3  # Can exceed in panic
        self.radius = radius
        self.visibility_range = visibility_range
        self.panic_level = 0.0
        self.panic_threshold = panic_threshold
        
        # Navigation
        self.goal = None
        self.target_exit = None
        self.path = []
        self.path_index = 0
        
        # State
        self.alive = True
        self.evacuated = False
        self.health = 1.0  # 0-1 scale
        
        # History for analytics
        self.trajectory = [self.position.copy()]
        self.panic_history = [0.0]
        
        # Perception
        self.perceived_agents = []
        self.perceived_hazards = []
        self.perceived_exits = []
        
    def update_position(self, dt: float):
        """Update position based on current velocity."""
        if not self.alive or self.evacuated:
            return
            
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())
        
    def update_panic(self, hazard_proximity: float, nearby_panic_levels: List[float], dt: float):
        """
        Update panic level based on hazards and nearby agents.
        
        Args:
            hazard_proximity: Distance to nearest hazard (normalized)
            nearby_panic_levels: List of panic levels of nearby agents
            dt: Time step
        """
        if not self.alive:
            return
            
        # Panic increases due to hazards
        if hazard_proximity < 1.0:
            hazard_panic = (1.0 - hazard_proximity) * 0.5 * dt
            self.panic_level = min(1.0, self.panic_level + hazard_panic)
        
        # Panic spreads from nearby panicked agents (emotional contagion)
        if nearby_panic_levels:
            avg_nearby_panic = np.mean(nearby_panic_levels)
            if avg_nearby_panic > self.panic_threshold:
                contagion = (avg_nearby_panic - self.panic_level) * 0.1 * dt
                self.panic_level = min(1.0, self.panic_level + contagion)
        
        # Panic slowly decreases when away from danger
        if hazard_proximity > 1.0 and self.panic_level > 0:
            self.panic_level = max(0.0, self.panic_level - 0.05 * dt)
            
        self.panic_history.append(self.panic_level)
    
    def get_actual_speed(self) -> float:
        """Get current speed accounting for panic."""
        if self.panic_level > self.panic_threshold:
            # Panic increases speed but becomes erratic
            panic_factor = 1.0 + self.panic_level * 0.5
            return min(self.max_speed, self.desired_speed * panic_factor)
        return self.desired_speed
    
    def take_damage(self, damage: float):
        """Apply damage to agent's health."""
        self.health = max(0.0, self.health - damage)
        if self.health <= 0:
            self.alive = False
            self.velocity = np.zeros(2)
    
    def set_goal(self, goal_position: np.ndarray):
        """Set navigation goal."""
        self.goal = np.array(goal_position, dtype=float)
    
    def set_path(self, path: List[np.ndarray]):
        """Set planned path to follow."""
        self.path = [np.array(p, dtype=float) for p in path]
        self.path_index = 0
    
    def get_next_waypoint(self) -> Optional[np.ndarray]:
        """Get next waypoint on path."""
        if self.path and self.path_index < len(self.path):
            waypoint = self.path[self.path_index]
            # Move to next waypoint if current one is reached
            if np.linalg.norm(self.position - waypoint) < 0.5:
                self.path_index += 1
                if self.path_index < len(self.path):
                    return self.path[self.path_index]
            return waypoint
        return self.goal if self.goal is not None else None
    
    def is_at_goal(self, threshold: float = 0.5) -> bool:
        """Check if agent has reached goal."""
        if self.goal is None:
            return False
        return np.linalg.norm(self.position - self.goal) < threshold
    
    def perceive_environment(self, agents: List['Agent'], obstacles: List, exits: List, hazards: List):
        """
        Update agent's perception of environment.
        
        Args:
            agents: List of all agents
            obstacles: List of obstacles
            exits: List of exits
            hazards: List of active hazards
        """
        if not self.alive:
            return
            
        # Visibility reduced by panic and smoke
        effective_visibility = self.visibility_range * (1.0 - self.panic_level * 0.3)
        
        # Perceive nearby agents
        self.perceived_agents = []
        for agent in agents:
            if agent.id != self.id and agent.alive:
                dist = np.linalg.norm(agent.position - self.position)
                if dist < effective_visibility:
                    self.perceived_agents.append(agent)
        
        # Perceive exits
        self.perceived_exits = []
        for exit_info in exits:
            dist = np.linalg.norm(exit_info['position'] - self.position)
            if dist < effective_visibility and exit_info['status'] == 'open':
                self.perceived_exits.append(exit_info)
        
        # Perceive hazards
        self.perceived_hazards = hazards  # Simplified: perceive all hazards
    
    def select_target_exit(self) -> Optional[int]:
        """
        Select best exit based on distance and congestion.
        
        Returns:
            Exit ID or None
        """
        if not self.perceived_exits:
            return None
        
        best_exit = None
        best_score = float('inf')
        
        for exit_info in self.perceived_exits:
            dist = np.linalg.norm(exit_info['position'] - self.position)
            congestion = exit_info.get('agent_count', 0)
            
            # Score: distance + congestion penalty
            score = dist + congestion * 2.0
            
            # Panic reduces rational decision-making
            if self.panic_level > 0.5:
                score += np.random.uniform(-5, 5)
            
            if score < best_score:
                best_score = score
                best_exit = exit_info['id']
        
        return best_exit
    
    def __repr__(self) -> str:
        status = "evacuated" if self.evacuated else ("dead" if not self.alive else "active")
        return f"Agent({self.id}, pos={self.position}, panic={self.panic_level:.2f}, {status})"
