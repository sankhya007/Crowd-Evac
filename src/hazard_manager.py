"""
Hazard Management System
Simulates fire spread, smoke diffusion, and dynamic exit failures
"""

import numpy as np
from typing import List, Tuple, Dict


class FireCell:
    """Represents a fire cell in the grid."""
    
    def __init__(self, position: Tuple[int, int], intensity: float = 1.0):
        self.position = position
        self.intensity = intensity
        self.age = 0.0


class HazardManager:
    """
    Manages all hazards: fire, smoke, and exit failures.
    """
    
    def __init__(self, config: dict, grid, environment):
        self.grid = grid
        self.environment = environment
        self.config = config
        
        # Fire configuration
        self.fire_enabled = config.get('fire', {}).get('enabled', True)
        self.fire_start_time = config.get('fire', {}).get('start_time', 30.0)
        self.fire_spread_rate = config.get('fire', {}).get('spread_rate', 0.05)
        self.fire_damage_rate = config.get('fire', {}).get('damage_rate', 0.1)
        self.fire_growth_rate = config.get('fire', {}).get('growth_rate', 1.5)
        self.ignition_points = config.get('fire', {}).get('ignition_points', [[25.0, 25.0]])
        
        # Smoke configuration
        self.smoke_enabled = config.get('smoke', {}).get('enabled', True)
        self.smoke_diffusion_rate = config.get('smoke', {}).get('diffusion_rate', 0.3)
        self.smoke_visibility_reduction = config.get('smoke', {}).get('visibility_reduction', 0.8)
        self.smoke_damage_rate = config.get('smoke', {}).get('damage_rate', 0.02)
        
        # Exit failure configuration
        self.exit_failures_enabled = config.get('exit_failures', {}).get('enabled', True)
        self.failure_times = config.get('exit_failures', {}).get('failure_times', [])
        self.failure_exits = config.get('exit_failures', {}).get('failure_exits', [])
        
        # Active fire cells
        self.fire_cells: Dict[Tuple[int, int], FireCell] = {}
        self.fire_started = False
        
    def update(self, dt: float, current_time: float):
        """
        Update all hazards.
        
        Args:
            dt: Time step
            current_time: Current simulation time
        """
        # Start fire at designated time
        if self.fire_enabled and not self.fire_started and current_time >= self.fire_start_time:
            self._ignite_fire()
            self.fire_started = True
        
        # Update fire and smoke
        if self.fire_started:
            self._update_fire(dt)
            self._update_smoke(dt)
        
        # Check for exit failures
        self._check_exit_failures(current_time)
    
    def _ignite_fire(self):
        """Start fire at ignition points."""
        for point in self.ignition_points:
            ix, iy = self.grid.world_to_grid(np.array(point))
            if self.grid.is_valid(ix, iy):
                self.fire_cells[(ix, iy)] = FireCell((ix, iy), intensity=1.0)
                self.grid.fire_intensity[ix, iy] = 1.0
    
    def _update_fire(self, dt: float):
        """
        Update fire spread using cellular automata.
        
        Args:
            dt: Time step
        """
        new_fires = []
        
        # Grow existing fires
        for cell_pos, fire_cell in self.fire_cells.items():
            fire_cell.age += dt
            fire_cell.intensity = min(10.0, fire_cell.intensity + self.fire_growth_rate * dt)
            self.grid.fire_intensity[cell_pos[0], cell_pos[1]] = fire_cell.intensity
            
            # Spread to neighbors
            ix, iy = cell_pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = ix + dx, iy + dy
                    if not self.grid.is_valid(nx, ny):
                        continue
                    
                    if (nx, ny) not in self.fire_cells and self.grid.walkable[nx, ny]:
                        # Probabilistic spread
                        if np.random.random() < self.fire_spread_rate * dt * fire_cell.intensity:
                            new_fires.append((nx, ny))
        
        # Add new fire cells
        for pos in new_fires:
            if pos not in self.fire_cells:
                self.fire_cells[pos] = FireCell(pos, intensity=0.5)
                self.grid.fire_intensity[pos[0], pos[1]] = 0.5
    
    def _update_smoke(self, dt: float):
        """
        Update smoke diffusion.
        
        Args:
            dt: Time step
        """
        # Generate smoke from fire
        for cell_pos, fire_cell in self.fire_cells.items():
            ix, iy = cell_pos
            smoke_generation = fire_cell.intensity * 0.5 * dt
            self.grid.smoke_density[ix, iy] = min(1.0, 
                self.grid.smoke_density[ix, iy] + smoke_generation)
        
        # Diffuse smoke
        new_smoke = self.grid.smoke_density.copy()
        
        for ix in range(self.grid.nx):
            for iy in range(self.grid.ny):
                if self.grid.smoke_density[ix, iy] > 0:
                    # Spread to neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            nx, ny = ix + dx, iy + dy
                            if self.grid.is_valid(nx, ny):
                                diffusion = self.grid.smoke_density[ix, iy] * self.smoke_diffusion_rate * dt / 8
                                new_smoke[nx, ny] = min(1.0, new_smoke[nx, ny] + diffusion)
        
        self.grid.smoke_density = new_smoke
    
    def _check_exit_failures(self, current_time: float):
        """
        Check and apply exit failures.
        
        Args:
            current_time: Current simulation time
        """
        if not self.exit_failures_enabled:
            return
        
        for i, failure_time in enumerate(self.failure_times):
            if current_time >= failure_time and i < len(self.failure_exits):
                exit_id = self.failure_exits[i]
                for exit_obj in self.environment.exits:
                    if exit_obj.id == exit_id and exit_obj.status == 'open':
                        exit_obj.status = 'blocked'
    
    def apply_hazard_effects(self, agents: List, dt: float):
        """
        Apply hazard damage to agents.
        
        Args:
            agents: List of all agents
            dt: Time step
        """
        for agent in agents:
            if not agent.alive or agent.evacuated:
                continue
            
            ix, iy = self.grid.world_to_grid(agent.position)
            
            # Fire damage
            if self.grid.fire_intensity[ix, iy] > 0:
                damage = self.grid.fire_intensity[ix, iy] * self.fire_damage_rate * dt
                agent.take_damage(damage)
            
            # Smoke damage and visibility reduction
            if self.grid.smoke_density[ix, iy] > 0:
                smoke_level = self.grid.smoke_density[ix, iy]
                
                # Health damage
                damage = smoke_level * self.smoke_damage_rate * dt
                agent.take_damage(damage)
                
                # Reduce visibility
                agent.visibility_range *= (1.0 - smoke_level * self.smoke_visibility_reduction)
    
    def get_hazard_proximity(self, position: np.ndarray, radius: float = 5.0) -> float:
        """
        Get normalized proximity to nearest hazard.
        
        Args:
            position: Agent position
            radius: Search radius
            
        Returns:
            Proximity value [0-1], where 0 is at hazard, 1 is far away
        """
        ix, iy = self.grid.world_to_grid(position)
        
        min_dist = radius
        cell_radius = int(np.ceil(radius / self.grid.resolution))
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                nx, ny = ix + dx, iy + dy
                if not self.grid.is_valid(nx, ny):
                    continue
                
                if self.grid.fire_intensity[nx, ny] > 0 or self.grid.smoke_density[nx, ny] > 0.5:
                    dist = np.sqrt(dx**2 + dy**2) * self.grid.resolution
                    min_dist = min(min_dist, dist)
        
        return min_dist / radius
    
    def get_fire_positions(self) -> List[np.ndarray]:
        """Get world positions of all fire cells."""
        positions = []
        for cell_pos in self.fire_cells:
            positions.append(self.grid.grid_to_world(*cell_pos))
        return positions
    
    def get_smoke_positions(self) -> List[Tuple[np.ndarray, float]]:
        """Get world positions and densities of smoke cells."""
        positions = []
        for ix in range(self.grid.nx):
            for iy in range(self.grid.ny):
                if self.grid.smoke_density[ix, iy] > 0.1:
                    pos = self.grid.grid_to_world(ix, iy)
                    density = self.grid.smoke_density[ix, iy]
                    positions.append((pos, density))
        return positions
