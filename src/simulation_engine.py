"""
Simulation Engine
Main time-stepping loop and coordination
"""

import numpy as np
from typing import List
import time

from .agent import Agent
from .environment import Environment, Exit, Obstacle
from .motion_models import MotionController
from .hazard_manager import HazardManager
from .analytics import AnalyticsCollector
from .visualizer import Visualizer


class SimulationEngine:
    """
    Main simulation engine that coordinates all components.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.dt = config['simulation']['time_step']
        self.duration = config['simulation']['duration']
        self.floorplan_path = config.get('floorplan_path', None)  # Store floorplan path if provided
        np.random.seed(config['simulation'].get('seed', 42))
        
        # Initialize environment
        env_config = config['environment']
        self.environment = Environment(
            width=env_config['width'],
            height=env_config['height'],
            resolution=env_config['grid_resolution']
        )
        
        # Add obstacles from floorplan OR from config
        if 'floorplan_obstacles' in env_config and env_config['floorplan_obstacles']:
            print(f"Loading {len(env_config['floorplan_obstacles'])} obstacles from floorplan...")
            for obstacle in env_config['floorplan_obstacles']:
                self.environment.add_obstacle(obstacle)
        elif 'obstacles' in config and config['obstacles']['rectangles']:
            for obs_data in config['obstacles']['rectangles']:
                obstacle = Obstacle(*obs_data)
                self.environment.add_obstacle(obstacle)
        
        # Add exits from floorplan OR from config  
        if 'floorplan_exits' in env_config and env_config['floorplan_exits']:
            print(f"Loading {len(env_config['floorplan_exits'])} exits from floorplan...")
            for exit_obj in env_config['floorplan_exits']:
                self.environment.add_exit(exit_obj)
        elif 'exits' in config:
            for i, (pos, width) in enumerate(zip(config['exits']['positions'], 
                                                  config['exits']['widths'])):
                exit_obj = Exit(
                    exit_id=i,
                    position=np.array(pos),
                    width=width,
                    capacity=config['exits']['capacities'][i]
                )
                self.environment.add_exit(exit_obj)
        
        # Initialize agents
        self.agents = self._create_agents(config['agents'])
        
        # Initialize motion controller
        self.motion_controller = MotionController(
            config['motion'],
            self.environment.grid
        )
        
        # Initialize hazard manager
        self.hazard_manager = HazardManager(
            config['hazards'],
            self.environment.grid,
            self.environment
        )
        
        # Initialize analytics
        self.analytics = AnalyticsCollector(config['analytics'])
        
        # Initialize visualizer
        self.visualizer = Visualizer(config['visualization'], self.environment)
        
        # Simulation state
        self.current_time = 0.0
        self.running = False
    
    def _create_agents(self, agent_config: dict) -> List[Agent]:
        """Create initial agent population distributed across entire map."""
        agents = []
        count = agent_config['count']
        
        # Calculate grid for even distribution
        grid_size = int(np.ceil(np.sqrt(count)))
        cell_width = self.environment.width / grid_size
        cell_height = self.environment.height / grid_size
        
        agents_created = 0
        max_attempts = count * 10  # Prevent infinite loops
        attempts = 0
        
        for i in range(count):
            # Try grid-based placement first for even distribution
            grid_x = (i % grid_size) * cell_width + cell_width / 2
            grid_y = (i // grid_size) * cell_height + cell_height / 2
            
            # Add randomization within cell
            grid_x += np.random.uniform(-cell_width * 0.3, cell_width * 0.3)
            grid_y += np.random.uniform(-cell_height * 0.3, cell_height * 0.3)
            
            # Random attributes
            speed = np.random.uniform(*agent_config['speed_range'])
            radius = np.random.uniform(*agent_config['radius_range'])
            visibility = np.random.uniform(*agent_config['visibility_range'])
            
            # Try grid position first
            position = np.array([grid_x, grid_y])
            
            # Check if position is valid (not in obstacle)
            if self.environment.grid.is_walkable(position):
                agent = Agent(
                    agent_id=agents_created,
                    position=position,
                    desired_speed=speed,
                    radius=radius,
                    visibility_range=visibility,
                    panic_threshold=agent_config['panic_threshold']
                )
                agents.append(agent)
                agents_created += 1
            else:
                # Fallback: try random valid position
                attempts += 1
                if attempts < max_attempts:
                    position = self.environment.get_random_valid_position(radius)
                    if position is not None:
                        agent = Agent(
                            agent_id=agents_created,
                            position=position,
                            desired_speed=speed,
                            radius=radius,
                            visibility_range=visibility,
                            panic_threshold=agent_config['panic_threshold']
                        )
                        agents.append(agent)
                        agents_created += 1
        
        print(f"Created {len(agents)} agents distributed across map")
        return agents
    
    def run(self):
        """Run the main simulation loop."""
        print("=" * 60)
        print("Starting Crowd Evacuation Simulation")
        print("=" * 60)
        print(f"Duration: {self.duration}s, Time step: {self.dt}s")
        print(f"Agents: {len(self.agents)}")
        print(f"Environment: {self.environment.width}m × {self.environment.height}m")
        print(f"Obstacles: {len(self.environment.obstacles)} (including boundaries)")
        print(f"Exits: {len(self.environment.exits)}")
        print("=" * 60)
        
        self.running = True
        start_time = time.time()
        frame_count = 0
        
        while self.current_time < self.duration and self.running:
            self._step()
            
            # Progress indicator
            if int(self.current_time) % 5 == 0 and self.current_time > 0 and frame_count % int(5.0/self.dt) == 0:
                active = len([a for a in self.agents if a.alive and not a.evacuated])
                evacuated = len([a for a in self.agents if a.evacuated])
                print(f"Time: {self.current_time:.1f}s | Active: {active} | Evacuated: {evacuated}")
            
            # Visualization
            if frame_count % max(1, int(1.0 / (self.dt * self.visualizer.fps))) == 0:
                self.visualizer.render_frame(
                    self.agents,
                    self.hazard_manager,
                    self.current_time,
                    self.analytics,
                    show=True
                )
            
            frame_count += 1
            
            # Check if simulation should end early
            active_agents = [a for a in self.agents if a.alive and not a.evacuated]
            if not active_agents:
                print("\nAll agents evacuated or deceased. Ending simulation.")
                break
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        print(f"Simulated time: {self.current_time:.1f}s")
        print(f"Real time: {elapsed_time:.1f}s")
        print(f"Speed: {self.current_time / elapsed_time:.1f}x realtime")
        
        self._finalize()
    
    def _step(self):
        """Execute a single simulation step."""
        # Update spatial grid
        self.environment.grid.update_agent_positions(self.agents)
        
        # Update environment exit counts
        self.environment.update_exit_counts(self.agents)
        
        # Update hazards
        self.hazard_manager.update(self.dt, self.current_time)
        
        # Update each agent
        for agent in self.agents:
            if not agent.alive or agent.evacuated:
                continue
            
            # Perception
            exits_info = self.environment.get_exits_info()
            agent.perceive_environment(
                self.agents,
                self.environment.obstacles,
                exits_info,
                []
            )
            
            # Debug perception for first agent
            if agent.id == 0 and int(self.current_time * 10) % 50 == 0:
                print(f"  [DEBUG] Agent 0 can see {len(agent.perceived_exits)} exits (total={len(exits_info)}, visibility={agent.visibility_range:.1f}m)")
                if len(agent.perceived_exits) > 0:
                    print(f"         Perceived exits: {[e['id'] for e in agent.perceived_exits]}")
            
            # Select target exit if none set
            if agent.goal is None:
                target_exit_id = agent.select_target_exit()
                if target_exit_id is not None:
                    for exit_info in agent.perceived_exits:
                        if exit_info['id'] == target_exit_id:
                            agent.goal = exit_info['position'].copy()
                            if agent.id == 0:
                                print(f"  [DEBUG] Agent 0 selected exit {target_exit_id} at {agent.goal}")
                            break
                elif agent.id == 0 and int(self.current_time * 10) % 50 == 0:
                    print(f"  [DEBUG] Agent 0 cannot select exit (no perceived exits)")
            
            # Get nearby agents for motion computation
            neighbors = self.environment.grid.get_neighbors(
                agent.position,
                radius=5.0,
                all_agents=self.agents
            )
            
            # Update panic level
            nearby_panic = [n.panic_level for n in neighbors if n.panic_level > 0]
            hazard_proximity = self.hazard_manager.get_hazard_proximity(agent.position)
            agent.update_panic(hazard_proximity, nearby_panic, self.dt)
            
            # Update velocity using motion model
            self.motion_controller.update_agent_velocity(
                agent,
                neighbors,
                self.environment,
                self.current_time,
                self.dt
            )
            
            # Debug: Print first agent's info occasionally
            if agent.id == 0 and int(self.current_time * 10) % 10 == 0:
                speed = np.linalg.norm(agent.velocity)
                goal_str = f"({agent.goal[0]:.1f},{agent.goal[1]:.1f})" if agent.goal is not None else "None"
                print(f"  Agent 0: pos=({agent.position[0]:.1f},{agent.position[1]:.1f}), vel={speed:.2f}m/s, goal={goal_str}")
            
            # Update position
            agent.update_position(self.dt)
            
            # Check boundaries
            agent.position[0] = np.clip(agent.position[0], 0, self.environment.width)
            agent.position[1] = np.clip(agent.position[1], 0, self.environment.height)
            
            # Check if reached exit
            exit_reached = self.environment.check_exit_reached(agent)
            if exit_reached:
                agent.evacuated = True
                agent.velocity = np.zeros(2)
                exit_reached.total_evacuated += 1
                self.analytics.record_evacuation(agent.id, self.current_time)
        
        # Apply hazard effects
        self.hazard_manager.apply_hazard_effects(self.agents, self.dt)
        
        # Record deaths
        for agent in self.agents:
            if not agent.alive and agent.health <= 0:
                if agent.id not in [a for a in self.agents if hasattr(a, '_death_recorded')]:
                    self.analytics.record_death(agent.id, self.current_time)
                    agent._death_recorded = True
        
        # Update analytics
        self.analytics.update(self.agents, self.environment.grid, self.current_time, self.dt)
        
        # Advance time
        self.current_time += self.dt
    
    def _finalize(self):
        """Finalize simulation and generate outputs."""
        print("\nGenerating analytics...")
        
        # Compute KPIs
        kpis = self.analytics.compute_kpis(len(self.agents), self.current_time)
        
        # Print summary
        print(self.analytics.generate_summary_report())
        
        # Detect bottlenecks
        bottlenecks = self.analytics.detect_bottlenecks(self.environment.grid)
        if bottlenecks:
            print(f"\nDetected {len(bottlenecks)} bottleneck locations:")
            for i, bn in enumerate(bottlenecks[:5]):
                print(f"  {i+1}. Position ({bn['position'][0]:.1f}, {bn['position'][1]:.1f}), "
                      f"Density: {bn['density']:.2f} agents/m²")
        
        # Export CSV
        self.analytics.export_to_csv()
        print(f"\nTime series data exported to {self.analytics.csv_path}")
        
        # Generate heatmaps
        if self.analytics.compute_heatmaps:
            density_heatmap = self.analytics.generate_heatmap('density')
            if density_heatmap is not None:
                self.visualizer.export_heatmap(
                    density_heatmap,
                    'Agent Density Heatmap',
                    'output/heatmaps/density_heatmap.png'
                )
            
            panic_heatmap = self.analytics.generate_heatmap('panic')
            if panic_heatmap is not None:
                self.visualizer.export_heatmap(
                    panic_heatmap,
                    'Panic Level Heatmap',
                    'output/heatmaps/panic_heatmap.png'
                )
        
        # Export agent movement paths visualization
        print("\nGenerating agent movement paths visualization...")
        self.visualizer.export_movement_paths(
            self.agents,
            'output/agent_paths.png',
            floorplan_path=self.floorplan_path
        )
        
        # Close visualizer
        self.visualizer.close()
        
        print("\n" + "=" * 60)
        print("All outputs generated successfully!")
        print("Check 'output/agent_paths.png' to see how agents moved to exits!")
        print("=" * 60)
