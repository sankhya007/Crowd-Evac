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
        self.visualizer = Visualizer(
            config['visualization'],
            self.environment,
            canvas=config.get("canvas", None)
        )
        
        # Simulation state
        self.current_time = 0.0
        self.running = False
    
    # substracted because the spawing of the agents were spawnig in random spots rather than staying in the rooms 
    
    # def _create_agents(self, agent_config: dict) -> List[Agent]:
    #     """Create agents randomly across the whole map avoiding obstacles."""

    #     agents = []
    #     count = agent_config['count']

    #     for i in range(count):

    #         speed = np.random.uniform(*agent_config['speed_range'])
    #         radius = np.random.uniform(*agent_config['radius_range'])
    #         visibility = np.random.uniform(*agent_config['visibility_range'])

    #         # find valid spawn position
    #         position = None
    #         while position is None:
    #             x = np.random.uniform(0, self.environment.width)
    #             y = np.random.uniform(0, self.environment.height)

    #             candidate = np.array([x, y])

    #             if self.environment.grid.is_walkable(candidate):
    #                 position = candidate

    #         agent = Agent(
    #             agent_id=i,
    #             position=position,
    #             desired_speed=speed,
    #             radius=radius,
    #             visibility_range=visibility,
    #             panic_threshold=agent_config['panic_threshold']
    #         )

    #         agents.append(agent)

    #     print(f"Created {len(agents)} agents distributed across map")
    #     return agents
       
       
    def _create_agents(self, agent_config):

        agents = []
        count = agent_config["count"]

        valid_cells = []

        for ix in range(self.environment.grid.nx):
            for iy in range(self.environment.grid.ny):

                if self.environment.grid.walkable[ix, iy]:

                    pos = self.environment.grid.grid_to_world(ix, iy)

                    # avoid spawning too close to exits
                    too_close = False
                    for exit_obj in self.environment.exits:
                        if np.linalg.norm(pos - exit_obj.position) < 5:
                            too_close = True
                            break

                    if not too_close:
                        valid_cells.append(pos)

        for i in range(count):

            pos = valid_cells[np.random.randint(len(valid_cells))]

            speed = np.random.uniform(*agent_config["speed_range"])
            radius = np.random.uniform(*agent_config["radius_range"])
            visibility = np.random.uniform(*agent_config["visibility_range"])

            agent = Agent(
                agent_id=i,
                position=pos,
                desired_speed=speed,
                radius=radius,
                visibility_range=visibility,
                panic_threshold=agent_config["panic_threshold"]
            )

            agents.append(agent)

        print(f"Created {len(agents)} agents distributed across walkable area")

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
            if frame_count % 3 == 0:
                self.visualizer.render_frame(
                    self.agents,
                    self.hazard_manager,
                    self.current_time,
                    self.analytics,
                    show=True
                )
                # recuced the frequency for performance, matlab redraw is expensive. 
            
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
        
        
        
    def step(self):
        """
        Advance simulation by one timestep (GUI mode)
        """

        if self.current_time >= self.duration:
            return

        # run one physics step
        self._step()

        # render frame for GUI canvas
        if self.visualizer:
            self.visualizer.render_frame(
                self.agents,
                self.hazard_manager,
                self.current_time,
                self.analytics,
                show=True
            )
    
    def _step(self):
        
        # ------------------------------------------------
        # Terminal progress logging every 5 seconds
        # ------------------------------------------------

        if int(self.current_time) % 5 == 0:

            if not hasattr(self, "_last_log_time"):
                self._last_log_time = -1

            if int(self.current_time) != self._last_log_time:

                active = len([a for a in self.agents if a.alive and not a.evacuated])
                evacuated = len([a for a in self.agents if a.evacuated])
                dead = len([a for a in self.agents if not a.alive])

                print(
                    f"[SIM {self.current_time:6.1f}s] "
                    f"Active:{active:4} | "
                    f"Evacuated:{evacuated:4} | "
                    f"Dead:{dead:3}"
                )

                self._last_log_time = int(self.current_time)
                
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
            if False and agent.id == 0 and int(self.current_time * 10) % 50 == 0:
                #dsabled the debug pringing here
                #print(f"  [DEBUG] Agent 0 can see {len(agent.perceived_exits)} exits (total={len(exits_info)}, visibility={agent.visibility_range:.1f}m)")
                if len(agent.perceived_exits) > 0:
                    print(f"         Perceived exits: {[e['id'] for e in agent.perceived_exits]}")
            
            # Select nearest exit if none set
            if agent.goal is None:

                best_exit = None
                best_score = float("inf")

                for exit_obj in self.environment.exits:

                    dist = np.linalg.norm(agent.position - exit_obj.position)

                    congestion = exit_obj.agent_count

                    score = dist + congestion * 3

                    if score < best_score:
                        best_score = score
                        best_exit = exit_obj

                if best_exit is not None:
                    agent.goal = best_exit.position.copy()
                
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
            if False and agent.id == 0 and int(self.current_time * 10) % 10 == 0:
                #debug printing disabled for perfomance boost(CPU was litrally vergeing close to death)
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

        print("\nGenerating outputs...")

        import os
        import pandas as pd
        import numpy as np

        os.makedirs("output", exist_ok=True)
        os.makedirs("output/heatmaps", exist_ok=True)

        # ------------------------------------------------
        # 1️⃣ TIME SERIES ANALYTICS CSV
        # ------------------------------------------------

        self.analytics.export_to_csv()
        print("Saved floorplan_analytics.csv")

        # ------------------------------------------------
        # 2️⃣ SUMMARY ANALYSIS CSV
        # ------------------------------------------------

        total_agents = len(self.agents)
        evacuated = len([a for a in self.agents if a.evacuated])
        dead = len([a for a in self.agents if not a.alive])
        active = total_agents - evacuated - dead

        avg_panic = np.mean([a.panic_level for a in self.agents])

        summary = {
            "total_agents": total_agents,
            "evacuated_agents": evacuated,
            "dead_agents": dead,
            "active_agents": active,
            "average_panic": avg_panic,
            "simulation_time": self.current_time,
            "num_exits": len(self.environment.exits),
            "num_obstacles": len(self.environment.obstacles)
        }

        df_summary = pd.DataFrame([summary])
        df_summary.to_csv("output/analysis.csv", index=False)

        print("Saved analysis.csv")

        # ------------------------------------------------
        # 3️⃣ FLOORPLAN ANALYTICS CSV
        # ------------------------------------------------

        floorplan_data = []

        for exit_obj in self.environment.exits:
            floorplan_data.append({
                "type": "exit",
                "x": exit_obj.position[0],
                "y": exit_obj.position[1],
                "width": exit_obj.width,
                "capacity": exit_obj.capacity
            })

        for obs in self.environment.obstacles:
            floorplan_data.append({
                "type": "obstacle",
                "x": obs.x,
                "y": obs.y,
                "width": obs.width,
                "height": obs.height
            })

        df_floor = pd.DataFrame(floorplan_data)
        df_floor.to_csv("output/floorplan_structure.csv", index=False)

        print("Saved floorplan_structure.csv")

        # ------------------------------------------------
        # 4️⃣ HEATMAPS
        # ------------------------------------------------

        if self.analytics.compute_heatmaps:

            density_heatmap = self.analytics.generate_heatmap("density")

            if density_heatmap is not None:
                self.visualizer.export_heatmap(
                    density_heatmap,
                    "Agent Density Heatmap",
                    "output/heatmaps/density_heatmap.png"
                )

            panic_heatmap = self.analytics.generate_heatmap("panic")

            if panic_heatmap is not None:
                self.visualizer.export_heatmap(
                    panic_heatmap,
                    "Panic Heatmap",
                    "output/heatmaps/panic_heatmap.png"
                )
                
            # Generate congestion animation
        self.analytics.export_congestion_animation(
            self.environment.grid,
            self.environment
        )

        # ------------------------------------------------
        # 5️⃣ AGENT PATH VISUALIZATION
        # ------------------------------------------------

        self.visualizer.export_movement_paths(
            self.agents,
            "output/agent_paths.png",
            floorplan_path=self.floorplan_path
        )

        # ------------------------------------------------
        # 6️⃣ BOTTLENECK DETECTION
        # ------------------------------------------------

        bottlenecks = self.analytics.detect_bottlenecks(self.environment.grid)

        if bottlenecks:

            df_bottleneck = pd.DataFrame(bottlenecks)
            df_bottleneck.to_csv("output/bottlenecks.csv", index=False)

            print("Saved bottlenecks.csv")
            
        # ------------------------------------------------
        # HEATMAP GENERATION
        # ------------------------------------------------

        print("Generating heatmaps...")

        density_heatmap = self.analytics.generate_heatmap("density")

        if density_heatmap is not None:
            self.visualizer.export_heatmap(
                density_heatmap,
                "Agent Density Heatmap",
                "output/heatmaps/density_heatmap.png"
            )

        panic_heatmap = self.analytics.generate_heatmap("panic")

        if panic_heatmap is not None:
            self.visualizer.export_heatmap(
                panic_heatmap,
                "Panic Level Heatmap",
                "output/heatmaps/panic_heatmap.png"
            )

        print("Heatmaps generated")

        # ------------------------------------------------

        self.visualizer.close()

        print("\nAll outputs saved in /output/")

        score = self.analytics.compute_evacuation_score(
            len(self.agents),
            self.current_time
        )

        import pandas as pd

        df = pd.DataFrame([score])
        df.to_csv("output/evacuation_score.csv", index=False)

        print("Evacuation Score:", round(score["score"],2), "/ 100")
