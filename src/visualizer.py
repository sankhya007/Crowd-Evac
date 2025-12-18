"""
Visualization System
Real-time rendering and video export
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional


class Visualizer:
    """
    Handles real-time visualization and video export.
    """
    
    def __init__(self, config: dict, environment):
        self.enabled = config.get('enabled', True)
        self.fps = config.get('fps', 240)
        self.realtime = config.get('realtime', False)
        self.show_trajectories = config.get('show_trajectories', True)
        self.show_panic_levels = config.get('show_panic_levels', True)
        self.show_hazards = config.get('show_hazards', True)
        self.video_export = config.get('video_export', False)  # Disabled by default
        self.video_path = config.get('video_path', 'output/simulation.mp4')
        
        # Trail configuration
        self.trail_length = config.get('trail_length', 50)  # Number of positions to show in trail
        self.show_all_trails = config.get('show_all_trails', True)  # Show trails for all agents
        self.trail_alpha = config.get('trail_alpha', 0.6)  # Base alpha for trails
        
        self.environment = environment
        
        # Setup figure with non-interactive backend to avoid GUI issues
        if self.enabled:
            plt.ion()  # Interactive mode
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.fig.canvas.toolbar_visible = False  # Hide toolbar to avoid tkinter issues
            self.setup_plot()
            
            # Storage for animation frames
            self.frames = []
            
            # Track agent trajectories
            self.agent_trails = {}
    
    def setup_plot(self):
        """Initialize plot settings."""
        self.ax.set_xlim(0, self.environment.width)
        self.ax.set_ylim(0, self.environment.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Crowd Evacuation Simulation')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obstacle in self.environment.obstacles:
            rect = Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                facecolor='gray',
                edgecolor='black',
                alpha=0.7
            )
            self.ax.add_patch(rect)
    
    def render_frame(self, agents: List, hazard_manager, current_time: float, 
                    analytics=None, show: bool = True):
        """
        Render a single frame.
        
        Args:
            agents: List of all agents
            hazard_manager: HazardManager instance
            current_time: Current simulation time
            analytics: Optional AnalyticsCollector
            show: Whether to display the frame
        """
        if not self.enabled:
            return
        
        # Clear previous frame (except obstacles and exits)
        self.ax.clear()
        self.setup_plot()
        
        # Draw exits
        for exit_obj in self.environment.exits:
            color = 'green' if exit_obj.status == 'open' else 'red'
            circle = Circle(
                exit_obj.position,
                exit_obj.width / 2,
                facecolor=color,
                edgecolor='darkgreen' if exit_obj.status == 'open' else 'darkred',
                alpha=0.6,
                linewidth=2
            )
            self.ax.add_patch(circle)
            self.ax.text(
                exit_obj.position[0],
                exit_obj.position[1],
                f'Exit {exit_obj.id}',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold'
            )
        
        # Draw hazards
        if self.show_hazards and hazard_manager:
            # Fire
            fire_positions = hazard_manager.get_fire_positions()
            if fire_positions:
                fire_array = np.array(fire_positions)
                self.ax.scatter(
                    fire_array[:, 0],
                    fire_array[:, 1],
                    c='red',
                    marker='*',
                    s=100,
                    alpha=0.8,
                    label='Fire'
                )
            
            # Smoke (as semi-transparent overlay)
            smoke_positions = hazard_manager.get_smoke_positions()
            for pos, density in smoke_positions:
                if density > 0.3:
                    circle = Circle(
                        pos,
                        self.environment.grid.resolution,
                        facecolor='gray',
                        alpha=density * 0.5
                    )
                    self.ax.add_patch(circle)
        
        # Draw agents
        active_agents = [a for a in agents if a.alive and not a.evacuated]
        if active_agents:
            positions = np.array([a.position for a in active_agents])
            
            if self.show_panic_levels:
                panic_levels = np.array([a.panic_level for a in active_agents])
                colors = cm.YlOrRd(panic_levels)
            else:
                colors = 'blue'
            
            self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=colors,
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            
            # Draw trajectories with fading trails
            if self.show_trajectories:
                agents_to_show = active_agents if self.show_all_trails else active_agents[::5]
                
                for agent in agents_to_show:
                    if len(agent.trajectory) > 1:
                        # Get recent trajectory points
                        traj = np.array(agent.trajectory[-self.trail_length:])
                        
                        if len(traj) > 1:
                            # Draw trail with gradient effect (older = more transparent)
                            num_segments = len(traj) - 1
                            
                            for i in range(num_segments):
                                # Alpha increases from old to recent positions
                                alpha = self.trail_alpha * (i + 1) / num_segments
                                
                                # Color based on panic level
                                if agent.panic_level > 0.6:
                                    color = 'red'
                                elif agent.panic_level > 0.3:
                                    color = 'orange'
                                else:
                                    color = 'cyan'
                                
                                self.ax.plot(
                                    traj[i:i+2, 0],
                                    traj[i:i+2, 1],
                                    color=color,
                                    alpha=alpha,
                                    linewidth=1.5,
                                    solid_capstyle='round'
                                )
        
        # Add statistics text
        stats_text = f"Time: {current_time:.1f}s\n"
        stats_text += f"Active: {len([a for a in agents if a.alive and not a.evacuated])}\n"
        stats_text += f"Evacuated: {len([a for a in agents if a.evacuated])}\n"
        stats_text += f"Deceased: {len([a for a in agents if not a.alive])}"
        
        if analytics and analytics.avg_panic_levels:
            stats_text += f"\nAvg Panic: {analytics.avg_panic_levels[-1]:.2f}"
        
        self.ax.text(
            0.02, 0.98,
            stats_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10,
            fontfamily='monospace'
        )
        
        # Draw movement vectors (arrows showing direction to exits)
        if active_agents:
            for agent in active_agents[::3]:  # Show every 3rd agent's direction
                if agent.velocity is not None and np.linalg.norm(agent.velocity) > 0.1:
                    # Draw arrow showing movement direction
                    self.ax.arrow(
                        agent.position[0],
                        agent.position[1],
                        agent.velocity[0] * 0.5,  # Scale for visibility
                        agent.velocity[1] * 0.5,
                        head_width=0.3,
                        head_length=0.2,
                        fc='blue' if agent.panic_level < 0.5 else 'red',
                        ec='blue' if agent.panic_level < 0.5 else 'red',
                        alpha=0.4,
                        linewidth=1
                    )
        
        # Legend (simplified - no colorbar or unicode to avoid tkinter issues)
        if self.show_panic_levels:
            # Create manual legend instead of colorbar
            legend_text = "Panic Level:\nLow (Green)\nMedium (Yellow)\nHigh (Red)"
            self.ax.text(
                0.98, 0.85,
                legend_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=9
            )
        
        if show:
            plt.draw()
            plt.pause(0.001)
    
    def save_frame(self):
        """Save current frame for video export."""
        if self.video_export:
            self.frames.append(self.fig)
    
    def export_video(self):
        """Export simulation as video."""
        if not self.video_export or not self.frames:
            return
        
        Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting video to {self.video_path}...")
        # Video export would require additional setup with FFmpeg
        # For MVP, we'll just save the figure
        print("Video export requires FFmpeg - skipping for MVP")
    
    def export_movement_paths(self, agents: List, filename: str = 'output/agent_paths.png', floorplan_path: str = None):
        """
        Export a comprehensive visualization showing all agent movement paths to exits.
        Optionally overlays on original floorplan image.
        
        Args:
            agents: List of all agents
            filename: Output filename
            floorplan_path: Optional path to original floorplan image to use as background
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Set up plot
        ax.set_xlim(0, self.environment.width)
        ax.set_ylim(0, self.environment.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=13, fontweight='bold')
        ax.set_title('Agent Movement Paths on Floorplan - Evacuation Simulation', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # If floorplan image provided, use it as background
        if floorplan_path and Path(floorplan_path).exists():
            try:
                from PIL import Image
                img = Image.open(floorplan_path)
                # Calculate scale to match simulation coordinates
                img_array = np.array(img)
                extent = [0, self.environment.width, 0, self.environment.height]
                ax.imshow(img_array, extent=extent, origin='lower', alpha=0.3, aspect='auto')
            except Exception as e:
                print(f"  Warning: Could not load floorplan image: {e}")
        
        # Draw obstacles (dark gray)
        for obstacle in self.environment.obstacles:
            rect = Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                facecolor='#333333',
                edgecolor='black',
                alpha=0.8,
                linewidth=1
            )
            ax.add_patch(rect)
        
        # Draw exits (bright green with labels)
        for exit_obj in self.environment.exits:
            circle = Circle(
                exit_obj.position,
                exit_obj.width,
                facecolor='#00FF00',
                edgecolor='darkgreen',
                alpha=0.8,
                linewidth=2,
                label='Exit' if exit_obj.id == 0 else None
            )
            ax.add_patch(circle)
            ax.text(
                exit_obj.position[0],
                exit_obj.position[1],
                f'EXIT\n{exit_obj.id}',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='darkgreen'
            )
        
        # Draw all agent trajectories with color gradient and arrows
        evacuated_count = 0
        deceased_count = 0
        
        for agent in agents:
            if len(agent.trajectory) > 2:
                traj = np.array(agent.trajectory)
                
                # Color based on outcome
                if agent.evacuated:
                    color = '#0066FF'  # Bright blue
                    alpha = 0.7
                    linewidth = 2.5
                    evacuated_count += 1
                elif not agent.alive:
                    color = '#FF0000'  # Red
                    alpha = 0.8
                    linewidth = 3
                    deceased_count += 1
                else:
                    color = '#FFA500'  # Orange
                    alpha = 0.6
                    linewidth = 2
                
                # Draw trajectory line
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    solid_capstyle='round'
                )
                
                # Add arrows showing direction of movement
                if len(traj) > 5:
                    # Draw arrows at intervals along the path
                    arrow_indices = np.linspace(0, len(traj)-2, min(5, len(traj)//3), dtype=int)
                    for idx in arrow_indices:
                        if idx < len(traj) - 1:
                            dx = traj[idx+1, 0] - traj[idx, 0]
                            dy = traj[idx+1, 1] - traj[idx, 1]
                            if np.sqrt(dx**2 + dy**2) > 0.1:  # Only if movement is significant
                                ax.arrow(
                                    traj[idx, 0],
                                    traj[idx, 1],
                                    dx * 0.5,
                                    dy * 0.5,
                                    head_width=0.5,
                                    head_length=0.3,
                                    fc=color,
                                    ec=color,
                                    alpha=alpha * 0.8,
                                    linewidth=0.5
                                )
                
                # Draw start point
                ax.plot(
                    traj[0, 0],
                    traj[0, 1],
                    'o',
                    color='green',
                    markersize=4,
                    alpha=0.6
                )
                
                # Draw end point
                if agent.evacuated:
                    ax.plot(
                        traj[-1, 0],
                        traj[-1, 1],
                        's',
                        color='blue',
                        markersize=5,
                        alpha=0.8
                    )
                elif not agent.alive:
                    ax.plot(
                        traj[-1, 0],
                        traj[-1, 1],
                        'x',
                        color='red',
                        markersize=6,
                        alpha=0.8
                    )
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, alpha=0.6, label=f'Evacuated ({evacuated_count})'),
            Line2D([0], [0], color='orange', lw=2, alpha=0.6, label='Still Active'),
            Line2D([0], [0], color='red', lw=2, alpha=0.6, label=f'Deceased ({deceased_count})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=6, label='Start Position'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                   markersize=6, label='Evacuated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', 
                   markersize=10, label='Exit')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add statistics
        total_agents = len(agents)
        stats_text = f"Total Agents: {total_agents}\n"
        stats_text += f"Evacuated: {evacuated_count} ({100*evacuated_count/total_agents:.1f}%)\n"
        stats_text += f"Deceased: {deceased_count} ({100*deceased_count/total_agents:.1f}%)"
        
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=11,
            fontfamily='monospace',
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved agent movement paths to: {filename}")
        plt.close(fig)
    
    def export_heatmap(self, heatmap_data: np.ndarray, title: str, filename: str):
        """
        Export heatmap as image.
        
        Args:
            heatmap_data: 2D array of heatmap values
            title: Plot title
            filename: Output filename
        """
        if heatmap_data is None:
            return
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        extent = [0, self.environment.width, 0, self.environment.height]
        im = ax.imshow(
            heatmap_data.T,
            origin='lower',
            extent=extent,
            cmap='hot',
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Draw obstacles
        for obstacle in self.environment.obstacles:
            rect = Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                facecolor='gray',
                edgecolor='black',
                alpha=0.5
            )
            ax.add_patch(rect)
        
        # Draw exits
        for exit_obj in self.environment.exits:
            circle = Circle(
                exit_obj.position,
                exit_obj.width / 2,
                facecolor='lime',
                edgecolor='darkgreen',
                alpha=0.8,
                linewidth=2
            )
            ax.add_patch(circle)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(title)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density (agents/m²)' if 'Density' in title else 'Value')
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Heatmap saved to {filename}")
    
    def close(self):
        """Close visualization."""
        if self.enabled:
            plt.close(self.fig)
