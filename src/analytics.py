"""
Analytics Collection and Computation
Tracks simulation metrics, computes evacuation KPIs, detects bottlenecks
"""

import numpy as np
from typing import List, Dict, Tuple
import csv
from pathlib import Path


class AnalyticsCollector:
    """
    Collects and computes simulation analytics.
    """
    
    def __init__(self, config: dict):
        self.enabled = config.get('enabled', True)
        self.sampling_rate = config.get('sampling_rate', 0.5)
        self.export_csv = config.get('export_csv', True)
        self.csv_path = config.get('csv_path', 'output/analytics.csv')
        self.compute_heatmaps = config.get('compute_heatmaps', True)
        self.bottleneck_threshold = config.get('bottleneck_threshold', 4.0)
        
        # Time series data
        self.timestamps = []
        self.agent_counts = []
        self.evacuated_counts = []
        self.deceased_counts = []
        self.avg_panic_levels = []
        self.avg_speeds = []
        
        # Evacuation times for individual agents
        self.evacuation_times = []
        self.death_times = []
        
        # Spatial data for heatmaps
        self.density_history = []
        self.panic_history = []
        
        # KPIs
        self.kpis = {}
        
        # Last sampling time
        self.last_sample_time = 0.0
    
    def update(self, agents: List, grid, current_time: float, dt: float):
        """
        Update analytics at each timestep.
        
        Args:
            agents: List of all agents
            grid: Environment grid
            current_time: Current simulation time
            dt: Time step
        """
        if not self.enabled:
            return
        
        # Sample at specified rate
        if current_time - self.last_sample_time >= self.sampling_rate:
            self._sample_metrics(agents, grid, current_time)
            self.last_sample_time = current_time
    
    def _sample_metrics(self, agents: List, grid, current_time: float):
        """Sample current metrics."""
        active_agents = [a for a in agents if a.alive and not a.evacuated]
        evacuated_agents = [a for a in agents if a.evacuated]
        deceased_agents = [a for a in agents if not a.alive]
        
        self.timestamps.append(current_time)
        self.agent_counts.append(len(active_agents))
        self.evacuated_counts.append(len(evacuated_agents))
        self.deceased_counts.append(len(deceased_agents))
        
        # Average panic
        if active_agents:
            avg_panic = np.mean([a.panic_level for a in active_agents])
            avg_speed = np.mean([np.linalg.norm(a.velocity) for a in active_agents])
        else:
            avg_panic = 0.0
            avg_speed = 0.0
        
        self.avg_panic_levels.append(avg_panic)
        self.avg_speeds.append(avg_speed)
        
        # Update density grid
        if self.compute_heatmaps:
            self._update_density_grid(agents, grid)
    
    def _update_density_grid(self, agents: List, grid):
        """Update agent density grid."""
        density = np.zeros((grid.nx, grid.ny), dtype=float)
        panic_map = np.zeros((grid.nx, grid.ny), dtype=float)
        count_map = np.zeros((grid.nx, grid.ny), dtype=int)
        
        for agent in agents:
            if agent.alive and not agent.evacuated:
                ix, iy = grid.world_to_grid(agent.position)
                density[ix, iy] += 1.0
                panic_map[ix, iy] += agent.panic_level
                count_map[ix, iy] += 1
        
        # Normalize panic map
        mask = count_map > 0
        panic_map[mask] = panic_map[mask] / count_map[mask]
        
        # Convert to agents per square meter
        cell_area = grid.resolution ** 2
        density = density / cell_area
        
        grid.agent_density = density
        self.density_history.append(density.copy())
        self.panic_history.append(panic_map.copy())
    
    def record_evacuation(self, agent_id: int, time: float):
        """Record agent evacuation."""
        self.evacuation_times.append(time)
    
    def record_death(self, agent_id: int, time: float):
        """Record agent death."""
        self.death_times.append(time)
    
    def compute_kpis(self, total_agents: int, total_time: float):
        """
        Compute Key Performance Indicators.
        
        Args:
            total_agents: Total number of agents
            total_time: Total simulation time
        """
        self.kpis = {
            'total_agents': total_agents,
            'total_evacuated': len(self.evacuation_times),
            'total_deceased': len(self.death_times),
            'evacuation_rate': len(self.evacuation_times) / total_agents if total_agents > 0 else 0,
            'casualty_rate': len(self.death_times) / total_agents if total_agents > 0 else 0,
            'simulation_time': total_time
        }
        
        # Evacuation time percentiles
        if self.evacuation_times:
            sorted_times = sorted(self.evacuation_times)
            n = len(sorted_times)
            
            self.kpis['T50'] = sorted_times[int(n * 0.5)] if n > 0 else None
            self.kpis['T80'] = sorted_times[int(n * 0.8)] if n > 0 else None
            self.kpis['T90'] = sorted_times[int(n * 0.9)] if n > 0 else None
            self.kpis['T95'] = sorted_times[int(n * 0.95)] if n > 0 else None
            self.kpis['T99'] = sorted_times[int(n * 0.99)] if n > 0 else None
            self.kpis['mean_evacuation_time'] = np.mean(sorted_times)
            self.kpis['max_evacuation_time'] = max(sorted_times)
        else:
            self.kpis['T50'] = None
            self.kpis['T80'] = None
            self.kpis['T90'] = None
            self.kpis['T95'] = None
            self.kpis['T99'] = None
            self.kpis['mean_evacuation_time'] = None
            self.kpis['max_evacuation_time'] = None
        
        # Average panic
        if self.avg_panic_levels:
            self.kpis['mean_panic'] = np.mean(self.avg_panic_levels)
            self.kpis['max_panic'] = max(self.avg_panic_levels)
        
        return self.kpis
    
    def detect_bottlenecks(self, grid) -> List[Dict]:
        """
        Detect bottleneck locations from density history.
        
        Args:
            grid: Environment grid
            
        Returns:
            List of bottleneck dictionaries with position and severity
        """
        if not self.density_history:
            return []
        
        # Average density over time
        avg_density = np.mean(np.array(self.density_history), axis=0)
        
        # Find high-density cells
        bottlenecks = []
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                if avg_density[ix, iy] > self.bottleneck_threshold:
                    position = grid.grid_to_world(ix, iy)
                    bottlenecks.append({
                        'position': position,
                        'density': avg_density[ix, iy],
                        'severity': avg_density[ix, iy] / self.bottleneck_threshold
                    })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        return bottlenecks
    
    def generate_heatmap(self, data_type: str = 'density') -> np.ndarray:
        """
        Generate heatmap from collected data.
        
        Args:
            data_type: Type of heatmap ('density' or 'panic')
            
        Returns:
            2D array representing heatmap
        """
        if data_type == 'density' and self.density_history:
            return np.mean(np.array(self.density_history), axis=0)
        elif data_type == 'panic' and self.panic_history:
            return np.mean(np.array(self.panic_history), axis=0)
        else:
            return None
    
    def export_to_csv(self):
        """Export time series data to CSV."""
        if not self.export_csv or not self.timestamps:
            return
        
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time', 'Active_Agents', 'Evacuated', 'Deceased',
                'Avg_Panic', 'Avg_Speed'
            ])
            
            for i in range(len(self.timestamps)):
                writer.writerow([
                    f"{self.timestamps[i]:.2f}",
                    self.agent_counts[i],
                    self.evacuated_counts[i],
                    self.deceased_counts[i],
                    f"{self.avg_panic_levels[i]:.3f}",
                    f"{self.avg_speeds[i]:.3f}"
                ])
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary of simulation results.
        
        Returns:
            Formatted summary string
        """
        if not self.kpis:
            return "No analytics data available."
        
        report = []
        report.append("=" * 60)
        report.append("EVACUATION SIMULATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("OVERALL STATISTICS:")
        report.append(f"  Total Agents: {self.kpis['total_agents']}")
        report.append(f"  Successfully Evacuated: {self.kpis['total_evacuated']}")
        report.append(f"  Casualties: {self.kpis['total_deceased']}")
        report.append(f"  Evacuation Rate: {self.kpis['evacuation_rate']:.1%}")
        report.append(f"  Casualty Rate: {self.kpis['casualty_rate']:.1%}")
        report.append(f"  Simulation Time: {self.kpis['simulation_time']:.1f}s")
        report.append("")
        
        if self.kpis['T50'] is not None:
            report.append("EVACUATION TIME PERCENTILES:")
            report.append(f"  T50 (50% evacuated): {self.kpis['T50']:.1f}s")
            report.append(f"  T80 (80% evacuated): {self.kpis['T80']:.1f}s")
            report.append(f"  T90 (90% evacuated): {self.kpis['T90']:.1f}s")
            report.append(f"  T95 (95% evacuated): {self.kpis['T95']:.1f}s")
            report.append(f"  Mean Evacuation Time: {self.kpis['mean_evacuation_time']:.1f}s")
            report.append(f"  Max Evacuation Time: {self.kpis['max_evacuation_time']:.1f}s")
            report.append("")
        
        if 'mean_panic' in self.kpis:
            report.append("BEHAVIORAL METRICS:")
            report.append(f"  Mean Panic Level: {self.kpis['mean_panic']:.2f}")
            report.append(f"  Peak Panic Level: {self.kpis['max_panic']:.2f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_exit_statistics(self, exits: List) -> Dict:
        """
        Get statistics for each exit.
        
        Args:
            exits: List of exit objects
            
        Returns:
            Dictionary of exit statistics
        """
        stats = {}
        for exit_obj in exits:
            stats[exit_obj.id] = {
                'position': exit_obj.position.tolist(),
                'total_evacuated': exit_obj.total_evacuated,
                'final_status': exit_obj.status
            }
        return stats
