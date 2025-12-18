"""
Floorplan Parsers
DXF and raster image parsing for environment generation
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .environment import Obstacle, Exit


class MapMeta:
    """Coordinate transformation manager."""
    
    def __init__(self, scale: float = 1.0, origin: Tuple[float, float] = (0, 0), 
                 rotation: float = 0.0):
        self.scale = scale  # pixels or DXF units per meter
        self.origin = np.array(origin, dtype=float)
        self.rotation = rotation  # radians
    
    def to_world(self, point: np.ndarray) -> np.ndarray:
        """Convert from file coordinates to world coordinates."""
        # Translate
        p = point - self.origin
        # Rotate
        if self.rotation != 0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            p = rot_matrix @ p
        # Scale
        return p / self.scale
    
    def from_world(self, point: np.ndarray) -> np.ndarray:
        """Convert from world coordinates to file coordinates."""
        # Scale
        p = point * self.scale
        # Rotate
        if self.rotation != 0:
            cos_r = np.cos(-self.rotation)
            sin_r = np.sin(-self.rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            p = rot_matrix @ p
        # Translate
        return p + self.origin


class DXFParser:
    """Parse AutoCAD DXF files for floorplan geometry."""
    
    def __init__(self, file_path: str, scale: float = 100.0):
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf library not available. Install with: pip install ezdxf")
        
        self.file_path = Path(file_path)
        self.scale = scale
        self.meta = MapMeta(scale=scale)
        
        self.walls = []
        self.exits = []
        self.obstacles = []
    
    def parse(self) -> Tuple[List[Obstacle], List[Exit]]:
        """
        Parse DXF file and extract geometry.
        
        Returns:
            Tuple of (obstacles, exits)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"DXF file not found: {self.file_path}")
        
        doc = ezdxf.readfile(str(self.file_path))
        msp = doc.modelspace()
        
        # Extract lines (walls)
        for entity in msp.query('LINE'):
            start = np.array([entity.dxf.start.x, entity.dxf.start.y])
            end = np.array([entity.dxf.end.x, entity.dxf.end.y])
            
            # Convert to world coordinates
            start_world = self.meta.to_world(start)
            end_world = self.meta.to_world(end)
            
            # Create wall as thin obstacle
            self._create_wall_obstacle(start_world, end_world)
        
        # Extract polylines (walls)
        for entity in msp.query('LWPOLYLINE POLYLINE'):
            points = []
            for point in entity.get_points():
                p = np.array([point[0], point[1]])
                points.append(self.meta.to_world(p))
            
            # Create walls from segments
            for i in range(len(points) - 1):
                self._create_wall_obstacle(points[i], points[i + 1])
        
        # Extract circles (could be exits or obstacles)
        exit_layer_names = ['EXIT', 'EXITS', 'DOOR', 'DOORS']
        for entity in msp.query('CIRCLE'):
            center = np.array([entity.dxf.center.x, entity.dxf.center.y])
            center_world = self.meta.to_world(center)
            radius = entity.dxf.radius / self.scale
            
            layer = entity.dxf.layer.upper() if hasattr(entity.dxf, 'layer') else ''
            
            if any(name in layer for name in exit_layer_names):
                # This is an exit
                exit_obj = Exit(
                    exit_id=len(self.exits),
                    position=center_world,
                    width=radius * 2,
                    capacity=100
                )
                self.exits.append(exit_obj)
        
        return self.obstacles, self.exits
    
    def _create_wall_obstacle(self, start: np.ndarray, end: np.ndarray, thickness: float = 0.2):
        """Create obstacle from wall line."""
        # Calculate wall direction and perpendicular
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            return
        
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Create rectangular obstacle
        half_thickness = thickness / 2
        corner1 = start - perpendicular * half_thickness
        corner2 = start + perpendicular * half_thickness
        corner3 = end + perpendicular * half_thickness
        corner4 = end - perpendicular * half_thickness
        
        # Bounding box
        min_x = min(corner1[0], corner2[0], corner3[0], corner4[0])
        min_y = min(corner1[1], corner2[1], corner3[1], corner4[1])
        max_x = max(corner1[0], corner2[0], corner3[0], corner4[0])
        max_y = max(corner1[1], corner2[1], corner3[1], corner4[1])
        
        obstacle = Obstacle(min_x, min_y, max_x - min_x, max_y - min_y)
        self.obstacles.append(obstacle)


class ImageParser:
    """Parse raster images (PNG/JPG) for floorplan geometry."""
    
    def __init__(self, file_path: str, scale: float = 10.0):
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow library not available. Install with: pip install Pillow")
        
        self.file_path = Path(file_path)
        self.scale = scale  # pixels per meter
        self.meta = MapMeta(scale=scale)
        
        self.image = None
        self.walls = []
        self.exits = []
        self.obstacles = []
    
    def parse(self) -> Tuple[List[Obstacle], List[Exit]]:
        """
        Parse image file and extract geometry using adaptive analysis.
        Works with ANY image - no strict color requirements!
        
        Returns:
            Tuple of (obstacles, exits)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.file_path}")
        
        # Load image
        img = Image.open(str(self.file_path))
        self.image = np.array(img.convert('RGB'))
        
        height, width = self.image.shape[:2]
        print(f"  Analyzing {width}×{height} image...")
        
        # Convert to grayscale for analysis
        grayscale = np.mean(self.image, axis=2)
        
        # ADAPTIVE thresholding - works with any image!
        # Calculate histogram to find dark vs light regions
        hist, bin_edges = np.histogram(grayscale, bins=256, range=(0, 256))
        
        # Find threshold using Otsu's method (automatic)
        total = grayscale.size
        sum_total = np.sum(np.arange(256) * hist)
        sum_background = 0
        weight_background = 0
        max_variance = 0
        threshold = 0
        
        for i in range(256):
            weight_background += hist[i]
            if weight_background == 0:
                continue
            
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
            
            sum_background += i * hist[i]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold = i
        
        print(f"  Auto-detected threshold: {threshold:.0f} (dark<{threshold:.0f}, light>={threshold:.0f})")
        
        # ═══════════════════════════════════════════════════════════════
        # FILTER OUT COLORED ELEMENTS (arrows, signs, etc.)
        # ═══════════════════════════════════════════════════════════════
        # Detect colored regions that should NOT be walls
        red_channel = self.image[:, :, 0].astype(float)
        green_channel = self.image[:, :, 1].astype(float)
        blue_channel = self.image[:, :, 2].astype(float)
        
        # Green arrows/signs (bright green)
        green_mask = (green_channel > red_channel + 30) & (green_channel > blue_channel + 30) & (green_channel > 100)
        
        # Red arrows/signs (bright red)
        red_mask = (red_channel > green_channel + 20) & (red_channel > blue_channel + 20) & (red_channel > 100)
        
        # Blue arrows/signs (bright blue)
        blue_mask = (blue_channel > red_channel + 30) & (blue_channel > green_channel + 30) & (blue_channel > 100)
        
        # Yellow arrows/signs (bright yellow)
        yellow_mask = (red_channel > 150) & (green_channel > 150) & (blue_channel < 100)
        
        # Combine all colored regions
        colored_mask = green_mask | red_mask | blue_mask | yellow_mask
        
        print(f"  Detected {np.sum(colored_mask)} colored pixels (arrows/signs) to exclude from walls")
        
        # Create masks using adaptive threshold
        wall_mask = grayscale < threshold
        walkable_mask = grayscale >= threshold
        
        # EXCLUDE colored regions from walls (arrows, signs, etc.)
        wall_mask = wall_mask & ~colored_mask
        
        # ═══════════════════════════════════════════════════════════════
        # REMOVE SMALL ISOLATED REGIONS (labels, text, noise)
        # ═══════════════════════════════════════════════════════════════
        if SCIPY_AVAILABLE:
            print(f"  Filtering out small regions (labels/text)...")
            
            # Label connected components
            labeled_array, num_features = ndimage.label(wall_mask)
            
            # Calculate size of each component
            component_sizes = ndimage.sum(wall_mask, labeled_array, range(num_features + 1))
            
            # Calculate minimum size threshold
            # Labels are typically < 1m² = scale² pixels
            # Walls are typically > 2m² = 4×scale² pixels
            min_wall_size = (self.scale * 1.5) ** 2  # 1.5m × 1.5m minimum
            
            # Remove small components (likely text/labels)
            small_components = component_sizes < min_wall_size
            remove_mask = small_components[labeled_array]
            wall_mask[remove_mask] = False
            
            removed_count = num_features - np.sum(component_sizes >= min_wall_size)
            print(f"  Removed {removed_count} small regions (text/labels), kept {num_features - removed_count} wall segments")
            
            # ═══════════════════════════════════════════════════════════════
            # OPTIONAL: Close small gaps in walls (connect broken segments)
            # ═══════════════════════════════════════════════════════════════
            kernel_size = int(max(3, self.scale * 0.1))  # ~10cm closing
            if kernel_size > 1:
                print(f"  Closing wall gaps with {kernel_size}×{kernel_size} kernel...")
                wall_mask = ndimage.binary_closing(wall_mask, structure=np.ones((kernel_size, kernel_size)))
        else:
            print(f"  Warning: scipy not available - cannot filter text labels. Install with: pip install scipy")
        
        # Debug: show percentages
        wall_percent = 100 * np.sum(wall_mask) / wall_mask.size
        walkable_percent = 100 * np.sum(walkable_mask) / walkable_mask.size
        print(f"  Wall coverage: {wall_percent:.1f}%, Walkable: {walkable_percent:.1f}%")
        
        # Detect exits - look for red-ish areas OR edges of walkable regions
        # Note: red_channel/green_channel/blue_channel already extracted above
        
        # Red detection for exits (more permissive than arrow detection)
        exit_red_mask = (red_channel > green_channel + 20) & (red_channel > blue_channel + 20)
        
        # Create obstacles from wall regions
        print(f"  Extracting obstacles from dark regions...")
        self._extract_obstacles_from_mask(wall_mask)
        
        # Create exits - try red regions first, then use perimeter if no red found
        if np.any(exit_red_mask):
            print(f"  Found red-marked exits")
            self._extract_exits_from_mask(exit_red_mask)
        else:
            print(f"  No red exits found - auto-detecting exits on perimeter...")
            self._auto_detect_exits(walkable_mask)
        
        print(f"  ✓ Extracted {len(self.obstacles)} obstacles, {len(self.exits)} exits")
        
        return self.obstacles, self.exits
    
    def _extract_obstacles_from_mask(self, mask: np.ndarray):
        """Extract rectangular obstacles from binary mask."""
        height, width = mask.shape
        
        # Use smaller cells for better resolution
        cell_size = int(max(5, self.scale * 1))  # ~1 meter per cell, minimum 5 pixels
        
        obstacles_created = 0
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                y_end = min(y + cell_size, height)
                x_end = min(x + cell_size, width)
                cell = mask[y:y_end, x:x_end]
                
                # If majority of cell is wall, create obstacle
                if cell.size > 0 and np.mean(cell) > 0.2955:
                    # Convert pixel position to world coordinates
                    pos = self.meta.to_world(np.array([x, y]))
                    cell_width = (x_end - x) / self.scale
                    cell_height = (y_end - y) / self.scale
                    
                    obstacle = Obstacle(pos[0], pos[1], cell_width, cell_height)
                    self.obstacles.append(obstacle)
                    obstacles_created += 1
        
        print(f"  Created {obstacles_created} obstacle cells (cell size: {cell_size}px = {cell_size/self.scale:.2f}m)")
    
    def _extract_exits_from_mask(self, mask: np.ndarray):
        """Extract MULTIPLE exits from binary mask by finding separate regions."""
        height, width = mask.shape
        
        if not np.any(mask):
            return
        
        # Find all red pixels
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return
        
        # Cluster red pixels into separate exits using simple spatial grouping
        points = np.column_stack([x_coords, y_coords])
        
        # Group nearby points into exits (using a distance threshold)
        min_distance = self.scale * 8  # 8 meters minimum separation between exits (larger to merge close ones)
        exit_clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if used[i]:
                continue
            
            # Start new cluster
            cluster = [points[i]]
            used[i] = True
            
            # Find all nearby points
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                
                # Check if point j is close to any point in current cluster
                for cluster_point in cluster:
                    dist = np.linalg.norm(points[j] - cluster_point)
                    if dist < min_distance:
                        cluster.append(points[j])
                        used[j] = True
                        break
            
            exit_clusters.append(np.array(cluster))
        
        print(f"  Found {len(exit_clusters)} separate exit regions")
        
        # Create exit for each cluster and merge if too close in world coordinates
        temp_exits = []
        for cluster_id, cluster in enumerate(exit_clusters):
            # Get centroid of this exit cluster
            center_x = np.mean(cluster[:, 0])
            center_y = np.mean(cluster[:, 1])
            
            pos = self.meta.to_world(np.array([center_x, center_y]))
            
            # Calculate exit width based on cluster spread
            cluster_width = np.std(cluster[:, 0]) + np.std(cluster[:, 1])
            exit_width = max(1.5, min(3.0, cluster_width / self.scale))  # Between 1.5m and 3m
            
            temp_exits.append({'position': pos, 'width': exit_width})
        
        # Final merge: combine exits within 5 meters of each other
        final_exits = []
        used_temp = [False] * len(temp_exits)
        
        for i in range(len(temp_exits)):
            if used_temp[i]:
                continue
            
            # Start merge group
            merge_positions = [temp_exits[i]['position']]
            merge_widths = [temp_exits[i]['width']]
            used_temp[i] = True
            
            # Find nearby exits to merge
            for j in range(i + 1, len(temp_exits)):
                if used_temp[j]:
                    continue
                    
                dist = np.linalg.norm(temp_exits[j]['position'] - temp_exits[i]['position'])
                if dist < 5.0:  # 5 meters
                    merge_positions.append(temp_exits[j]['position'])
                    merge_widths.append(temp_exits[j]['width'])
                    used_temp[j] = True
            
            # Average merged exits
            final_pos = np.mean(merge_positions, axis=0)
            final_width = np.mean(merge_widths)
            final_exits.append({'position': final_pos, 'width': final_width})
        
        print(f"  Merged into {len(final_exits)} unique exits")
        
        # Create Exit objects
        for exit_data in final_exits:
            exit_obj = Exit(
                exit_id=len(self.exits),
                position=exit_data['position'],
                width=exit_data['width'],
                capacity=100
            )
            self.exits.append(exit_obj)
            print(f"    Exit {exit_obj.id}: position ({exit_obj.position[0]:.1f}, {exit_obj.position[1]:.1f}), width {exit_obj.width:.1f}m")
    
    def _auto_detect_exits(self, walkable_mask: np.ndarray):
        """
        Automatically detect exits on the perimeter of walkable areas.
        Creates exits at the edges of the building.
        """
        height, width = walkable_mask.shape
        
        # Check all four edges for openings
        edges = [
            ('top', walkable_mask[0, :]),
            ('bottom', walkable_mask[-1, :]),
            ('left', walkable_mask[:, 0]),
            ('right', walkable_mask[:, -1])
        ]
        
        for edge_name, edge_data in edges:
            # Find walkable segments on this edge
            walkable_positions = np.where(edge_data)[0]
            
            if len(walkable_positions) > 0:
                # Create exit at middle of walkable segment
                mid_pos = int(np.median(walkable_positions))
                
                if edge_name == 'top':
                    pixel_pos = np.array([mid_pos, 0])
                elif edge_name == 'bottom':
                    pixel_pos = np.array([mid_pos, height - 1])
                elif edge_name == 'left':
                    pixel_pos = np.array([0, mid_pos])
                else:  # right
                    pixel_pos = np.array([width - 1, mid_pos])
                
                world_pos = self.meta.to_world(pixel_pos)
                
                exit_obj = Exit(
                    exit_id=len(self.exits),
                    position=world_pos,
                    width=2.0,
                    capacity=100
                )
                self.exits.append(exit_obj)
        
        # If still no exits found, create default exits at corners
        if len(self.exits) == 0:
            print("  Warning: No perimeter openings found, creating corner exits")
            corners = [
                np.array([width * 0.1, height * 0.1]),
                np.array([width * 0.9, height * 0.1]),
                np.array([width * 0.1, height * 0.9]),
                np.array([width * 0.9, height * 0.9])
            ]
            
            for i, corner in enumerate(corners):
                pos = self.meta.to_world(corner)
                exit_obj = Exit(
                    exit_id=i,
                    position=pos,
                    width=2.0,
                    capacity=100
                )
                self.exits.append(exit_obj)


def load_floorplan(config: dict):
    """
    Load floorplan based on configuration.
    
    Args:
        config: Floorplan configuration
        
    Returns:
        Tuple of (obstacles, exits) or (None, None) if no floorplan
    """
    floorplan_type = config.get('type', 'none')
    
    if floorplan_type == 'none':
        return None, None
    
    file_path = config.get('file_path')
    scale = config.get('scale', 1.0)
    
    if not file_path:
        print("Warning: Floorplan type specified but no file path provided.")
        return None, None
    
    if floorplan_type == 'dxf':
        parser = DXFParser(file_path, scale)
        return parser.parse()
    elif floorplan_type == 'image':
        parser = ImageParser(file_path, scale)
        return parser.parse()
    else:
        print(f"Warning: Unknown floorplan type '{floorplan_type}'")
        return None, None
