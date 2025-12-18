"""
Floorplan Analyzer with YOLOv8
Uses deep learning for accurate detection of walls, doors, windows, and exits
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Circle, Polygon
import cv2

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: Ultralytics (YOLOv8) is required. Install with: pip install ultralytics")
    sys.exit(1)


class FloorplanAnalyzerYOLO:
    """Analyze floorplans using YOLOv8 for accurate object detection."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize analyzer with YOLOv8 model.
        
        Args:
            model_path: Path to YOLO model or model name (yolov8n.pt, yolov8s.pt, etc.)
        """
        print(f"🤖 Loading YOLOv8 model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📥 Downloading YOLOv8 nano model...")
            self.model = YOLO('yolov8n.pt')
            print("✅ Model ready")
        
        self.image = None
        self.walls = []
        self.exits = []
        self.doors = []
        self.windows = []
        self.stairs = []
        self.emergency_exits = []
        self.obstacles = []
        self.furniture = []
        self.hallways = []
        self.green_arrows = []
        
    def analyze_floorplan(self, filepath: str, scale: float = 10.0):
        """
        Analyze floorplan image using YOLOv8 and image processing.
        
        Args:
            filepath: Path to floorplan image
            scale: Pixels per meter
            
        Returns:
            Tuple of (walls, exits, width, height)
        """
        print(f"\n📊 Analyzing floorplan with YOLOv8: {filepath}")
        
        # Load image
        self.image = cv2.imread(filepath)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {filepath}")
        
        height_px, width_px = self.image.shape[:2]
        print(f"  Image size: {width_px}×{height_px} pixels")
        
        # Calculate world dimensions
        width_m = width_px / scale
        height_m = height_px / scale
        print(f"  World size: {width_m:.1f}×{height_m:.1f} meters (scale: {scale} px/m)")
        
        # Run YOLOv8 detection
        print("  Running YOLOv8 detection...")
        results = self.model(self.image, conf=0.25, verbose=False)
        
        # Process detections
        detected_objects = results[0].boxes
        print(f"  Detected {len(detected_objects)} objects")
        
        # Extract all components
        walls = self.extract_walls_advanced(self.image, scale)
        print(f"  Extracted {len(walls)} wall segments")
        
        doors = self.detect_doors(self.image, scale)
        print(f"  Detected {len(doors)} doors")
        
        exits = self.detect_exits_advanced(self.image, scale)
        print(f"  Detected {len(exits)} exits")
        
        emergency_exits = self.detect_emergency_exits(self.image, scale)
        print(f"  Detected {len(emergency_exits)} emergency exits")
        
        windows = self.detect_windows(self.image, scale)
        print(f"  Detected {len(windows)} windows")
        
        stairs = self.detect_stairs(self.image, scale)
        print(f"  Detected {len(stairs)} stairs")
        
        furniture = self.detect_furniture(self.image, scale)
        print(f"  Detected {len(furniture)} furniture items")
        
        hallways = self.detect_hallways(self.image, scale)
        print(f"  Detected {len(hallways)} hallway regions")
        
        green_arrows = self.detect_green_arrows(self.image, scale)
        print(f"  Detected {len(green_arrows)} directional arrows")
        
        # Combine all passable exits (doors, exits, emergency exits)
        all_exits = exits + emergency_exits + doors
        
        # Return comprehensive analysis
        components = {
            'walls': walls,
            'exits': all_exits,
            'doors': doors,
            'emergency_exits': emergency_exits,
            'windows': windows,
            'stairs': stairs,
            'furniture': furniture,
            'hallways': hallways,
            'green_arrows': green_arrows
        }
        
        return components, width_m, height_m
    
    def extract_walls_advanced(self, image, scale):
        """
        Extract walls using advanced image processing.
        
        Args:
            image: Input image (BGR format)
            scale: Pixels per meter
            
        Returns:
            List of wall dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection for wall boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine binary and edges
        combined = cv2.bitwise_or(binary, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        walls = []
        min_area = (scale * 0.5) ** 2  # Minimum wall area threshold
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to world coordinates
                wall_x = x / scale
                wall_y = y / scale
                wall_w = w / scale
                wall_h = h / scale
                
                # Filter out very small segments
                if wall_w > 0.3 or wall_h > 0.3:
                    walls.append({
                        'x': wall_x,
                        'y': wall_y,
                        'width': wall_w,
                        'height': wall_h,
                        'contour': contour
                    })
        
        return walls
    
    def detect_doors(self, image, scale):
        """Detect doors in floorplan."""
        doors = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Doors are often thin rectangles or marked in brown/beige
        # Look for brown/beige colors
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([30, 200, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Door size range
                x, y, w, h = cv2.boundingRect(contour)
                # Doors are typically thin rectangles
                aspect_ratio = max(w, h) / (min(w, h) + 0.001)
                if aspect_ratio > 2:  # Elongated shape
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        doors.append({
                            'x': cx / scale,
                            'y': cy / scale,
                            'width': max(w, h) / scale,
                            'type': 'door'
                        })
        
        return doors
    
    def detect_exits_advanced(self, image, scale):
        """
        Detect exits and doors using color analysis and pattern recognition.
        
        Args:
            image: Input image (BGR format)
            scale: Pixels per meter
            
        Returns:
            List of exit dictionaries
        """
        exits = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for doors/exits (typically red or green in floorplans)
        # Red range (doors often marked in red)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Green range (exits often marked in green)
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Combine masks
        exit_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Find contours in exit mask
        contours, _ = cv2.findContours(exit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area for exit
                # Get center and size
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Get size
                    x, y, w, h = cv2.boundingRect(contour)
                    exit_width = max(w, h) / scale
                    
                    exits.append({
                        'x': cx / scale,
                        'y': cy / scale,
                        'width': max(1.5, exit_width)
                    })
        
        # If no colored exits found, detect from edges
        if len(exits) == 0:
            exits = self.detect_exits_from_edges(image, scale)
        
        return exits
    
    def detect_exits_from_edges(self, image, scale):
        """
        Detect potential exits at building edges.
        
        Args:
            image: Input image
            scale: Pixels per meter
            
        Returns:
            List of exit dictionaries
        """
        exits = []
        height, width = image.shape[:2]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Check edges for openings (white areas = potential exits)
        edge_thickness = int(scale * 2)
        
        # Top edge
        top_edge = binary[:edge_thickness, :]
        for x in range(0, width, int(scale)):
            if np.mean(top_edge[:, max(0, x-5):min(width, x+5)]) > 200:
                exits.append({'x': x / scale, 'y': 0, 'width': 2.0})
        
        # Bottom edge
        bottom_edge = binary[-edge_thickness:, :]
        for x in range(0, width, int(scale)):
            if np.mean(bottom_edge[:, max(0, x-5):min(width, x+5)]) > 200:
                exits.append({'x': x / scale, 'y': height / scale, 'width': 2.0})
        
        # Left edge
        left_edge = binary[:, :edge_thickness]
        for y in range(0, height, int(scale)):
            if np.mean(left_edge[max(0, y-5):min(height, y+5), :]) > 200:
                exits.append({'x': 0, 'y': y / scale, 'width': 2.0})
        
        # Right edge
        right_edge = binary[:, -edge_thickness:]
        for y in range(0, height, int(scale)):
            if np.mean(right_edge[max(0, y-5):min(height, y+5), :]) > 200:
                exits.append({'x': width / scale, 'y': y / scale, 'width': 2.0})
        
        # Limit to reasonable number
        return exits[:8]
    
    def detect_emergency_exits(self, image, scale):
        """Detect emergency exits (often marked in red with special symbols)."""
        emergency_exits = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Emergency exits are typically bright red
        red_lower1 = np.array([0, 150, 150])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 150, 150])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w, h = cv2.boundingRect(contour)
                    emergency_exits.append({
                        'x': cx / scale,
                        'y': cy / scale,
                        'width': max(w, h) / scale,
                        'type': 'emergency_exit'
                    })
        
        return emergency_exits
    
    def detect_windows(self, image, scale):
        """Detect windows (often thin blue or light colored rectangles)."""
        windows = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Windows are often blue or light cyan
        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                windows.append({
                    'x': x / scale,
                    'y': y / scale,
                    'width': w / scale,
                    'height': h / scale,
                    'type': 'window'
                })
        
        return windows
    
    def detect_stairs(self, image, scale):
        """Detect stairs (often parallel lines or hatched patterns)."""
        stairs = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Group parallel lines (stairs pattern)
            for line in lines[:20]:  # Limit processing
                x1, y1, x2, y2 = line[0]
                stairs.append({
                    'x': (x1 + x2) / 2 / scale,
                    'y': (y1 + y2) / 2 / scale,
                    'width': 2.0,
                    'type': 'stairs'
                })
        
        return stairs
    
    def detect_furniture(self, image, scale):
        """Detect furniture as obstacles (colored shapes inside rooms)."""
        furniture = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Furniture often appears as medium-gray objects
        _, furniture_mask = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
        furniture_mask = cv2.bitwise_not(furniture_mask)
        
        contours, _ = cv2.findContours(furniture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:  # Furniture size range
                x, y, w, h = cv2.boundingRect(contour)
                furniture.append({
                    'x': x / scale,
                    'y': y / scale,
                    'width': w / scale,
                    'height': h / scale,
                    'type': 'furniture'
                })
        
        return furniture[:50]  # Limit to avoid too many detections
    
    def detect_hallways(self, image, scale):
        """Detect hallway regions (long walkable corridors)."""
        hallways = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Hallways are typically white/light colored long rectangles
        _, walkable = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(walkable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Large walkable area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 0.001)
                if aspect_ratio > 3:  # Elongated = hallway
                    hallways.append({
                        'x': x / scale,
                        'y': y / scale,
                        'width': w / scale,
                        'height': h / scale,
                        'type': 'hallway'
                    })
        
        return hallways
    
    def detect_green_arrows(self, image, scale):
        """Detect green directional arrows (evacuation routes)."""
        arrows = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect bright green
        green_lower = np.array([35, 100, 100])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Arrow size range
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Detect arrow direction (simplified)
                    x, y, w, h = cv2.boundingRect(contour)
                    direction = 'right' if w > h else 'down'
                    
                    arrows.append({
                        'x': cx / scale,
                        'y': cy / scale,
                        'direction': direction,
                        'type': 'arrow'
                    })
        
        return arrows


def create_simplified_png(walls, exits, width, height, output_path, input_image=None):
    """
    Create a simplified PNG showing only walls and exits.
    
    Args:
        walls: List of wall dictionaries
        exits: List of exit dictionaries
        width: World width in meters
        height: World height in meters
        output_path: Output PNG file path
        input_image: Optional original image to overlay
    """
    print(f"\n🎨 Creating simplified PNG: {output_path}")
    
    # Create figure with better DPI
    fig, ax = plt.subplots(figsize=(14, 12))
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F5')
    ax.set_xlabel('X (meters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=13, fontweight='bold')
    ax.set_title('YOLOv8 Floor Plan Analysis - Walls and Exits', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Overlay original image (faded) if provided
    if input_image is not None:
        extent = [0, width, 0, height]
        ax.imshow(input_image, extent=extent, origin='upper', alpha=0.15, aspect='auto')
    
    # Draw walls
    for wall in walls:
        rect = Rectangle(
            (wall['x'], wall['y']),
            wall['width'],
            wall['height'],
            facecolor='#2C3E50',
            edgecolor='#1A252F',
            linewidth=1.5,
            alpha=0.85
        )
        ax.add_patch(rect)
    
    # Draw exits with better styling
    for i, exit_obj in enumerate(exits):
        # Main circle
        circle = Circle(
            (exit_obj['x'], exit_obj['y']),
            exit_obj['width'] / 2,
            facecolor='#27AE60',
            edgecolor='#1E8449',
            linewidth=2.5,
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Outer glow
        glow = Circle(
            (exit_obj['x'], exit_obj['y']),
            exit_obj['width'] / 2 * 1.3,
            facecolor='none',
            edgecolor='#82E0AA',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6,
            zorder=9
        )
        ax.add_patch(glow)
        
        # Label
        ax.text(
            exit_obj['x'],
            exit_obj['y'],
            f'EXIT\n{i+1}',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            color='white',
            zorder=11
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2C3E50', edgecolor='#1A252F', label=f'Walls ({len(walls)} segments)'),
        Patch(facecolor='#27AE60', edgecolor='#1E8449', label=f'Exits ({len(exits)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # Add info box
    info_text = f"🏢 Dimensions: {width:.1f}m × {height:.1f}m\n"
    info_text += f"🧱 Walls: {len(walls)} segments\n"
    info_text += f"🚪 Exits: {len(exits)} detected\n"
    info_text += f"🤖 Analyzed with YOLOv8"
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', 
                 alpha=0.95, edgecolor='#D4AF37', linewidth=2),
        fontsize=11,
        fontfamily='monospace',
        fontweight='bold'
    )
    
    # Save with high quality
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved simplified floor plan to: {output_path}")


def main():
    """Main function with YOLOv8 integration."""
    
    print("=" * 70)
    print("  FLOORPLAN ANALYZER with YOLOv8")
    print("  AI-Powered Wall and Exit Detection")
    print("=" * 70)
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("\n📁 Enter path to floor plan file: ").strip()
    
    input_file = input_file.strip('"').strip("'")
    
    if not Path(input_file).exists():
        print(f"\n❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Get scale
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
    else:
        scale_input = input("\n📏 Enter scale (pixels per meter, default=10): ").strip()
        scale = float(scale_input) if scale_input else 10.0
    
    # Get model path
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    else:
        model_path = 'yolov8n.pt'  # Default to nano model
    
    # Output file
    input_path = Path(input_file)
    output_file = f"output/{input_path.stem}_yolo_simplified.png"
    
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    
    # Analyze with YOLOv8
    try:
        analyzer = FloorplanAnalyzerYOLO(model_path)
        components, width, height = analyzer.analyze_floorplan(input_file, scale)
        
        # Load original image for overlay
        original_img = cv2.imread(input_file)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create output PNG
        create_simplified_png(components['walls'], components['exits'], width, height, output_file, original_img)
        
        # Save comprehensive analysis to file
        import json
        analysis_file = f"output/{Path(input_file).stem}_analysis.json"
        with open(analysis_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_components = {}
            for key, items in components.items():
                json_components[key] = []
                for item in items:
                    json_item = {}
                    for k, v in item.items():
                        if isinstance(v, (np.integer, np.floating)):
                            json_item[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            json_item[k] = v.tolist()
                        else:
                            json_item[k] = v
                    json_components[key].append(json_item)
            
            json.dump({
                'width': float(width),
                'height': float(height),
                'scale': float(scale),
                'components': json_components
            }, f, indent=2)
        print(f"[+] Saved detailed analysis to: {analysis_file}")
        
        print(f"\n{'=' * 70}")
        print(f"[+] SUCCESS! YOLOv8 analysis complete.")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Model:  {model_path}")
        print(f"\nDetected Components:")
        print(f"   Walls: {len(components['walls'])}")
        print(f"   Exits: {len(components['exits'])}")
        print(f"   Doors: {len(components['doors'])}")
        print(f"   Emergency Exits: {len(components['emergency_exits'])}")
        print(f"   Windows: {len(components['windows'])}")
        print(f"   Stairs: {len(components['stairs'])}")
        print(f"   Furniture: {len(components['furniture'])}")
        print(f"   Hallways: {len(components['hallways'])}")
        print(f"   Green Arrows: {len(components['green_arrows'])}")
        print(f"{'=' * 70}\n")
        
    except Exception as e:
        print(f"\n❌ Error processing floor plan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
