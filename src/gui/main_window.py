from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QPushButton,
    QLabel,
    QFrame
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from networkx import config
from .map_view import MapView
from src.visualizer import Visualizer
from .control_panel import ControlPanel

from src.simulation_engine import SimulationEngine
import yaml
import numpy as np


class MainWindow(QMainWindow):
    
    def create_stat_box(self, title):

        box = QFrame()

        box.setStyleSheet("""
            background-color: #2b2b2b;
            border-radius: 8px;
            padding: 4px;
            
        """)
        # margin-bottom: 6px;
            
        layout = QVBoxLayout(box)

        title_label = QLabel(title)
        value_label = QLabel("0")

        title_label.setStyleSheet("color: gray;")
        value_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        return box, value_label

    def __init__(self):
        super().__init__()

        self.setWindowTitle("TRAGIC Crowd Simulator")
        self.setGeometry(100, 100, 1400, 800)

        layout = QHBoxLayout()

        # Left control panel
        self.control_panel = ControlPanel()

        # Map view
        self.map_view = MapView()
        
        # Right panel
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)

        # Stats boxes
        self.sim_time_box, self.sim_time_label = self.create_stat_box("Sim Time")
        self.active_box, self.active_label = self.create_stat_box("Active Agents")
        self.evacuated_box, self.evacuated_label = self.create_stat_box("Evacuated")
        self.casualties_box, self.casualties_label = self.create_stat_box("Casualties")
        self.panic_box, self.panic_label = self.create_stat_box("Avg Panic")
        self.fire_box, self.fire_label = self.create_stat_box("Fire Cells")

        right_layout.addWidget(self.sim_time_box)
        right_layout.addWidget(self.active_box)
        right_layout.addWidget(self.evacuated_box)
        right_layout.addWidget(self.casualties_box)
        right_layout.addWidget(self.panic_box)
        right_layout.addWidget(self.fire_box)

        # Mini map placeholder
        # self.mini_map = QLabel()
        
        self.mini_map = QLabel("Mini Map")
        self.mini_map.setFixedHeight(200)
        self.mini_map.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mini_map.setStyleSheet("""
        background-color:#1e1e1e;
        color:white;
        border-radius:8px;
        """)
        self.mini_map.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(self.mini_map)

        # Fit map button
        self.fit_btn = QPushButton("Fit Map")
        right_layout.addWidget(self.fit_btn)

        right_layout.addStretch()

        self.fit_btn.clicked.connect(self.map_view.fit_to_window)

        layout.addWidget(self.control_panel, 3)
        layout.addWidget(self.map_view, 12)
        layout.addWidget(self.right_panel, 3)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        # Connect buttons
        self.control_panel.load_btn.clicked.connect(self.load_floorplan)
        self.control_panel.run_btn.clicked.connect(self.run_simulation)
        self.control_panel.add_exit_btn.clicked.connect(self.enable_exit_mode)
        self.control_panel.undo_exit_btn.clicked.connect(self.undo_exit)
        
        self.setMinimumSize(1200, 700)

    def load_floorplan(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Floorplan",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.map_view.load_image(file_path)

            # Update mini map preview
            pixmap = QPixmap(file_path)

            scaled = pixmap.scaled(
                180,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.mini_map.setPixmap(scaled)
            
            
    def run_simulation(self):

        agents = self.control_panel.agent_input.value()
        duration = self.control_panel.duration_input.value()
        scale = self.control_panel.scale_input.value()

        exits = self.map_view.exits

        print("Starting simulation")
        print("Agents:", agents)
        print("Exit positions:", exits)

        if not hasattr(self.map_view, "image_path"):
            print("No floorplan loaded")
            return

        import yaml
        from src.simulation_engine import SimulationEngine

        with open("config.yaml") as f:
            config = yaml.safe_load(f)
            
        from src.floorplan_parser import ImageParser

        parser = ImageParser(self.map_view.image_path, scale)
        obstacles, exits_from_image = parser.parse()

        config["environment"]["floorplan_obstacles"] = obstacles
        
        from PIL import Image

        img = Image.open(self.map_view.image_path)
        w, h = img.size

        config["environment"]["width"] = w / scale
        config["environment"]["height"] = h / scale

        # If user placed exits -> disable auto exits
        auto_detect = self.control_panel.auto_exit_detection.isChecked()

        if auto_detect:

            print("Auto detecting exits from floorplan")
            config["environment"]["floorplan_exits"] = exits_from_image

        else:

            if len(self.map_view.exits) > 0:

                print("Using manual exits from GUI")
                config["environment"]["floorplan_exits"] = None

            else:

                print("No exits placed — simulation may fail")
                config["environment"]["floorplan_exits"] = []

        config["agents"]["count"] = agents
        config["simulation"]["duration"] = duration
        config["floorplan"] = {
            "type": "image",
            "file_path": self.map_view.image_path,
            "scale": scale,
            "origin": [0, 0]
        }

        config["canvas"] = self.map_view

        engine = SimulationEngine(config)

        engine.visualizer = Visualizer(
            config["visualization"],
            engine.environment,
            canvas=self.map_view
        )

        # -----------------------------------------
        # Override exits ONLY if user placed them
        # -----------------------------------------

        from src.environment import Exit
        import numpy as np

        if len(self.map_view.exits) > 0:

            print("Using manual exits from GUI")

            engine.environment.exits = []

            for i, (px, py) in enumerate(self.map_view.exits):

                inv = self.map_view.ax.transData.inverted()
                sim_x, sim_y = inv.transform((px, py))

                engine.environment.add_exit(
                    Exit(
                        exit_id=i,
                        position=np.array([sim_x, sim_y]),
                        width=2.0,
                        capacity=100
                    )
                )

        # -----------------------------------------
        # START SIMULATION LOOP
        # -----------------------------------------

        import time

        print("Simulation running...")

        while engine.current_time < config["simulation"]["duration"]:

            engine.step()

            self.update_stats(engine)

            self.map_view.draw_idle()

            QApplication.processEvents()

            time.sleep(engine.dt)

        print("Simulation finished")

        engine._finalize()
        
    def update_stats(self, engine):

        # Simulation time
        self.sim_time_label.setText(f"{engine.current_time:.1f}s")

        # Active agents
        active = len([a for a in engine.agents if a.alive and not a.evacuated])
        self.active_label.setText(str(active))

        # Evacuated
        evacuated = len([a for a in engine.agents if a.evacuated])
        total = len(engine.agents)

        if total > 0:
            percent = (evacuated / total) * 100
        else:
            percent = 0

        self.evacuated_label.setText(f"{evacuated} ({percent:.0f}%)")

        # Casualties
        casualties = len([a for a in engine.agents if not a.alive])
        self.casualties_label.setText(str(casualties))

        # Average panic
        if engine.agents:
            avg_panic = sum(a.panic_level for a in engine.agents) / len(engine.agents)
        else:
            avg_panic = 0

        self.panic_label.setText(f"{avg_panic:.2f}")

        # Fire cells
        fire_cells = len(engine.hazard_manager.get_fire_positions())
        self.fire_label.setText(str(fire_cells))
        
    def enable_exit_mode(self):

        print("Exit placement mode enabled")

        self.map_view.exit_mode = True


    def undo_exit(self):

        if self.map_view.exits:
            self.map_view.exits.pop()

            print("Last exit removed")

            self.map_view.update()