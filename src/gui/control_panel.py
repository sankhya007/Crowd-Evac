from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QToolBox,
    QScrollArea
)

from PyQt6.QtCore import Qt


class ControlPanel(QWidget):

    def __init__(self):
        super().__init__()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)

        # Title
        title = QLabel("Simulation Controls")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # Browse floorplan
        self.load_btn = QPushButton("Browse")
        layout.addWidget(self.load_btn)

        # Run simulation
        self.run_btn = QPushButton("Run Simulation")
        layout.addWidget(self.run_btn)

        # Exit controls
        self.add_exit_btn = QPushButton("Add Exit")
        layout.addWidget(self.add_exit_btn)

        self.undo_exit_btn = QPushButton("Undo Exit")
        layout.addWidget(self.undo_exit_btn)

        # ---------------- TOOLBOX ----------------

        toolbox = QToolBox()

        # ===== Simulation Section =====
        sim_widget = QWidget()
        sim_layout = QVBoxLayout()

        sim_layout.addWidget(QLabel("Agents"))
        self.agent_input = QSpinBox()
        self.agent_input.setRange(10, 5000)
        self.agent_input.setValue(50)
        sim_layout.addWidget(self.agent_input)

        sim_layout.addWidget(QLabel("Duration (s)"))
        self.duration_input = QSpinBox()
        self.duration_input.setRange(10, 3600)
        self.duration_input.setValue(300)
        sim_layout.addWidget(self.duration_input)

        sim_layout.addWidget(QLabel("Scale (px/m)"))
        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(1, 100)
        self.scale_input.setValue(10)
        sim_layout.addWidget(self.scale_input)

        sim_widget.setLayout(sim_layout)
        toolbox.addItem(sim_widget, "Simulation")

        # ===== Motion Model =====
        motion_widget = QWidget()
        motion_layout = QVBoxLayout()

        motion_layout.addWidget(QLabel("Motion Model"))
        self.motion_model = QComboBox()
        self.motion_model.addItems(["sfm", "rvo", "pathfinding", "hybrid"])
        motion_layout.addWidget(self.motion_model)

        motion_widget.setLayout(motion_layout)
        toolbox.addItem(motion_widget, "Motion Model")

        # ===== Agent Behaviour =====
        agent_widget = QWidget()
        agent_layout = QVBoxLayout()

        agent_layout.addWidget(QLabel("Speed Min"))
        self.speed_min = QDoubleSpinBox()
        self.speed_min.setValue(0.8)
        agent_layout.addWidget(self.speed_min)

        agent_layout.addWidget(QLabel("Speed Max"))
        self.speed_max = QDoubleSpinBox()
        self.speed_max.setValue(1.8)
        agent_layout.addWidget(self.speed_max)

        agent_layout.addWidget(QLabel("Panic Threshold"))
        self.panic_threshold = QDoubleSpinBox()
        self.panic_threshold.setValue(0.3)
        agent_layout.addWidget(self.panic_threshold)

        agent_widget.setLayout(agent_layout)
        toolbox.addItem(agent_widget, "Agent Behaviour")

        # ===== Hazards =====
        hazard_widget = QWidget()
        hazard_layout = QVBoxLayout()

        self.fire_checkbox = QCheckBox("Enable Fire")
        self.fire_checkbox.setChecked(True)

        self.smoke_checkbox = QCheckBox("Enable Smoke")
        self.smoke_checkbox.setChecked(True)

        self.exit_fail_checkbox = QCheckBox("Enable Exit Failures")
        self.exit_fail_checkbox.setChecked(True)

        hazard_layout.addWidget(self.fire_checkbox)
        hazard_layout.addWidget(self.smoke_checkbox)
        hazard_layout.addWidget(self.exit_fail_checkbox)

        hazard_widget.setLayout(hazard_layout)
        toolbox.addItem(hazard_widget, "Hazards")

        # ===== Visualization =====
        vis_widget = QWidget()
        vis_layout = QVBoxLayout()

        self.show_traj = QCheckBox("Show Trajectories")
        self.show_traj.setChecked(True)

        self.show_panic = QCheckBox("Show Panic Levels")
        self.show_panic.setChecked(True)

        self.show_hazards = QCheckBox("Show Hazards")
        self.show_hazards.setChecked(True)

        vis_layout.addWidget(self.show_traj)
        vis_layout.addWidget(self.show_panic)
        vis_layout.addWidget(self.show_hazards)

        vis_widget.setLayout(vis_layout)
        toolbox.addItem(vis_widget, "Visualization")

        # ===== Analytics =====
        analytics_widget = QWidget()
        analytics_layout = QVBoxLayout()

        self.enable_analytics = QCheckBox("Enable Analytics")
        self.enable_analytics.setChecked(True)

        analytics_layout.addWidget(self.enable_analytics)

        analytics_widget.setLayout(analytics_layout)
        toolbox.addItem(analytics_widget, "Analytics")

        layout.addWidget(toolbox)

        layout.addStretch()

        container.setLayout(layout)
        scroll.setWidget(container)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)
        
        self.auto_exit_detection = QCheckBox("Auto Detect Exits (AI)")
        self.auto_exit_detection.setChecked(True)
        sim_layout.addWidget(self.auto_exit_detection)
