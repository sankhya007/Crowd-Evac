from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QSlider
)

from PyQt6.QtCore import Qt


class ControlPanel(QWidget):

    def __init__(self):
        super().__init__()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Prevent scroll
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title
        title = QLabel("Simulation Controls")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("margin-bottom: 2px;")
        layout.addWidget(title)

        # File & Execution
        btn_layout = QGridLayout()
        btn_layout.setSpacing(6)
        
        self.load_btn = QPushButton("Browse")
        btn_layout.addWidget(self.load_btn, 0, 0)
        
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setObjectName("primaryButton")
        btn_layout.addWidget(self.run_btn, 0, 1)

        self.add_exit_btn = QPushButton("Add Exit")
        btn_layout.addWidget(self.add_exit_btn, 1, 0)
        
        self.undo_exit_btn = QPushButton("Undo Exit")
        btn_layout.addWidget(self.undo_exit_btn, 1, 1)

        self.add_wall_btn = QPushButton("Draw Wall")
        btn_layout.addWidget(self.add_wall_btn, 2, 0)
        
        self.clear_wall_btn = QPushButton("Clear Walls")
        btn_layout.addWidget(self.clear_wall_btn, 2, 1)
        
        self.add_hazard_btn = QPushButton("Add Hazard")
        btn_layout.addWidget(self.add_hazard_btn, 3, 0)

        self.undo_hazard_btn = QPushButton("Undo Hazard")
        btn_layout.addWidget(self.undo_hazard_btn, 3, 1)
        
        layout.addLayout(btn_layout)

        # ---------------- SECTIONS ----------------

        # ===== Advanced Playback =====
        adv_widget = QGroupBox("Playback")
        adv_layout = QVBoxLayout()
        adv_layout.setContentsMargins(5, 12, 5, 5)
        adv_layout.setSpacing(5)

        speed_label = QLabel("Simulation Speed Multiplier:")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(1)
        
        adv_layout.addWidget(speed_label)
        adv_layout.addWidget(self.speed_slider)

        adv_widget.setLayout(adv_layout)
        layout.addWidget(adv_widget)

        # ===== Simulation Section =====
        sim_widget = QGroupBox("Simulation")
        sim_layout = QGridLayout()
        sim_layout.setContentsMargins(5, 12, 5, 5)
        sim_layout.setSpacing(5)

        sim_layout.setHorizontalSpacing(2)

        self.agent_input = QSpinBox()
        self.agent_input.setRange(10, 5000)
        self.agent_input.setValue(50)
        sim_layout.addWidget(QLabel("Agent Count:"), 0, 0)
        sim_layout.addWidget(self.agent_input, 0, 1)

        self.duration_input = QSpinBox()
        self.duration_input.setRange(10, 3600)
        self.duration_input.setValue(300)
        sim_layout.addWidget(QLabel("Duration (s):"), 0, 2)
        sim_layout.addWidget(self.duration_input, 0, 3)

        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(1, 100)
        self.scale_input.setValue(10)
        sim_layout.addWidget(QLabel("Pixels/Meter:"), 1, 0)
        sim_layout.addWidget(self.scale_input, 1, 1)

        self.auto_exit_detection = QCheckBox("Auto Exits")
        self.auto_exit_detection.setChecked(True)
        sim_layout.addWidget(self.auto_exit_detection, 1, 2, 1, 2)

        sim_widget.setLayout(sim_layout)
        layout.addWidget(sim_widget)

        # ===== Motion Model =====
        motion_widget = QGroupBox("Motion Model")
        motion_layout = QFormLayout()
        motion_layout.setContentsMargins(5, 12, 5, 5)
        motion_layout.setSpacing(5)

        self.motion_model = QComboBox()
        self.motion_model.addItems(["sfm", "rvo", "pathfinding", "hybrid"])
        motion_layout.addRow("Model:", self.motion_model)

        motion_widget.setLayout(motion_layout)
        layout.addWidget(motion_widget)

        # ===== Agent Behaviour =====
        agent_widget = QGroupBox("Agent Behaviour")
        agent_layout = QGridLayout()
        agent_layout.setContentsMargins(5, 12, 5, 5)
        agent_layout.setSpacing(5)

        agent_layout.setHorizontalSpacing(2)

        self.speed_min = QDoubleSpinBox()
        self.speed_min.setValue(0.8)
        agent_layout.addWidget(QLabel("Minimum Speed:"), 0, 0)
        agent_layout.addWidget(self.speed_min, 0, 1)

        self.speed_max = QDoubleSpinBox()
        self.speed_max.setValue(1.8)
        agent_layout.addWidget(QLabel("Maximum Speed:"), 0, 2)
        agent_layout.addWidget(self.speed_max, 0, 3)

        self.panic_threshold = QDoubleSpinBox()
        self.panic_threshold.setValue(0.3)
        agent_layout.addWidget(QLabel("Panic Threshold:"), 1, 0)
        agent_layout.addWidget(self.panic_threshold, 1, 1)

        agent_widget.setLayout(agent_layout)
        layout.addWidget(agent_widget)

        # ===== Hazards =====
        hazard_widget = QGroupBox("Hazards")
        hazard_layout = QVBoxLayout()
        hazard_layout.setContentsMargins(5, 12, 5, 5)
        hazard_layout.setSpacing(5)

        h_checks = QHBoxLayout()
        self.fire_checkbox = QCheckBox("Fire")
        self.fire_checkbox.setChecked(True)

        self.smoke_checkbox = QCheckBox("Smoke")
        self.smoke_checkbox.setChecked(True)

        self.exit_fail_checkbox = QCheckBox("Failures")
        self.exit_fail_checkbox.setChecked(True)

        h_checks.addWidget(self.fire_checkbox)
        h_checks.addWidget(self.smoke_checkbox)
        h_checks.addWidget(self.exit_fail_checkbox)
        hazard_layout.addLayout(h_checks)

        h_input = QHBoxLayout()
        h_input.addWidget(QLabel("Hazard Start Time (s):"))
        self.fire_start_input = QDoubleSpinBox()
        self.fire_start_input.setRange(0, 3600)
        self.fire_start_input.setValue(10.0)
        h_input.addWidget(self.fire_start_input)
        hazard_layout.addLayout(h_input)

        hazard_widget.setLayout(hazard_layout)
        layout.addWidget(hazard_widget)

        # ===== Visualization & Analytics =====
        vis_widget = QGroupBox("Visualization & Analytics")
        vis_layout = QGridLayout()
        vis_layout.setContentsMargins(5, 12, 5, 5)
        vis_layout.setSpacing(5)

        self.show_traj = QCheckBox("Trajectories")
        self.show_traj.setChecked(True)
        self.show_panic = QCheckBox("Panic Levels")
        self.show_panic.setChecked(True)
        self.show_hazards = QCheckBox("Hazards")
        self.show_hazards.setChecked(True)
        self.enable_analytics = QCheckBox("Analytics")
        self.enable_analytics.setChecked(True)

        vis_layout.addWidget(self.show_traj, 0, 0)
        vis_layout.addWidget(self.show_panic, 0, 1)
        vis_layout.addWidget(self.show_hazards, 1, 0)
        vis_layout.addWidget(self.enable_analytics, 1, 1)

        vis_widget.setLayout(vis_layout)
        layout.addWidget(vis_widget)

        layout.addStretch()

        container.setLayout(layout)
        scroll.setWidget(container)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)
