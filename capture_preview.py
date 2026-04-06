import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from src.gui.main_window import MainWindow

MODERN_STYLE = """
QMainWindow { background-color: #1e1e1e; }
QWidget { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; color: #e0e0e0; font-size: 10pt; }

QScrollArea { border: none; background-color: transparent; }
QScrollArea > QWidget > QWidget { background-color: transparent; }

QScrollBar:vertical { border: none; background: #1e1e1e; width: 8px; margin: 0px 0px 0px 0px; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #606060; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QToolBox { background-color: transparent; }
QToolBox::tab { background-color: #2d2d30; color: #ffffff; border-radius: 4px; }
QToolBox::tab:selected { background-color: #3e3e42; color: #4daafc; font-weight: bold; }

QPushButton { background-color: #333337; border: 1px solid #3f3f46; border-radius: 4px; padding: 6px 12px; color: #f1f1f1; }
QPushButton:hover { background-color: #3e3e42; border: 1px solid #007acc; }
QPushButton:pressed { background-color: #007acc; color: white; }

QPushButton#primaryButton { background-color: #007acc; color: white; border: none; font-weight: bold; font-size: 11pt; padding: 8px; }
QPushButton#primaryButton:hover { background-color: #0098ff; }
QPushButton#primaryButton:pressed { background-color: #005a9e; }

QSpinBox, QDoubleSpinBox, QComboBox { background-color: #252526; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 8px; color: #ffffff; min-height: 24px; }
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border: 1px solid #007acc; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #252526; border: 1px solid #007acc; selection-background-color: #007acc; color: white; }

/* Group Boxes and Frames */
QFrame#StatCard { background-color: #252526; border: 1px solid #3f3f46; border-radius: 8px; }

QLabel#StatTitle { color: #aaaaaa; font-size: 10pt; font-weight: 500; }
QLabel#StatValue { font-size: 20pt; font-weight: bold; color: #ffffff; }
QLabel#PanelTitle { font-size: 16pt; font-weight: bold; color: #ffffff; margin-bottom: 15px; }

QCheckBox { spacing: 8px; padding: 4px; }
"""

def capture():
    pixmap = window.grab()
    pixmap.save("gui_preview.png")
    app.quit()

app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(MODERN_STYLE)

window = MainWindow()
window.show()

# Use QTimer to give the window a moment to render before grabbing
QTimer.singleShot(1000, capture)

sys.exit(app.exec())
