from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np


class MapView(FigureCanvasQTAgg):

    def __init__(self):

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.ax.axis("off")

        self.exits = []
        self.exit_mode = False

        # zoom limits
        self.min_zoom = 5
        self.max_zoom = 2000

        # pan state
        self.panning = False
        self.pan_start = None

        # mouse controls
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("button_press_event", self.on_mouse_press)
        self.mpl_connect("button_release_event", self.on_mouse_release)
        self.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.mpl_connect("button_press_event", self.on_double_click)
        

    # -------------------------------------------------------
    # Load floorplan
    # -------------------------------------------------------

    def load_image(self, path):

        self.image_path = path

        img = Image.open(path)
        img_array = np.array(img)

        h, w, _ = img_array.shape

        self.ax.clear()

        self.ax.imshow(
            img_array,
            extent=[0, w, 0, h],
            # origin="lower"
        )

        self.ax.axis("off")

        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        self.home_xlim = (0, w)
        self.home_ylim = (0, h)

        self.ax.set_aspect("auto")

        self.draw()

    # -------------------------------------------------------
    # Exit placement
    # -------------------------------------------------------

    # def mousePressEvent(self, event):

    #     if self.exit_mode:

    #         x = int(event.position().x())
    #         y = int(event.position().y())
            
    #         y = self.ax.get_ylim()[0] + self.ax.get_ylim()[1] - y

    #         self.exits.append((x, y))

    #         print("Exit added at:", x, y)

    #         self.ax.scatter(x, y, c="red", s=80, zorder=5)

    #         self.draw_idle()

    #         self.exit_mode = False

    # -------------------------------------------------------
    # Zoom with scroll wheel
    # -------------------------------------------------------

    def on_scroll(self, event):

        zoom_factor = 1.2

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        if event.button == "up":
            scale = 1 / zoom_factor
        else:
            scale = zoom_factor

        width = (cur_xlim[1] - cur_xlim[0])
        height = (cur_ylim[1] - cur_ylim[0])

        new_width = width * scale
        new_height = height * scale

        if new_width < self.min_zoom or new_width > self.max_zoom:
            return

        relx = (cur_xlim[1] - xdata) / width
        rely = (cur_ylim[1] - ydata) / height

        self.ax.set_xlim(
            xdata - new_width * (1 - relx),
            xdata + new_width * relx
        )

        self.ax.set_ylim(
            ydata - new_height * (1 - rely),
            ydata + new_height * rely
        )

        self.draw_idle()

    # -------------------------------------------------------
    # Drag to pan
    # -------------------------------------------------------

    def on_mouse_press(self, event):

        if event.xdata is None or event.ydata is None:
            return

        # EXIT PLACEMENT
        if self.exit_mode:

            x = event.xdata
            y = event.ydata

            self.exits.append((x, y))

            print("Exit added at:", x, y)

            self.ax.scatter(x, y, c="red", s=80, zorder=5)

            self.draw_idle()

            self.exit_mode = False
            return

        # PAN START
        if event.button == 1 and event.dblclick is False:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):

        self.panning = False
        self.pan_start = None

    def on_mouse_move(self, event):

        if not self.panning or event.xdata is None:
            return

        x0, y0 = self.pan_start

        dx = event.xdata - x0
        dy = event.ydata - y0

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
        self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)

        self.pan_start = (event.xdata, event.ydata)

        self.draw_idle()

    # -------------------------------------------------------
    # Double click = reset view
    # -------------------------------------------------------

    def on_double_click(self, event):

        if event.dblclick:

            self.ax.set_xlim(self.home_xlim)
            self.ax.set_ylim(self.home_ylim)

            self.draw_idle()
            
    def fit_to_window(self):

        if not hasattr(self, "home_xlim"):
            return

        self.ax.set_xlim(self.home_xlim)
        self.ax.set_ylim(self.home_ylim)

        self.draw_idle()
        
    def zoom_to_agents(self, agents):

        if not agents:
            return

        xs = [a.position[0] for a in agents]
        ys = [a.position[1] for a in agents]

        padding = 5

        self.ax.set_xlim(min(xs)-padding, max(xs)+padding)
        self.ax.set_ylim(min(ys)-padding, max(ys)+padding)

        self.draw_idle()