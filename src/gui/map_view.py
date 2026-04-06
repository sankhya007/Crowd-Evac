from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np


class MapView(FigureCanvasQTAgg):

    def __init__(self):

        self.figure = Figure()
        self.figure.patch.set_facecolor('#0f172a')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')

        super().__init__(self.figure)

        self.ax.axis("off")

        self.exits = []
        self.exit_mode = False

        self.hazards = []
        self.hazard_mode = False

        self.user_walls = []
        self.wall_patches = []
        self.wall_mode = False
        self.wall_start = None
        self.temp_wall_patch = None

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

        self.ax.imshow(img_array)

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
            # print("Exit added at:", x, y)

            self.ax.scatter(x, y, c="#06b6d4", s=80, marker="s", zorder=5)
            self.draw_idle()

            self.exit_mode = False
            return

        # HAZARD PLACEMENT
        if self.hazard_mode:

            x = event.xdata
            y = event.ydata

            self.hazards.append((x, y))

            self.ax.scatter(x, y, c="#f97316", s=100, marker="x", zorder=6)
            self.draw_idle()

            self.hazard_mode = False
            return
            
        # WALL PLACEMENT
        if self.wall_mode:
            self.wall_start = (event.xdata, event.ydata)
            from matplotlib.patches import Rectangle
            self.temp_wall_patch = Rectangle(
                (event.xdata, event.ydata), 0, 0, 
                facecolor='#3b82f6', alpha=0.5, edgecolor='#1e3a8a',
                linewidth=2, zorder=6
            )
            self.ax.add_patch(self.temp_wall_patch)
            return

        # PAN START
        if event.button == 1 and event.dblclick is False:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):

        if self.wall_mode and self.wall_start and event.xdata:
            x0, y0 = self.wall_start
            x1, y1 = event.xdata, event.ydata
            
            min_x, max_x = min(x0, x1), max(x0, x1)
            min_y, max_y = min(y0, y1), max(y0, y1)

            if (max_x - min_x) > 2 and (max_y - min_y) > 2:
                self.user_walls.append((min_x, min_y, max_x - min_x, max_y - min_y))
                from matplotlib.patches import Rectangle
                perm_wall = Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y, 
                    facecolor='#312e81', alpha=0.7, edgecolor='#06b6d4', 
                    linewidth=1, zorder=5
                )
                self.ax.add_patch(perm_wall)
                self.wall_patches.append(perm_wall)
                
            if self.temp_wall_patch:
                self.temp_wall_patch.remove()
            self.temp_wall_patch = None
            self.wall_start = None
            self.wall_mode = False
            self.draw_idle()
            return

        self.panning = False
        self.pan_start = None

    def on_mouse_move(self, event):
    
        if self.wall_mode and self.wall_start and self.temp_wall_patch and event.xdata:
            x0, y0 = self.wall_start
            w = event.xdata - x0
            h = event.ydata - y0
            
            self.temp_wall_patch.set_width(w)
            self.temp_wall_patch.set_height(h)
            self.draw_idle()
            return

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
        
    def clear_all_walls(self):
        for p in self.wall_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.wall_patches = []
        self.user_walls = []
        self.draw_idle()