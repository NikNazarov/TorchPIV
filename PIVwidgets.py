import matplotlib
from scipy.interpolate import LinearNDInterpolator
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PlotterFunctions import Database, show_message, autoscale_y, make_name, save_table
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QSplitter
)
from ControlsWidgets import ProfileControls

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,  parent=None, width=7, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height),dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.axes  = self.fig.add_subplot(1, 1, 1)
        self.key   = None
        self.line  = None
        self.field = None
        self.fig.tight_layout()
        self.data  = Database()
        self.x_data = None
        self.y_data = None
        self.orientation = True #True for horizontal and False for vertical

    def change_orientation(self, key):
        self.orientation = not self.orientation
        if self.line is None:
            return
        self.line.remove()
        self.line = None
    
    def update_canvas(self):
        self.fig.canvas.draw()
    def clear(self):
        self.axes.cla()

class ProfileCanvas(MplCanvas):
    def __init__(self):
        super().__init__()
        self.axes.autoscale(True)
        self.axes.grid(linestyle = '--', linewidth = 0.7)
        self.value: int = 0
        self.key = None

    def set_key(self, key: str):
        self.key = key
        self.set_field()

    def set_field(self):
        piv_data = self.data.get()
        if not piv_data:
            return
        self.field = piv_data[self.key]
        x, y =  [*piv_data.values()][:2]
        self.x_data = x[0]
        self.y_data = y[:, 0]
        if isinstance(self.line, matplotlib.lines.Line2D):
            self.line.remove()
            self.line = None
        self.draw_line(self.value)

    def draw_horizontal(self, value):
        if self.line is None:
            self.line, = self.axes.plot(self.x_data, self.field[value,:], 
                    linewidth=.8, color="red", marker='s')
        self.line.set_ydata(self.field[value,:])

    def draw_vertical(self, value):
        if self.line is None:
            self.line, = self.axes.plot(self.y_data, self.field[:,value], 
                    linewidth=.8, color="red", marker='s')
        self.line.set_ydata(self.field[:,value])

    def draw_line(self, value):
        self.value = value
        if self.field is None:
            return
        if self.orientation:
            self.draw_horizontal(value)
        else:
            self.draw_vertical(value)
        autoscale_y(self.axes)
        self.update_canvas()
    
    def save_profile(self):
        if self.line is None:
            return
        x_data = self.line.get_xdata()
        y_data = self.line.get_ydata()
        data = {
            "x[mm]" if self.orientation else "y[mm]": x_data,
            self.key: y_data
        }
        filename, save_dir = make_name(self.data.name, self.key, self.orientation)
        save_table(filename, save_dir, data)
        show_message(f"Profile{self.key} saved in \n{save_dir}")


class PIVcanvas(MplCanvas):
    topChanged  = pyqtSignal(str)
    botChanged  = pyqtSignal(str)
    lineChanged = pyqtSignal(float)
    def __init__(self):
        super().__init__()
        self.cb          = None
        self.streamlines = None
        self.pos_scale   = .5
        self.neg_scale   = .5
        self.visible_line  = True
        self.img_data      = None
        self.key           = None
        self.averaged_data = None
        self.data_index    = 0
        self.axes.axis("off")


    def draw_horizontal(self, value):
        if self.line is None:
            self.line, = self.axes.plot(self.x_data, np.ones_like(self.x_data)*self.y_data[value], 
                            linewidth=1.5, color="white")
        self.line.set_ydata(np.ones_like(self.x_data)*self.y_data[value])
        self.lineChanged.emit(self.y_data[value])

    def draw_vertical(self, value):
        if self.line is None:
            self.line, = self.axes.plot(np.ones_like(self.y_data)*self.x_data[value], 
            self.y_data, linewidth=1.5, color="white")
        self.line.set_xdata(np.ones_like(self.y_data)*self.x_data[value])
        self.lineChanged.emit(self.x_data[value])
    

    def draw_line(self, value):
        if isinstance(self.line, matplotlib.lines.Line2D):
            self.line.remove()
            self.line = None
        if self.x_data is None:
            return
        if self.orientation:
            self.draw_horizontal(value)
        else:
            self.draw_vertical(value)
        self.update_canvas()

    def hide_line(self):
        if self.line is None:
            return
        self.visible_line = not self.visible_line
        self.line.set_visible(self.visible_line)
        self.update_canvas()

    def set_key(self, key: str):
        self.key = key
        self.set_field()
    
    def set_field(self):
        piv_data = self.data.get()
        if not piv_data:
            return
        x, y = [*piv_data.values()][:2]
        self.x_data = x[0]
        self.y_data = y[:, 0]
        field = piv_data[self.key]
        self.pos_avg = np.max(np.abs(field), initial=0)
        self.neg_avg = -self.pos_avg

        if isinstance(self.cb, matplotlib.colorbar.Colorbar):
            self.cb.remove()
        if isinstance(self.img_data, matplotlib.collections.QuadMesh):
            self.img_data.remove()
        self.img_data = self.axes.pcolormesh(x, y, 
                                            field, 
                                            cmap="jet", 
                                            shading='auto',
                                            vmin=None,#self.neg_avg*self.neg_scale,
                                            vmax=None #self.pos_avg*self.pos_scale,
                                            )
        self.cb = self.fig.colorbar(self.img_data, ax=self.axes)
        self.update_canvas()

    def set_v_max(self, value):
        if self.img_data is None:
            return
        value = (value-1000)/1000
        if value*self.pos_avg <= self.neg_avg*self.neg_scale:
            return
        self.pos_scale = value
        self.topChanged.emit(f"{value*self.pos_avg:.2f}")
        self.set_field()

    def set_v_min(self, value):
        if self.img_data is None:
            return
        value = (1000-value)/1000
        if value*self.neg_avg >= self.pos_scale*self.pos_avg:
            return
        self.neg_scale = value
        self.botChanged.emit(f"{value*self.neg_avg:.2f}")
        self.set_field()

    def draw_streamlines(self):
        piv_data = self.data.get()
        if not piv_data:
            return
        u, v = [*piv_data.values()][2:4]
        x0 = self.x_data
        y0 = self.y_data
        xi = np.linspace(x0.min(), x0.max(), y0.size)
        yi = np.linspace(y0.min(), y0.max(), x0.size)
        x0, y0 = np.meshgrid(x0, y0)
        xi, yi = np.meshgrid(xi, yi)
        xflat = x0.reshape(-1)
        yflat = y0.reshape(-1)
        interp_ui = LinearNDInterpolator(list(zip(xflat, yflat)), u.reshape(-1))
        interp_vi = LinearNDInterpolator(list(zip(xflat, yflat)), v.reshape(-1))
        ui = interp_ui(xi, yi)
        vi = interp_vi(xi, yi)
        self.streamlines = self.axes.streamplot(xi, yi, ui, vi, 
            density=4, linewidth=.8, arrowsize=.8, color="black"
            )
        self.update_canvas()

    def hide_streamlines(self):
        self.clear()
        self.set_field()
        self.line = None
        self.update_canvas()

    def show_grid(self):
        self.axes.grid(True, linestyle = '--', linewidth = 0.7)
        self.update_canvas()
    def show_axis(self):
        self.axes.axis("on")
        self.update_canvas()
    def hide_grid(self):
        self.axes.grid(False)
        self.update_canvas()
    def hide_axis(self):
        self.axes.axis("off")
        self.update_canvas()



class PIVview(QSplitter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.piv     = PIVcanvas()
        self.profile = ProfileCanvas()
        self.initUI()


    def initUI(self):
        piv_toolbar     = NavigationToolbar(self.piv, self)
        profile_toolbar = NavigationToolbar(self.profile, self)
        piv_box = QVBoxLayout()
        piv_box.addWidget(piv_toolbar)
        piv_box.addWidget(self.piv)
        dummyPiv = QWidget()
        dummyPiv.setLayout(piv_box)
        
        profile_box = QVBoxLayout()
        profile_box.addWidget(profile_toolbar)
        profile_box.addWidget(self.profile)
        dummyProfile = QWidget()
        dummyProfile.setLayout(profile_box)

        self.addWidget(dummyPiv)
        self.addWidget(dummyProfile)

        # self.setLayout(layout)
    def set_key(self, key):
        self.piv.set_key(key)
        self.profile.set_key(key)
    def set_field(self):
        self.piv.set_field()
        self.profile.set_field()


        
class PIVWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piv_view = PIVview()
        self.controls = ProfileControls()
        self.controls.hide_line_box.stateChanged.connect(self.piv_view.piv.hide_line)
        self.controls.field_box.activated[str].connect(self.piv_view.set_key)
        self.controls.slider.valueChanged.connect(self.piv_view.piv.draw_line)
        self.controls.slider.valueChanged.connect(self.piv_view.profile.draw_line)
        self.controls.streamlines_box.stateChanged.connect(self.streamlines_checker)
        self.controls.grid_box.stateChanged.connect(self.grid_checker)
        self.controls.axes_box.stateChanged.connect(self.axes_checker)
        self.controls.orientation_qbox.activated[str].connect(self.piv_view.profile.change_orientation)
        self.controls.orientation_qbox.activated[str].connect(self.piv_view.piv.change_orientation)
        self.piv_view.piv.lineChanged.connect(self.controls.slider_LCD.display)
        self.controls.settings.pos_scale_slider.valueChanged.connect(self.piv_view.piv.set_v_max)
        self.controls.settings.neg_scale_slider.valueChanged.connect(self.piv_view.piv.set_v_min)
        self.piv_view.piv.topChanged.connect(self.controls.settings.pos_scale_text.setText)
        self.piv_view.piv.botChanged.connect(self.controls.settings.neg_scale_text.setText)


        self.initUI()
   
    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.piv_view)
        self.setLayout(layout)

    def streamlines_checker(self):
        if self.controls.streamlines_box.isChecked():
            self.piv_view.piv.draw_streamlines()
        else:
            self.piv_view.piv.hide_streamlines()
    def grid_checker(self):
        if self.controls.grid_box.isChecked():
            self.piv_view.piv.show_grid()
        else:
            self.piv_view.piv.hide_grid()
    def axes_checker(self):
        if self.controls.axes_box.isChecked():
            self.piv_view.piv.show_axis()
        else:
            self.piv_view.piv.hide_axis()
