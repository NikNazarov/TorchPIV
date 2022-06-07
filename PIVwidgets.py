import matplotlib
import json
from scipy.interpolate import LinearNDInterpolator
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt, QLocale, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PlotterFunctions import Database, show_message, autoscale_y, make_name, save_table, set_width
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QFileDialog,
    QTextEdit,
    QWidget,
    QComboBox,
)


class PIVparams(object):
    """
    Naive Singletone realization 
    Shared object

    """
    wind_size: int = 0
    overlap: int = 0 
    scale: float = 0.0
    dt: float = 0.0
    device: str = ""
    iterations: int = 1
    resize: int = 0
    file_fmt: str = ""
    save_opt: str = "" 
    save_dir: str = "" 
    iter_scale: float = 2.0
    folder: str = ""

    @classmethod
    def __setattr__(cls, name, val):
        setattr(cls, name, val)

    @classmethod
    def __getattr__(cls, name):
        return getattr(cls, name)

    @classmethod
    def from_json(cls):
        """
        read json settings file
        convert to dict
        set attributes
        """
        with open("settings.json", 'r') as file:
            data = json.load(file)
            for key, val in data.items():
                if key not in dir(cls):
                    continue
                setattr(cls, key, val)
    @classmethod
    def to_json(cls):
        """
        Read attributes
        convert to dict
        save to json file
        """
        data = {
            name: getattr(cls, name) for name in dir(cls) 
            if not callable(getattr(cls, name)) and not name.startswith("__")
        }
        with open("settings.json", 'w') as file:
            json.dump(data, file)

class Settings(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.state = PIVparams()
        self.state.from_json()
        self.initUI()

    def initUI(self):
        hbox_top  = QHBoxLayout()
        hbox_bot  = QHBoxLayout()
        vbox_down = QVBoxLayout()
        main_box  = QVBoxLayout(self)

        self.file_fmt = QComboBox()
        self.file_fmt.addItems([
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jp",
            ".png",
            ".tiff", 
            ".tif"
            ])
        lbl = QLabel("File format")
        file_fmt_box = QVBoxLayout()
        file_fmt_box.addWidget(lbl)
        file_fmt_box.addWidget(self.file_fmt)

        self.wind_size = QLineEdit()
        self.wind_size.setText("64") 
        self.wind_size.setAlignment(Qt.AlignRight)
        self.wind_size.setValidator(QIntValidator(1, 256))
        lbl = QLabel("Window size")
        wind_size_box = QVBoxLayout()
        wind_size_box.addWidget(lbl)
        wind_size_box.addWidget(self.wind_size)

        self.overlap = QLineEdit()
        self.overlap.setText("32")
        self.overlap.setAlignment(Qt.AlignRight)
        self.overlap.setValidator(QIntValidator(1, 256))
        lbl = QLabel("Overlap")
        overlap_box = QVBoxLayout()
        overlap_box.addWidget(lbl)
        overlap_box.addWidget(self.overlap)

        self.piv_resize = QComboBox()
        self.piv_resize.addItems([
            "No rescale",
            "2",
            "3",
            "4",
            "5",
            "6"
            ])
        lbl = QLabel("Field rescale")
        piv_resize_box = QVBoxLayout()
        piv_resize_box.addWidget(lbl)
        piv_resize_box.addWidget(self.piv_resize)

        self.device = QComboBox()
        self.device.addItems([
            "cpu", 
            "cuda:0",
            ])
        lbl = QLabel("Device")
        device_box = QVBoxLayout()
        device_box.addWidget(lbl)
        device_box.addWidget(self.device)
        
        self.scale = QLineEdit()
        self.scale.setAlignment(Qt.AlignRight)
        self.scale.setText("0.0165364")
        validator = QDoubleValidator()
        validator.setLocale(QLocale("en_US"))
        self.scale.setValidator(validator)
        lbl = QLabel("scale mm/px")
        scale_box = QVBoxLayout()
        scale_box.addWidget(lbl)
        scale_box.addWidget(self.scale)

        self.time = QLineEdit()
        self.time.setText("11")
        self.time.setValidator(QIntValidator(1, 2147483647))
        self.time.setAlignment(Qt.AlignRight)
        lbl = QLabel("dt, us")
        time_box = QVBoxLayout()
        time_box.addWidget(lbl)
        time_box.addWidget(self.time)

        self.save_options = QComboBox()
        self.save_options.addItems([
            "Save statistics",
            "Save all",
            "Dont save",
        ])
        lbl = QLabel("Save options")
        save_box = QVBoxLayout()
        save_box.addWidget(lbl)
        save_box.addWidget(self.save_options)
        
        self.iteration_scale = QLineEdit()
        self.iteration_scale.setAlignment(Qt.AlignRight)
        self.iteration_scale.setText("2.0")
        validator = QDoubleValidator()
        validator.setLocale(QLocale("en_US"))
        self.iteration_scale.setValidator(validator)
        lbl = QLabel("Iter wind rescale")
        iter_scale_box = QVBoxLayout()
        iter_scale_box.addWidget(lbl)
        iter_scale_box.addWidget(self.iteration_scale)


        self.iteration = QLineEdit()
        self.iteration.setText("1")
        self.iteration.setValidator(QIntValidator(1, 10))
        self.iteration.setAlignment(Qt.AlignRight)
        lbl = QLabel("Iterations")
        iteration_box = QVBoxLayout()
        iteration_box.addWidget(lbl)
        iteration_box.addWidget(self.iteration)

        file_button = QPushButton("...")
        file_button.setFixedWidth(40)
        file_button.clicked.connect(self.open_dialog)
        lbl = QLabel()
        file_button_box = QVBoxLayout()
        file_button_box.addWidget(lbl)
        file_button_box.addWidget(file_button)

        self.save_folder = QTextEdit()
        self.save_folder.setFixedHeight(27)
        lbl = QLabel("Save directory")
        folder_box = QVBoxLayout()
        folder_box.addWidget(lbl)
        folder_box.addWidget(self.save_folder)

        self.confirm = QPushButton("Confirm")
        self.confirm.setFixedWidth(180)
        self.confirm.clicked.connect(self.confirm_changes)

        hbox_top.addLayout(wind_size_box)
        hbox_top.addLayout(overlap_box)
        hbox_top.addLayout(piv_resize_box)
        hbox_top.addLayout(file_fmt_box)
        hbox_bot.addLayout(scale_box)
        hbox_bot.addLayout(time_box)
        hbox_bot.addLayout(save_box)
        hbox_bot.addLayout(device_box)
        set_width(self, QLineEdit, 120)
        set_width(self, QComboBox, 180)
        set_width(self, QTextEdit, 180)

        hhbox = QHBoxLayout()
        
        hhbox.addLayout(iteration_box)
        hhbox.setAlignment(Qt.AlignRight)

        hhbox.addLayout(iter_scale_box)
        hhbox.setAlignment(Qt.AlignRight)



        hhbox.addLayout(folder_box)
        hhbox.addLayout(file_button_box)
        dummybox = QVBoxLayout()
        lbl = QLabel()
        self.confirm.setFixedWidth(140)
        dummybox.addWidget(lbl)
        dummybox.addWidget(self.confirm)
        dummybox.setAlignment(Qt.AlignRight)


        hhbox.addLayout(dummybox)
        hhbox.setAlignment(Qt.AlignRight)

    
        vbox_down.addLayout(hhbox)



        main_box.addLayout(hbox_top)
        main_box.addLayout(hbox_bot)
        main_box.addLayout(vbox_down)
        self.set_valeues()

    def set_valeues(self):
        
        self.wind_size.setText(str(self.state.wind_size))
        self.overlap.setText(str(self.state.overlap))
        idx = self.piv_resize.findText(str(self.state.resize), Qt.MatchContains)
        if idx >=0: self.piv_resize.setCurrentIndex(idx)
        idx = self.device.findText(self.state.device, Qt.MatchContains)
        if idx >= 0: self.device.setCurrentIndex(idx) 
        self.scale.setText(str(self.state.scale))
        self.time.setText(str(self.state.dt))
        idx = self.file_fmt.findText(self.state.file_fmt, Qt.MatchContains)
        if idx >= 0: self.file_fmt.setCurrentIndex(idx)
        idx = self.save_options.findText(self.state.save_opt, Qt.MatchContains)
        if idx >=0: self.save_options.setCurrentIndex(idx)
        self.save_folder.setText(self.state.save_dir)
        self.iteration.setText(str(self.state.iterations))
        self.iteration_scale.setText(str(self.state.iter_scale))
    
    
    def open_dialog(self, checked):
        folder = QFileDialog.getExistingDirectory()
        self.save_folder.setText(folder)

    def close(self) -> bool:
        self.set_valeues()
        return super().close()

    def confirm_changes(self, checked):
        
        self.state.wind_size = int(self.wind_size.text())
        self.state.overlap = int(self.overlap.text())
        if self.piv_resize.currentText().isdigit():
            self.state.resize = int(self.piv_resize.currentText())
        else:
            self.state.resize = 1
        self.state.device   = self.device.currentText()
        self.state.scale    = float(self.scale.text())
        self.state.dt       = int(self.time.text())
        self.state.file_fmt = self.file_fmt.currentText()
        self.state.save_opt = self.save_options.currentText()
        self.state.save_dir = self.save_folder.toPlainText()
        self.state.iterations = int(self.iteration.text())
        self.state.iter_scale = float(self.iteration_scale.text())
        self.state.to_json()
        if self.isVisible():
            self.hide()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,  parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
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


class ProfileCanvas(MplCanvas):
    def __init__(self):
        super().__init__()
        self.axes.autoscale(True)
        self.axes.grid(linestyle = '--', linewidth = 0.7)

    def set_field(self, key):
        piv_data = self.data.get()
        self.field = piv_data[key]
        self.key = key
        x, y =  [*piv_data.values()][:2]
        self.x_data = x[0]
        self.y_data = y[:, 0]

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
    topChanged  = pyqtSignal(float)
    botChanged  = pyqtSignal(float)
    lineChanged = pyqtSignal(float)
    def __init__(self):
        super().__init__()
        self.coords      = None
        self.cb          = None
        self.streamlines = None
        self.pos_scale   = 1.
        self.neg_scale   = 1.
        self.visible_lines = False
        self.visible_line  = True
        self.img_data      = None
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

    def set_coords(self):
        if self.coords is not None:
            return
        
        piv_data = self.data.get()
        x, y = [*piv_data.values()][:2]
        self.coords = x, y
        self.x_data = x[0]
        self.y_data = y[:, 0]
    
    def set_field(self, key):
        self.set_coords()
        self.key = key
        piv_data = self.data.get()
        field = piv_data[key]
        self.pos_avg = np.max(np.abs(field), initial=0)
        self.neg_avg = -self.pos_avg
        if isinstance(self.cb, matplotlib.colorbar.Colorbar):
            self.cb.remove()
        if isinstance(self.img_data, matplotlib.collections.QuadMesh):
            self.img_data.remove()
        self.img_data = self.axes.pcolormesh(*self.coords, 
                                            field, 
                                            cmap="jet", 
                                            shading='auto',
                                            vmin=self.neg_avg*self.neg_scale,
                                            vmax=self.pos_avg*self.pos_scale,
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
        self.topChanged.emit(value*self.pos_avg)
        self.set_field(self.key)

    def set_v_min(self, value):
        if self.img_data is None:
            return
        value = (1000-value)/1000
        if value*self.neg_avg >= self.pos_scale*self.pos_avg:
            return
        self.neg_scale = value
        self.botChanged.emit(value*self.neg_avg)
        self.set_field(self.key)

    def draw_stremlines(self):
        piv_data = self.data.get()
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
        if self.coords is None:
            return
        if self.streamlines is None:
            self.draw_stremlines()
        self.visible_lines = not self.visible_lines
        self.streamlines.lines.set_visible(self.visible_lines)
        self.streamlines.arrows.set_visible(self.visible_lines)
        self.update_canvas()



class PIVWidget(QWidget):
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
        
        profile_box = QVBoxLayout()
        profile_box.addWidget(profile_toolbar)
        profile_box.addWidget(self.profile)

        layout = QHBoxLayout()
        layout.addLayout(piv_box)
        layout.addLayout(profile_box)

        self.setLayout(layout)
        
class ControlsWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.folder_name = QTextEdit()
        self.settings = Settings()
        self.folder_name.setText(self.settings.state.folder)
        self.regime_box = QComboBox()
        self.regime_box.addItems(["offline", "online"])
        self.piv_button = QPushButton("Start PIV")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pbar = QProgressBar()

        self.initUI()
    def initUI(self):
        file_box = QHBoxLayout()
        file_button = QPushButton("Select folder")
        file_button.clicked.connect(self.open_dialog)
        self.folder_name.setFixedHeight(35)
        file_box.addWidget(self.folder_name)
        file_box.addWidget(file_button)

        control_v = QVBoxLayout()

        control_v.addLayout(file_box)

        settings_h = QHBoxLayout()

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)

        
        settings_h.addWidget(settings_button)
        settings_h.addSpacing(30)
        settings_h.addWidget(self.regime_box)
        settings_h.addSpacing(30)
        settings_h.addWidget(self.piv_button)

        control_v.addLayout(settings_h)

        bottom_left_frame = QFrame()
        bottom_left_frame.setLayout(control_v)
        bottom_left_frame.setLineWidth(1)
        bottom_left_frame.setFrameStyle(QFrame.Panel)

        bottom_left_box = QHBoxLayout()
        bottom_left_box.addWidget(bottom_left_frame)
        bottom_left_box.addSpacing(100)

        progress_box = QVBoxLayout()
        lbl = QLabel("Total progress: ")
        progress_box.addWidget(lbl)
        progress_box.addWidget(self.pbar)
        pb_h = QHBoxLayout()
        pb_h.addWidget(self.pause_button)
        pb_h.addWidget(self.stop_button)
        progress_box.addLayout(pb_h)

        bottom_right_frame = QFrame()
        bottom_right_frame.setFrameStyle(QFrame.Raised)
        bottom_left_frame.setLineWidth(1)

        main_box = QHBoxLayout()
        main_box.addLayout(bottom_left_box)
        main_box.addLayout(progress_box)

        self.setLayout(main_box)
        
    def open_dialog(self, checked):
        folder = QFileDialog.getExistingDirectory()
        self.folder_name.setText(folder)
        self.settings.state.folder = folder

    def show_settings(self, checked):
        if self.settings.isVisible():
            self.settings.hide()
        else:
            self.settings.show()
