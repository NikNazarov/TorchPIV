import matplotlib
import json
from scipy.interpolate import LinearNDInterpolator
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt, QLocale
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
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
    QMessageBox
)
def show_message(message:str) -> None:
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setText(message)
    msgbox.setStandardButtons(QMessageBox.Ok)
    msgbox.buttonClicked.connect(msgbox.close)
    msgbox.exec_()

def set_width(obj: object, target_type: type, width: int):
    '''
    Helper function to set size of object's inner widgets
    '''
    if "setFixedWidth" in dir(target_type):
        for key, val in obj.__dict__.items():
            if isinstance(val, target_type):
                obj.__dict__[key].setFixedWidth(width)

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


        
class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop _process anything it has to do
        cv.flush_events()

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,  parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class ProfileCanvas(MplCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        super().__init__(parent=parent, width=width, height=height, dpi=dpi)
        self.hor_axes  = self.fig.add_subplot(2, 1, 1)
        self.hor_line  = None
        self.vert_axes = self.fig.add_subplot(2, 1, 2)
        self.vert_line = None
        self.x = np.linspace(0, 1000, 10)
        self.fig.tight_layout()


    def onclick(self, event):
        if event.button:
            self.vert_axes.cla()
            self.hor_axes.cla()
            self.vert_line, = self.vert_axes.plot(self.x, np.ones_like(self.x)*event.xdata, 
                                            linewidth=.7, color="blue")
            self.hor_line, = self.hor_axes.plot(self.x, np.ones_like(self.x)*event.ydata, 
                                            linewidth=.7, color="blue")
            self.fig.canvas.draw()


class PIVcanvas(MplCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.axes        = self.fig.add_subplot(1, 1, 1)
        self.circle      = None
        self.vert_line   = None
        self.hor_line    = None
        self.coords      = None
        self.new_image   = True
        self.quiver_data = None
        self.quivers     = None
        self.cb          = None
        self.scale_max   = None
        self.scale_avg   = None
        self.stramlines  = None

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.img = np.zeros((1000, 1000))*np.nan
        self.img_data = self.axes.pcolormesh(self.img)
        self.y = np.linspace(0, self.img.shape[-2], 10)
        self.x = np.linspace(0, self.img.shape[-1], 10)
        self.axes.axis("off")
        self.vert_line_data = np.ones_like(self.y)*0, self.y
        self.hor_line_data = self.x, np.ones_like(self.x)*0
        self.fig.tight_layout()


        
    def onclick(self, event):
        if event.button:
            if self.circle is None:
                self.circle = plt.Circle((event.xdata, event.ydata), 12, fill=False, color='blue')
                self.axes.add_patch(self.circle)
                self.vert_line, = self.axes.plot(np.ones_like(self.x)*event.xdata, 
                                                self.y, linewidth=.7, color="red")
                self.hor_line, = self.axes.plot(self.x, np.ones_like(self.y)*event.ydata, 
                                                linewidth=.7, color="red")

            self.vert_line.set_xdata(np.ones_like(self.x)*event.xdata)
            self.hor_line.set_ydata(np.ones_like(self.y)*event.ydata)

            self.circle.center = event.xdata, event.ydata
            self.fig.canvas.draw()
        
    def set_coords(self, x, y):
        if self.coords is None:
            self.coords = x, y  
    
    def set_quiver(self, U ,V):
        if not self.coords:
            return
        self.quiver_data = U, V
        
        mod_V = np.hypot(U, V)
        if self.scale_max is None:
            self.scale_max = np.max(mod_V)
            self.scale_avg = np.average(mod_V)
        
        Uq =  U/self.scale_max
        Vq =  V/self.scale_max
        
        if self.quivers is None:
            if isinstance(self.cb, matplotlib.colorbar.Colorbar):
                self.cb.remove()
            if self.stramlines is not None:
                self.stramlines.lines.remove()
                self.stramlines.arrows.remove()
            self.img_data = self.axes.pcolormesh(*self.coords, 
                                                mod_V, 
                                                cmap=plt.get_cmap('jet'), 
                                                shading='auto', vmax=self.scale_avg*3)
            self.quivers = self.axes.quiver(*self.coords, Uq, Vq, scale_units="xy", scale=.01, pivot="mid", width=0.002)
            self.x = np.linspace(0, np.max(self.coords[0]), 10)
            self.y = np.linspace(0, np.max(self.coords[1]), 10)
            self.cb = self.fig.colorbar(self.img_data, ax=self.axes)
            self.circle = None

        self.quivers.set_UVC(Uq, Vq)
        self.img_data.set_array(mod_V.ravel())

    def restore(self):
        self.coords = None
        self.quivers = None
        
    def draw_stremlines(self):
        u, v = self.quiver_data
        x, y = self.coords
        x0 = x[0]
        y0 = y[:, 0]
        xi = np.linspace(x0.min(), x0.max(), x0.size)
        yi = np.linspace(y0.min(), y0.max(), y0.size)
        xflat = x.reshape(-1)
        yflat = y.reshape(-1)
        uflat = u.reshape(-1)
        vflat = v.reshape(-1)
        interp_ui = LinearNDInterpolator(list(zip(xflat, yflat)), uflat)
        interp_vi = LinearNDInterpolator(list(zip(xflat, yflat)), vflat)
        xi, yi = np.meshgrid(xi, yi)
        ui = interp_ui(xi, yi)
        vi = interp_vi(xi, yi)
        self.streamlines = self.axes.streamplot(xi, yi, ui, vi, 
            density=5, linewidth=.5, arrowsize=.5
            )

    def update_canvas(self):
        self.fig.canvas.draw()



class PIVWidget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.piv     = PIVcanvas(self, width=6, height=6, dpi=200)
        self.profile = ProfileCanvas()
        # self.piv.fig.canvas.mpl_connect('button_press_event', self.profile.onclick)
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
