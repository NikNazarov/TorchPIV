import json
import matplotlib.pyplot as plt
plt.ioff()
import torch
from bisect import bisect_left
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt, QLocale, pyqtSignal, pyqtSlot  
from PlotterFunctions import Database, PIVparams, set_width
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
    QLCDNumber,
    QSlider,
    QCheckBox,
)


class ListSlider(QSlider):
    elementChanged = pyqtSignal(int)

    def __init__(self, values=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(0)
        self._values = []
        self.valueChanged.connect(self._on_value_changed)
        self.values = values or []

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values: list):
        self._values = values
        maximum = max(0, len(self._values) - 1)
        self.setMaximum(maximum)
        self.setValue(0)

    @pyqtSlot(int)
    def _on_value_changed(self, index: int):
        value = self.values[index]
        self.elementChanged.emit(value)
    def set_value(self, value: str):
        index = bisect_left(self._values, float(value))
        self.setValue(index)
        return index



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
        available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        self.device.addItems(["cpu"]+available_gpus)
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

        regimebox = QVBoxLayout()
        lbl = QLabel("PIV regime")
        self.regime_box = QComboBox()
        self.regime_box.addItems([
            "offline",
            "online"
        ])
        regimebox.addWidget(lbl)
        regimebox.addWidget(self.regime_box)

        
        hhhbox = QHBoxLayout()
        hhhbox.addLayout(regimebox)
        hhhbox.addLayout(dummybox)    
    
        vbox_down.addLayout(hhbox)
        vbox_down.addLayout(hhhbox)


        main_box.addLayout(hbox_top)
        main_box.addLayout(hbox_bot)
        main_box.addLayout(vbox_down)
        set_width(self, QLineEdit, 120)
        set_width(self, QComboBox, 180)
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
        idx = self.regime_box.findText(str(self.state.regime), Qt.MatchContains)
        if idx >=0: self.regime_box.setCurrentIndex(idx)
    
    
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
        self.state.regime     = self.regime_box.currentText()
        self.state.to_json()
        if self.isVisible():
            self.hide()

class ViewSettings(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.initUI()

    def initUI(self):
        self.pos_scale_slider = ListSlider(orientation=Qt.Horizontal)
        self.neg_scale_slider = ListSlider(orientation=Qt.Horizontal)
        self.pos_scale_slider.setFixedWidth(200)
        self.neg_scale_slider.setFixedWidth(200)
        self.pos_scale_slider.values = list(range(2000))
        self.pos_scale_slider.setValue(1499)
        self.neg_scale_slider.values = list(range(2000))
        self.neg_scale_slider.setValue(499)
        
        self.pos_scale_text = QLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale("en_US"))
        self.pos_scale_text.setValidator(validator)
        self.neg_scale_text = QLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale("en_US"))
        self.neg_scale_text.setValidator(validator)

        self.pos_scale_text.editingFinished.connect(self.on_posLineEditChanged)
        self.neg_scale_text.editingFinished.connect(self.on_negLineEditChanged)
        
        self.quiver_box = QCheckBox("Show Quivers")
        self.streamlines_box = QCheckBox("Show Streamlines")
        self.hide_line_box = QCheckBox("Show Profile Line")
        self.hide_line_box.toggle()
        self.axes_box = QCheckBox("Show Axes")
        self.grid_box = QCheckBox("Show Grid")


        pos_hbox = QHBoxLayout()
        pos_hbox.addWidget(self.pos_scale_slider)
        pos_hbox.addWidget(self.pos_scale_text)
        neg_hbox = QHBoxLayout()
        neg_hbox.addWidget(self.neg_scale_slider)
        neg_hbox.addWidget(self.neg_scale_text)
        
        checkbox_vbox = QVBoxLayout()
        checkbox_vbox.addWidget(self.streamlines_box)
        checkbox_vbox.addWidget(self.hide_line_box)
        checkbox_vbox.addWidget(self.quiver_box)
        checkbox_vbox.addWidget(self.axes_box)
        checkbox_vbox.addWidget(self.grid_box)



        layout = QVBoxLayout()
        layout.addLayout(pos_hbox)
        layout.addLayout(neg_hbox)
        layout.addLayout(checkbox_vbox)

        self.setLayout(layout)
    def on_posLineEditChanged(self):
        self.pos_scale_slider.set_value(self.pos_scale_text.text())
    def on_negLineEditChanged(self):
        self.neg_scale_slider.set_value(self.neg_scale_text.text())

class ProfileControls(QWidget):
    fieldchosen = pyqtSignal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = ViewSettings()
        self.field_box = QComboBox()
        self.data = Database()
        self.initialized: bool = False
        self.setFixedHeight(120)
        self.hide_line_box = self.settings.hide_line_box
        self.streamlines_box = self.settings.streamlines_box
        self.grid_box = self.settings.grid_box
        self.axes_box = self.settings.axes_box



        self.initUI()
    def initUI(self):
        self.slider_LCD = QLCDNumber()
        self.slider_LCD.setFrameShape(QFrame.NoFrame)
        self.slider_LCD.setSegmentStyle(QLCDNumber.Flat)

        self.slider = ListSlider(orientation=Qt.Horizontal)
        self.orientation_qbox = QComboBox()
        self.orientation_qbox.setFixedWidth(100)
        slider_box = QHBoxLayout()
        slider_box.addWidget(self.slider)
        slider_box.addWidget(self.slider_LCD)
        slider_box.addWidget(self.orientation_qbox)
        bottom_right_box = QVBoxLayout()
        lbl = QLabel("Profile control:")
        bottom_right_box.addWidget(lbl)
        bottom_right_box.addLayout(slider_box)
        bottom_right_box.addWidget(self.field_box)
        
        bottom_right_frame = QFrame()
        bottom_right_frame.setLayout(bottom_right_box)
        bottom_right_frame.setLineWidth(1)
        bottom_right_frame.setFrameStyle(QFrame.Panel)

        main_box = QHBoxLayout()
        main_box.addWidget(bottom_right_frame)

        self.setLayout(main_box)
    

    @pyqtSlot(str)
    def on_activated(self, key):
        if key is None:
            return
        self.fieldchosen.emit(key)

    @pyqtSlot(str)
    def on_orientation(self, key):
        piv_data = self.data.get()
        if key == "Horizontal":
            self.slider.values = piv_data[[*piv_data.keys()][1]][:, 0]
        else:
            self.slider.values = piv_data[[*piv_data.keys()][0]][0]

        self.fieldchosen.emit(key)
        
    def open_dialog(self):
        name, check = QFileDialog.getOpenFileName()
        if not check:
            return
        self.data.load(name)
        self.set_field_box()
    
    def init_profile(self):
        self.field_box.activated[str].connect(self.on_activated)
        self.orientation_qbox.clear()
        self.orientation_qbox.addItems(["Horizontal", "Vertical"])
        self.orientation_qbox.activated[str].connect(self.on_orientation)


    def set_field_box(self):
        piv_data = self.data.get()
        self.field_box.clear()
        self.field_box.addItems([*piv_data.keys()][2:])
        self.field_box.setCurrentText("Vy[m/s]")
        self.slider.values = piv_data[[*piv_data.keys()][1]][:, 0]
        self.slider.setValue(0)
        self.init_profile()

    def show_settings(self, checked):
        if self.settings.isVisible():
            self.settings.hide()
        else:
            self.settings.show()

class AnalysControlWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = Settings()
        self.piv_button = QPushButton("Start PIV")
        self.pause_button = QPushButton("Pause")
        self.pbar = QProgressBar()
        self.piv_button.pressed.connect(self._changeButton)
        self.initUI()
    def initUI(self):
        # self.pbar.setStyleSheet("QProgressBar::chunk "
        #           "{"
        #             "background-color: green;"
        #             "border-style: outset;"
        #             "border-width: 2px;"
        #             "border-radius: 10px;"
        #             "border-color: beige;"
        #             "font: bold 14px;"
        #             "min-width: 10em;"
        #             "padding: 6px;"
        #           "}")

        progress_box = QVBoxLayout()
        lbl = QLabel("Total progress: ")
        progress_box.addWidget(lbl)
        progress_box.addWidget(self.pbar)
        pb_h = QHBoxLayout()
        pb_h.addWidget(self.pause_button)
        pb_h.addWidget(self.piv_button)
        progress_box.addLayout(pb_h)


        self.setLayout(progress_box)
        self.setFixedHeight(120)
        
    def open_dialog(self, checked):
        folder = QFileDialog.getExistingDirectory()
        self.settings.state.folder = folder
    
    def _changeButton(self):
        if self.piv_button.text() == "Start PIV":
            self.piv_button.setText("Stop PIV")
        else:
            self.piv_button.setText("Start PIV")

    def show_settings(self, checked):
        if self.settings.isVisible():
            self.settings.hide()
        else:
            self.settings.show()