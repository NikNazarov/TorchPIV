import logging
import traceback
import warnings
warnings.filterwarnings("ignore")
import sys
import gc
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QAction,
    QMenu,
    QHBoxLayout,
)
from torchPIV.PIVwidgets import PIVWidget
from torchPIV.ControlsWidgets import AnalysControlWidget
from torchPIV.PlotterFunctions import Database, show_message
from torchPIV.workers import PIVWorker, OnlineWorker
gc.disable() 

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piv_widget = PIVWidget()
        self.controls = AnalysControlWidget()
        self.data = Database()
        self.controls.piv_button.clicked.connect(self.start_piv)
        self.controls.pause_button.clicked.connect(self.pause_piv)
        self.controls.piv_button.clicked.connect(self.stop_piv)
        self.timer = QTimer()
        self.calc_thread = None
        self.worker = None
        self.timer.timeout.connect(self.piv_widget.piv_view.set_field)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.piv_widget)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.controls)
        h_layout.addWidget(self.piv_widget.controls)
        layout.addLayout(h_layout)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        
        menuBar = self.menuBar()
        exitAction = QAction("Exit", self)
        exitAction.triggered.connect(self.close)
        fileMenu = menuBar.addMenu("&File")
        settings = QAction("Analysis Settings", self)
        settings.triggered.connect(self.controls.show_settings)
        menuBar.addAction(settings)
        viewControls = QAction("View Settings", self)
        viewControls.triggered.connect(self.piv_widget.controls.show_settings)
        menuBar.addAction(viewControls)
        folderMenu = QMenu("PIV folder", self)
        selectFolder = QAction("&New", self)
        selectFolder.triggered.connect(self.controls.open_dialog)
        folderMenu.addAction(selectFolder)
        folderMenu.addAction(QAction(self.controls.settings.state.folder, self))
        pivfileMenu = QMenu("Open PIV file", self)
        pivfileAction = QAction("&New", self)
        pivfileAction.triggered.connect(self.piv_widget.controls.open_dialog)
        pivfileMenu.addAction(pivfileAction)

        
        saveMenu = QMenu("&Save", self)
        saveMenu.addActions([
            QAction("Save profile", self),
            QAction("Save colormap", self)
        ])

        videoMenu = QMenu("PIV Video File", self)
        selectVideo = QAction("&New", self)
        selectVideo.triggered.connect(self.controls.open_dialog_video)
        videoMenu.addAction(selectFolder)
        videoMenu.addAction(QAction(self.controls.settings.state.folder, self))

        fileMenu.addMenu(folderMenu)
        fileMenu.addMenu(videoMenu)
        fileMenu.addMenu(pivfileMenu)
        fileMenu.addMenu(saveMenu)
        fileMenu.addAction(exitAction)

    def exit(self, checked):
        self.controls.close()
        exit()
        
    def reportOutput(self, output: dict):
        self.data.set(output)
        if not self.piv_widget.controls.initialized:
            self.piv_widget.controls.set_field_box()
            self.piv_widget.controls.initialized = True
            self.piv_widget.piv_view.set_key("Vy[m/s]")
            self.piv_widget.controls.field_box.setCurrentText("Vy[m/s]")

    def reportProgress(self, value):
        self.controls.pbar.setValue(value)

    def reportFail(self):
        show_message(f'Dataset is empty')
        self.timer.stop()
        self.calc_thread.quit()
        self.calc_thread.wait()
        self.worker = None
        self.controls.piv_button.setText("Start PIV")
        gc.collect()

    def reportFinish(self, output: dict):
        if self.controls.settings.state.save_opt != "Dont save":
            show_message(
                f'Averaged data saved in\n{self.controls.settings.state.save_dir}'
                )
        self.data.set(output)
        self.timer.stop()
        self.piv_widget.controls.set_field_box()
        self.piv_widget.piv_view.set_field()
        self.calc_thread.quit()
        self.calc_thread.wait()
        self.calc_thread = None
        self.worker = None
        self.controls.piv_button.setText("Start PIV")
        gc.collect()
    
    def pause_piv(self):
        if self.calc_thread is None:
            return
        if self.worker.is_paused:
            text = "Pause"
        else:
            text = "Resume"

        self.controls.pause_button.setText(text)
        self.worker.is_paused = not self.worker.is_paused

    def stop_piv(self):
        if self.controls.piv_button.text() == "Stop PIV":
            return
        if self.calc_thread is None:
            return
        self.worker.is_running = False
        self.controls.pbar.setValue(0)


    def start_piv(self):
        if self.controls.piv_button.text() == "Start PIV":
            return
    
        self.piv_widget.controls.initialized = False

        self.timer.start(2000)
        self.controls.settings.state.to_json()
        self.calc_thread = QThread()
        piv_params = self.controls.settings.state
        if self.controls.settings.regime_box.currentText() == "offline":
            self.worker = PIVWorker(piv_params=piv_params)
        elif self.controls.settings.regime_box.currentText() == "online":
            self.worker = OnlineWorker(piv_params=piv_params)

        # self.worker.is_running = True
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.calc_thread)
        # Step 5: Connect signals and slots
        self._disable_buttons()
        self.calc_thread.finished.connect(
            self._enable_buttons
        )
        self.calc_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker.deleteLater)
        self.calc_thread.finished.connect(self.worker.deleteLater)
        self.calc_thread.finished.connect(self.calc_thread.deleteLater)
        self.worker.output.connect(self.reportOutput)
        self.worker.progress.connect(self.reportProgress)
        self.worker.finished.connect(self.reportFinish)
        self.worker.failed.connect(self.reportFail)
        # Step 6: Start the thread
        self.calc_thread.start()

        # Final resets
        
    def _disable_buttons(self):
        self.controls.settings.confirm.setEnabled(False)

    
    def _enable_buttons(self):
        self.controls.settings.confirm.setEnabled(True)

    def message(self, string: str):
        print(string)


# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)

def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message. 
    If unavailable (non-console application), log an additional notice.
    """
    #NOT IMPLEMENTED
    def onclick(button):
        if button.text() == "OK":
            QApplication.exit()
        elif button.text() == "Retry":
            pass


    if QApplication.instance() is not None:
            errorbox = QMessageBox()
            errorbox.setIcon(QMessageBox.Critical)
            errorbox.setText(f"Oops. An unexpected error occured:\n{log_msg}")
            errorbox.setStandardButtons(QMessageBox.Ok)
            errorbox.buttonClicked.connect(onclick)
            errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")


 
class UncaughtHook(QObject):
    _exception_caught = pyqtSignal(object)
 
    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)
 
    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        It is triggered each time an uncaught exception occurs. 
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)
 
# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()


def runGUI():
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    app.setFont(QFont("Helvetica", 12))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    runGUI()

