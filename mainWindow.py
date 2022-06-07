import logging
import traceback
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QMessageBox
)
from PIVwidgets import PIVWidget, ControlsWidget
from PlotterFunctions import show_message
from workers import PIVWorker, OnlineWorker


 

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piv_widget = PIVWidget()
        self.controls = ControlsWidget()
        self.controls.piv_button.clicked.connect(self.start_piv)
        self.controls.pause_button.clicked.connect(self.pause_piv)
        self.controls.stop_button.clicked.connect(self.stop_piv)
        self.timer = QTimer()
        self.calc_thread = None
        self.timer.timeout.connect(self.piv_widget.piv.update_canvas)
        self.initUI()
   
    def initUI(self):

        layout = QVBoxLayout()
        layout.addWidget(self.piv_widget)
        layout.addWidget(self.controls)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

    def exit(self, checked):
        self.controls.close()
        exit()
        
    def reportOutput(self, output):
        x, y, u, v = output
        self.piv_widget.piv.set_coords(x, y)
        self.piv_widget.piv.set_quiver(u, v)

    def reportProgress(self, value):
        self.controls.pbar.setValue(value)

    def reportFinish(self, output):
        show_message(
            f'Averaged data saved in\n{self.controls.settings.state.save_dir}'
            )
        x, y, u, v = output
        self.piv_widget.piv.new_image = True
        self.timer.stop()
        self.piv_widget.piv.set_coords(x, y)
        self.piv_widget.piv.set_quiver(u, v)
        self.piv_widget.piv.draw_stremlines()
        self.piv_widget.piv.update_canvas()
        self.piv_widget.piv.restore()


        

    
    def pause_piv(self):
        if self.calc_thread is None:
            return
        if self.worker.is_paused:
            text = "Pause"
            self.piv_widget.piv.restore()
        else:
            text = "Resume"

        self.controls.pause_button.setText(text)
        self.worker.is_paused = not self.worker.is_paused

        # if self.worker.is_paused:
        #     u, v = self.worker.avg_u/self.worker.idx, self.worker.avg_v/self.worker.idx
        #     self.piv_widget.piv.set_quiver(u, v)
        #     self.piv_widget.piv.draw_stremlines()
        #     self.piv_widget.piv.update_canvas()
    
    def stop_piv(self):
        if self.calc_thread is None:
            return
        self.worker.is_running = False
        self.controls.pbar.setValue(0)


    def start_piv(self):
        self.timer.start(4000)
        self.piv_widget.piv.restore()
        self.controls.settings.state.to_json()
        self.calc_thread = QThread(parent=None)
        piv_params = self.controls.settings.state
        if self.controls.regime_box.currentText() == "offline":
            self.worker = PIVWorker(self.controls.folder_name.toPlainText(),
                                    piv_params=piv_params)
        elif self.controls.regime_box.currentText() == "online":
            self.worker = OnlineWorker(self.controls.folder_name.toPlainText(),
                                        piv_params=piv_params)

        # self.worker.is_running = True
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.calc_thread)
        # Step 5: Connect signals and slots
        self.calc_thread.started.connect(self.worker.run)
        self.worker.signals.finished.connect(self.calc_thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.calc_thread.finished.connect(self.calc_thread.deleteLater)
        self.worker.signals.output.connect(self.reportOutput)
        self.worker.signals.progress.connect(self.reportProgress)
        self.worker.signals.finished.connect(self.reportFinish)
        # Step 6: Start the thread
        self.calc_thread.start()

        # Final resets
        self._disable_buttons()
        self.calc_thread.finished.connect(
            self._enable_buttons
        )
    def _disable_buttons(self):
        self.controls.piv_button.setEnabled(False)
        self.controls.settings.confirm.setEnabled(False)

    
    def _enable_buttons(self):
        self.controls.piv_button.setEnabled(True)
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



if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    app.setFont(QFont("Helvetica", 12))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
    