import os
import sys
import time
from xxlimited import new
import numpy as np
from collections import deque
from PlotterFunctions import PIVparams, natural_keys
from PyQt5.QtCore import QObject, pyqtSignal, QProcess
from PlotterFunctions import save_table
from torchPIV import OfflinePIV, free_cuda_memory



class WorkerSignals(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    output   = pyqtSignal(dict)

class PIVWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, folder, piv_params: PIVparams, *args, **kwargs) -> None:
        super().__init__(*args, parent=None, **kwargs)
        self.folder     = folder
        self.piv_params = piv_params
        self.avg_u      = None
        self.avg_v      = None
        self.is_paused  = False
        self.is_running = True
        self.idx        = 0

    def run(self):
        """Long PIV task.
        Emits: 
        x_grid, y_grid, U, V: np.ndarray
        """
        piv_gen = OfflinePIV(
            folder=self.folder,
            device=self.piv_params.device,
            file_fmt=self.piv_params.file_fmt,
            wind_size=self.piv_params.wind_size,
            overlap=self.piv_params.overlap,
            resize=self.piv_params.resize,
            iterations=self.piv_params.iterations,
            iter_scale=self.piv_params.iter_scale
        )
        u_inst = []
        v_inst = [] 
        start = time.time()
        for i, out in enumerate(piv_gen()):
            while self.is_paused and self.is_running: 
                time.sleep(0)
            if not self.is_running:
                break

            x, y, u, v = out
            x = x * self.piv_params.scale
            y = y * self.piv_params.scale
            u = u * self.piv_params.scale*1e3/self.piv_params.dt
            v = v * self.piv_params.scale*1e3/self.piv_params.dt

            u_inst.append(u.astype(np.float64))
            v_inst.append(v.astype(np.float64))
            self.signals.progress.emit((i + 1)/len(piv_gen)*100)
            self.signals.output.emit({
            "x[mm]": x,
            "y[mm]": y,
            "Vx[m/s]": u,
            "Vy[m/s]": v}
            )
        
        free_cuda_memory()
        
        if self.avg_u is None:
            self.avg_u = np.zeros_like(u, dtype=np.float64)
            self.avg_v = np.zeros_like(v, dtype=np.float64)

        print(f"Avg PIV time {((time.time() - start)/(i+1)*1000):.0f} ms")
        self.signals.progress.emit(0)
        u_inst = np.stack(u_inst, axis=0)
        v_inst = np.stack(v_inst, axis=0)
        self.avg_u = np.mean(u_inst, axis=0, dtype=np.float64) 
        self.avg_v = np.mean(v_inst, axis=0, dtype=np.float64)
        self.signals.progress.emit(25)
        uu = np.mean((u_inst - self.avg_u)**2, axis=0, dtype=np.float64)
        self.signals.progress.emit(50)
        vv = np.mean((v_inst - self.avg_v)**2, axis=0, dtype=np.float64)
        self.signals.progress.emit(75)
        uv = np.mean((u_inst - self.avg_u)*(v_inst - self.avg_v), axis=0, dtype=np.float64)
        self.signals.progress.emit(100)
        out = (x, y, self.avg_u, self.avg_v)


        mid_i, mid_j = x.shape[-2]//2, x.shape[-1]//2
        dx = (x[mid_i, mid_j + 1] - x[mid_i, mid_j]) / 1000
        dy = (y[mid_i + 1, mid_j] - y[mid_i, mid_j]) / 1000

        dUy, dUx = np.gradient(self.avg_u, dx, dy, edge_order=2) # U  - Xcomp
        dVy, dVx = np.gradient(self.avg_v, dx, dy, edge_order=2) # V  - Ycomp
        table = {
            "x[mm]": x,
            "y[mm]": y,
            "Vx[m/s]": self.avg_u,
            "Vy[m/s]": self.avg_v,
            "(vx-Vx)(vy-Vy)[m^2/s^2]": uv,
            "(vx-Vx)^2[m^2/s^2]": uu,
            "(vy-Vy)^2[m^2/s^2]": vv,
            "dVx/dx[1/s]": dUx,
            "dVx/dy[1/s]": dUy,
            "dVy/dx[1/s]": dVx,
            "dVy/dy[1/s]": dVy,
            "W[1/s]": (dVx - dUy),
            "S[1/s]": (dVx + dUy),
        }
        name = os.path.basename(os.path.normpath(self.folder))
        save_table(f"{name}.txt", self.piv_params.save_dir, table.copy())
        self.signals.finished.emit(table)



class OnlineWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, folder, piv_params: PIVparams, *args, **kwargs) -> None:
        super().__init__(*args, parent=None, **kwargs)
        self.folder = folder
        self.piv_params = piv_params
        self.avg_u      = None
        self.avg_v      = None
        self.is_paused  = False
        self.is_running = True
        self._pair    = []
        self._pairdeq = deque()
        self._process = QProcess() # Keep a reference to the QProcess (e.g. on self) while it's running.
        self._process.readyReadStandardOutput.connect(self.handle_stdout)
        self._process.readyReadStandardError.connect(self.handle_stderr)
        self._process.stateChanged.connect(self.handle_state)
        self._process.finished.connect(self.process_finished)  # Clean up once complete.


    def handle_stderr(self):
        data = self._process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self._process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        if stdout.endswith(
            f"a{self.piv_params.file_fmt}"
            ):
            self._pair.append(stdout)
        elif stdout.endswith(
            f"b{self.piv_params.file_fmt}"
            ) and self._pair:
            self._pair.append(stdout)
            self._pairdeq.append(self._pair)
            self._pair = []
        
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")

    def process_finished(self):
        self.message("Process finished.")
        self._process = None

    def message(self, string: str):
        print(string)

    def run(self):
        """Long PIV task.
        Emits: 
        img, x_grid, y_grid, U, V: np.ndarray
        """
        self._process.start(sys.executable, 
                    ['watchman.py', self.folder])
        while self._process.state() is QProcess.Running:
            while self.is_paused:
                time.sleep(0)
            pair = self._pairdeq.pop()

class WatchMan(QObject):
    signals = WorkerSignals()
    def __init__(self, folder: str, file_fmt: str, *args, **kwargs) -> None:
        super().__init__(*args, parent=None, **kwargs)
        self.folder = folder
        self.file_fmt = file_fmt
        self.filenames = {os.path.join(self.folder, name) for name 
            in os.listdir(folder) if name.endswith(file_fmt)}
        self.img_pairs: list = []

    def update(self):
        filenames = {os.path.join(self.folder, name) for name 
            in os.listdir(self.folder) if name.endswith(self.file_fmt)}
        new_files = [*filenames.difference(self.filenames)]
        self.filenames = filenames
        self.set_image_pairs(new_files=new_files)

    def set_image_pairs(self, new_files: list[str]):
        new_files.sort(key=natural_keys)
        if len(new_files) % 2 == 0 and new_files[0].endswith("_a" + self.file_fmt):
            self.img_pairs = list(zip(new_files[::2], new_files[1::2]))
        elif len(new_files) % 2 != 0 and new_files[0].endswith("_a" + self.file_fmt):
            self.img_pairs = list(zip(new_files[:-1:2], new_files[1:-1:2]))
        elif len(new_files) % 2 != 0 and new_files[0].endswith("_b" + self.file_fmt):
            self.img_pairs = list(zip(new_files[1::2], new_files[2::2]))
        elif len(new_files) % 2 == 0 and new_files[0].endswith("_b" + self.file_fmt):
            self.img_pairs = list(zip(new_files[1:-1:2], new_files[2:-1:2]))
    def get_image_pairs(self):
        if self.img_pairs:
            return self.img_pairs
        