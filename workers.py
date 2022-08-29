import os
import time
import numpy as np
from collections import deque
from PlotterFunctions import PIVparams, natural_keys
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from PlotterFunctions import save_table
from torchPIV import OfflinePIV, free_cuda_memory



class Worker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    output   = pyqtSignal(dict)
    failed   = pyqtSignal()

class PIVWorker(Worker):
    def __init__(self, piv_params: PIVparams, *args, **kwargs) -> None:
        super().__init__(*args, parent=None, **kwargs)
        self.folder     = piv_params.folder
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
        if len(piv_gen) == 0:
            self.failed.emit()
            return
        u_inst = []
        v_inst = [] 
        x = y = u = v = np.zeros((10, 10))
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
            self.progress.emit((i + 1)/len(piv_gen)*100)
            output = {
            "x[mm]": x,
            "y[mm]": y,
            "Vx[m/s]": u,
            "Vy[m/s]": v
            }
            if self.piv_params.save_opt == "Save all":
                name = os.path.basename(os.path.normpath(self.folder))
                save_table(f"{name}_pair.txt", self.piv_params.save_dir, output.copy())
            self.output.emit(output)
        
        if self.avg_u is None:
            self.avg_u = np.zeros_like(u, dtype=np.float64)
            self.avg_v = np.zeros_like(v, dtype=np.float64)

        print(f"Avg PIV time {((time.time() - start)/(len(piv_gen))*1000):.0f} ms")
        self.progress.emit(0)
        if u_inst:
            u_inst = np.stack(u_inst, axis=0)
            v_inst = np.stack(v_inst, axis=0)
        self.avg_u = np.mean(u_inst, axis=0, dtype=np.float64) 
        self.avg_v = np.mean(v_inst, axis=0, dtype=np.float64)
        self.progress.emit(25)
        uu = np.mean((u_inst - self.avg_u)**2, axis=0, dtype=np.float64)
        self.progress.emit(50)
        vv = np.mean((v_inst - self.avg_v)**2, axis=0, dtype=np.float64)
        self.progress.emit(75)
        uv = np.mean((u_inst - self.avg_u)*(v_inst - self.avg_v), axis=0, dtype=np.float64)
        self.progress.emit(100)
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
        free_cuda_memory()
        name = os.path.basename(os.path.normpath(self.folder))
        if self.piv_params.save_opt != "Dont save":
            save_table(f"{name}_statistics.txt", self.piv_params.save_dir, table.copy())
        self.finished.emit(table)



class OnlineWorker(Worker):

    def __init__(self, folder, piv_params: PIVparams, *args, **kwargs) -> None:
        super().__init__(*args, parent=None, **kwargs)
        self.folder = folder
        self.piv_params = piv_params
        self.avg_u      = None
        self.avg_v      = None
        self.is_paused  = False
        self.is_running = True
        self._pair    = []
        self.timer = QTimer()
        self.watchman = WatchMan(piv_params.folder, piv_params.file_fmt)

    def run(self):
        """Long PIV task.
        Emits: 
        img, x_grid, y_grid, U, V: np.ndarray
        """
        while self.is_running:
            while self.is_paused and self.is_running: 
                time.sleep(0)



class WatchMan(QObject):
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

    def set_image_pairs(self, new_files: list):
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
        