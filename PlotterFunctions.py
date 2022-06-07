import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox

def show_message(message: str) -> None:
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setText(message)
    msgbox.setStandardButtons(QMessageBox.Ok)
    msgbox.buttonClicked.connect(msgbox.close)
    msgbox.exec_()

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def set_width(obj: object, target_type: type, width: int):
    '''
    Helper function to set size of object's inner widgets
    '''
    if "setFixedWidth" in dir(target_type):
        for key, val in obj.__dict__.items():
            if isinstance(val, target_type):
                obj.__dict__[key].setFixedWidth(width)


def save_table(name, path, data, sep:str=', '):
    data = pd.DataFrame(data)
    if not os.path.exists(path):
        os.mkdir(path)
    path = uniquify(os.path.join(path, name))
    np.savetxt(path, data.values, 
        delimiter=sep, header=sep.join(data.columns), 
        comments='', fmt="%.6f")


def make_name(name: str, key: str, orientation: bool) -> tuple:
    orientation = "Hor" if orientation else "Vert"
    name = os.path.basename(os.path.normpath(name))
    key = key[:key.find("[")].replace("/", "_")
    filename =  f"{name}_{key}_{orientation}_profile.txt".replace(' ', '')
    curr_dir = os.getcwd()
    save_dir = os.path.join(curr_dir, "Out")
    return filename, save_dir

def autoscale_y(ax, margin=0.2):

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h

        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def find_grid(data: pd.DataFrame) -> int:
    first_key = data.keys()[0]
    values = data[first_key].values
    zero_val = values[0]
    for idx, val in enumerate(values):
        if val == zero_val and idx > 0:
            break
    return idx

def reshape_data(data: pd.DataFrame, grid: int) -> dict:
    data = {key: val.values.reshape(-1, grid) for key, val in data.items()}
    return data


class Singleton:
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Database(Singleton):
    def __init__(self):
        Singleton.__init__(self)
    
    def get(self):
        return self._data

    def load(self, name):
        try:
            data = pd.read_csv(name, sep=None, engine="python")
            grid = find_grid(data)
            self._data = reshape_data(data, grid)
        except Exception as e:
            show_message(f"Make sure that file is properly specified\n caught Exception{e}")
            return
        _, name = os.path.split(name)
        self.name, _ = os.path.splitext(name)
