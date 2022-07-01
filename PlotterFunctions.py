import os
import re
import pandas as pd
import numpy as np
import json
from PyQt5.QtWidgets import QMessageBox

def show_message(message: str) -> None:
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setText(message)
    msgbox.setStandardButtons(QMessageBox.Ok)
    msgbox.buttonClicked.connect(msgbox.close)
    msgbox.exec_()

def uniquify(path: str):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def set_width(obj: object, target_type: type, width: int):
    '''
    Helper function to set size of object's inner widgets
    '''
    if "setFixedWidth" in dir(target_type):
        for key, val in obj.__dict__.items():
            if isinstance(val, target_type):
                obj.__dict__[key].setFixedWidth(width)


def save_table(name, path, data: dict, sep:str=', '):
    for key in data.keys():
        data[key] = data[key].reshape(-1)
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

class PIVparams(object):
    """
    Naive Singletone inplementation 
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
    regime: str = ""

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

class Singleton(object):
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Database(Singleton):
    def __init__(self):
        Singleton.__init__(self)
        self._data: dict = {}

    def get(self):
        return self._data

    def set(self, data: dict):
        self._data = data
    def load(self, name):
        data = pd.read_csv(name, sep=None, engine="python")
        grid = find_grid(data)
        self._data = reshape_data(data, grid)
        _, name = os.path.split(name)
        self.name, _ = os.path.splitext(name)