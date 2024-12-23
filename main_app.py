'''
python3 main_app.py
26-Mar-24 14:15
'''
import ctypes
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import psutil
from PyQt5.QtCore import QCoreApplication, pyqtSignal, pyqtSlot

from controllers.main_ctrl import MainController
from globals import GV
from modules.general_model import GeneralModel
from signal_handler import SignalHandler


class App(QCoreApplication):
    signal_quit_app = pyqtSignal(int)
    slot_close_app = pyqtSlot()

    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self._general_model = GeneralModel()
        self._main_controller = MainController(self._general_model)
        self._main_controller.signal_disconnected_from_server.connect(
            self.slot_close_app)

    def initialize(self):
        self.init_log_file()
        self.signal_quit_app.connect(self.exit)
        self._main_controller.initialize()
        self._main_controller.start()

    @pyqtSlot()
    def slot_close_app(self):
        self.signal_quit_app.emit(0)

    def init_log_file(self):
        if not os.path.exists(GV().path_to_debug_folder):
            os.makedirs(GV().path_to_debug_folder)
        if not os.path.exists(GV().path_to_logs_folder):
            os.makedirs(GV().path_to_logs_folder)
        log_file_path = os.path.join(GV().path_to_logs_folder, "log_file.log")
        print("log_file_path = {}".format(log_file_path))
        rfh = RotatingFileHandler(filename=log_file_path, mode='a', maxBytes=1*1024*1024, backupCount=2, encoding=None, delay=0)
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s", datefmt="%y-%m-%d %H:%M:%S:%f", handlers=[rfh, logging.StreamHandler()])

        # file_handler = logging.FileHandler(log_file_path, mode='a')
        # file_handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s", datefmt="%y-%m-%d %H:%M:%S:%f")
        # file_handler.setFormatter(formatter)
        # logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, logging.StreamHandler()])
        
        GV().write_log("Init log file at {}".format(log_file_path))


def already_running():
    script_name = os.path.basename(__file__)
    pidFile = "/tmp/ml_subapp_storagewall.pid"
    my_pid = os.getpid()
    if os.path.exists(pidFile):
        with open(pidFile) as f:
            pid = f.read()
            pid = int(pid) if pid.isnumeric() else None
        is_pid_exist = psutil.pid_exists(pid)
        print("File {} exists".format(pidFile))
        print("The pid {} exists = {}".format(pid, is_pid_exist))
        is_same_script = False
        if is_pid_exist:
            print(psutil.Process(pid).cmdline())
            list_cmd = psutil.Process(pid).cmdline()
            for cmd in list_cmd:
                if cmd.endswith(script_name):
                    is_same_script = True
                    break
        if pid is not None \
                and is_pid_exist \
                and is_same_script:
            print("App {} is running with pid {}"
                .format(script_name, pid))
            return True
    with open(pidFile, 'w') as f:
        f.write(str(my_pid))
    return False


if __name__ == '__main__':
    isRunning = already_running()
    
    # Set process name
    libc = ctypes.CDLL(None)
    libc.prctl(15,b'appPython', 0, 0, 0)  # PR_SET_NAME = 15
        
    if isRunning:
        sys.exit()
    try:
        app = App(sys.argv)
        signal_handler = SignalHandler(app)
        signal_handler.activate()
        app.initialize()
        sys.exit(app.exec_())
    except Exception as e:
        logger = logging.getLogger('main')
        logger.exception("Caught exception" + str(e))
