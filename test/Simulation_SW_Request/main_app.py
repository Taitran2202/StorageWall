

import sys
from PyQt5.QtCore import QCoreApplication, pyqtSignal, pyqtSlot
from modules.general_model import GeneralModel
from controllers.main_ctrl import MainController
from globals import GV
import logging
from logging.handlers import RotatingFileHandler
import os
import psutil
import argparse
from signal_handler import SignalHandler


class App(QCoreApplication):
    signal_quit_app = pyqtSignal(int)
    slot_close_app = pyqtSlot()

    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self._general_model = GeneralModel()
        self._main_controller = MainController(self._general_model)
        print("Hello world")

    def initialize(self, master_pid):
        self.init_log_file()
        self.signal_quit_app.connect(self.exit)
        self._main_controller.signal_quit_app.connect(self.slot_close_app)
        self._main_controller.master_pid = int(master_pid)
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
        log_file_path = os.path.join(GV().path_to_logs_folder, "subapp_execute_ml.log")
        print("log_file_path = {}".format(log_file_path))
        
        rfh = RotatingFileHandler(
            filename=log_file_path,
            mode='a',
            maxBytes=1*1024*1024,
            backupCount=2,
            encoding=None,
            delay=0
        )

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            handlers=[
                rfh,
                logging.StreamHandler()
            ]
        )
        GV().write_log("Init log file at {}".format(log_file_path))


def already_running():
    script_name = os.path.basename(__file__)
    pidFile = "/tmp/ml_template_app_simulator_SW_request.pid"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master-pid', help='Master PID', required=True)
    args = parser.parse_args()
    print("args.master_pid = {}".format(args.master_pid))
    return args

if __name__ == '__main__':
    isRunning = already_running()
    isRunning = False
   
    if isRunning:
        sys.exit()
    # args = parse_args()

    try:
        app = App(sys.argv)
        signal_handler = SignalHandler(app)
        signal_handler.activate()
        # app.initialize(args.master_pid)
        app.initialize(1)
        sys.exit(app.exec_())
    except Exception as e:
        logger = logging.getLogger('main')
        logger.exception("Caught exception" + str(e))

### c