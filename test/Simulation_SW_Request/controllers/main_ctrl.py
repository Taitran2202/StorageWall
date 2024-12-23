from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
from modules.general_model import GeneralModel
import time
import logging
from controllers.manage_sw_cmd import ManageSWCmd
import psutil


class MainController(QThread):
    signal_quit_app = pyqtSignal()

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self.log = logging.getLogger(__name__)
        self._general_model = general_model
        self._manage_sw_cmd = ManageSWCmd(self._general_model)
        self._master_pid = 0
        # self._manage_sw_cmd.signal_disconnected_from_server.connect(
        #     self.signal_disconnected_from_server)

    @property
    def master_pid(self):
        return self._master_pid
    
    @master_pid.setter
    def master_pid(self, master_pid):
        self._master_pid = master_pid

    def initialize(self):
        self._manage_sw_cmd.start()

    @pyqtSlot()
    def run(self):
        print("Start thread MainController...")
        count_master_pid_not_exist = 0
        while not self._exit_flag:
            time.sleep(1)
            self.log.info("MainController thread is running")
            is_master_pid_exist = False
            if self._master_pid is not None:
                is_master_pid_exist = psutil.pid_exists(self._master_pid)
            else:
                self.log.info("MainController self._master_pid is None")
            
            if is_master_pid_exist is False:
                count_master_pid_not_exist = count_master_pid_not_exist + 1
                self.log.info("MainController self._master_pid is not exist count_master_pid_not_exist = {}".format(count_master_pid_not_exist))
            if count_master_pid_not_exist > 5:
                count_master_pid_not_exist = 0
                self.log.info("MainController self._master_pid is not exist! Stop app")
                self._manage_sw_cmd.exit_flag = True
                self._exit_flag = True
                time.sleep(1)
                self.signal_quit_app.emit()
        print("End thread MainController... self._exit_flag = {}".format(self._exit_flag))

