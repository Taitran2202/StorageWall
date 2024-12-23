import logging
import time

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from controllers.manage_sw_cmd import ManageSWCmd
from modules.general_model import GeneralModel


class MainController(QThread):
    signal_disconnected_from_server = pyqtSignal()

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self.log = logging.getLogger(__name__)
        self._general_model = general_model
        self._manage_sw_cmd = ManageSWCmd(self._general_model)
        self._manage_sw_cmd.signal_disconnected_from_server.connect(
            self.signal_disconnected_from_server)

    def initialize(self):
        self._manage_sw_cmd.start()

    @pyqtSlot()
    def run(self):
        print("Start thread MainController...")
        while not self._exit_flag:
            time.sleep(1)
            self.log.info("MainController thread is running")
        print("End thread MainController...")
