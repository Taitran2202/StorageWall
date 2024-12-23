import logging
import os
import time

from PyQt5.QtCore import QMutex, QThread, pyqtSignal, pyqtSlot

from controllers.analyze_ctrl import AnalyzeController
from controllers.analyze_ctrl_PI_fastmode import AnalyzeControllerPIFastmode
from globals import (ML_ANALYZE_APP_PORT, ML_APP_CMD_KEY_ANALYZE,
                     ML_APP_CMD_KEY_HANDSHAKE, ML_APP_CMD_KEY_HELLO,
                     ML_APP_CMD_KEY_PHYSICAL_INVENTORY,
                     ML_APP_CMD_KEY_STOP_ANALYZE,
                     ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY, ML_APP_COMMAND,
                     ML_APP_NAME, ML_CMD_ANALYZE_REQUEST,
                     ML_CMD_HANDSHAKE_REQUEST, ML_CMD_HANDSHAKE_RESPONSE,
                     ML_CMD_HELLO_RESPONSE, STRUCT_ANALYZE_INFO)
from modules.general_model import GeneralModel
from network.client_socket import ClientSocket


class ManageSWCmd(QThread):
    slot_analyze_finished = pyqtSlot(object)
    signal_send_data_to_server = pyqtSignal(dict)
    slot_receive_data_from_server = pyqtSlot(dict)
    signal_disconnected_from_server = pyqtSignal()

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self.log = logging.getLogger(__name__)
        self._general_model = general_model
        self._analyze_client = ClientSocket(self)
        self._analyze_client.port_info = ML_ANALYZE_APP_PORT
        self.signal_send_data_to_server.connect(self._analyze_client.slot_send_data_to_server)
        self._analyze_client.signal_receive_data_from_server.connect(self.slot_receive_data_from_server)
        self._analyze_client.signal_disconnected_from_server.connect(self.signal_disconnected_from_server)
        self._analyze_client.connect_to_server()
        
        self._analyze_controller = AnalyzeController(self._general_model)
        self._analyze_controller.signal_analyze_finished.connect(self.slot_analyze_finished)
        self._analyze_controller.start()
        
        # self._analyze_controller_PI = AnalyzeControllerPIFastmode(self._general_model)
        # self._analyze_controller_PI.signal_analyze_finished.connect(self.slot_analyze_finished)
        # self._analyze_controller_PI.start()

        
        self._list_cmd = list()
        self._lock_list_cmd = QMutex()

    @pyqtSlot()
    def run(self):
        print("Start thread ManageSWCmd...")
        start_time = time.time()
        timeout_in_sec = 10
        while not self._exit_flag:
            time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_in_sec:
                start_time = time.time()
                if not self._analyze_client.is_connected:
                    self.log.info(
                        "Client cannot connect to host after timeout.")
                    self.signal_disconnected_from_server.emit()
                    break
            if not self._analyze_client.is_connected:
                continue
            if len(self._list_cmd) <= 0:
                continue
            self.exec_list_cmd()
            self.check_list_cmd()
        print("End thread ManageSWCmd...")

    @pyqtSlot(dict)
    def slot_receive_data_from_server(self, data_dict):
        cmd = ML_APP_COMMAND()
        cmd.cmdKey = data_dict.get("cmdKey", str(""))
        cmd.uniqueID = data_dict.get("uniqueID", str(""))
        cmd.mapCmdData = data_dict.get("mapCmdData", str(""))
        cmd.timeoutRuntime = data_dict.get("timeoutRuntime", 20*1000)
        cmd.timeoutInQueue = data_dict.get("timeoutInQueue", 40*1000)
        cmd.isPriority = data_dict.get("isPriority",False)

        self._lock_list_cmd.lock()
        self.log.info("slot_receive_data_from_server \
            cmd.uniqueID = {}; cmd.cmdKey = {}"
                      .format(cmd.uniqueID, cmd.cmdKey))
        self._list_cmd.append(cmd)
        self._lock_list_cmd.unlock()

    @pyqtSlot(object)
    def slot_analyze_finished(self, analyze_info):
        response = analyze_info.response
        response_dict = dict()
        response_dict["result"] = response.result
        response_dict["result_calib"] = response.result_calib
        response_dict["result_string"] = response.result_string
        response_dict["slot_id"] = response.slot_id
        response_dict["cmdUniqueID"] = response.cmdUniqueID
        cmdUniqueID = analyze_info.cmdUniqueID
        self.log.info(
            "slot_analyze_finished receive cmdUniqueID = {}".format(cmdUniqueID))
        self._lock_list_cmd.lock()
        for idx in range(len(self._list_cmd)):
            cur_cmd = self._list_cmd[idx]
            if cur_cmd.uniqueID == cmdUniqueID:
                cur_cmd.mapCmdData = response_dict
                cur_cmd.isFinished = True
                self._list_cmd[idx] = cur_cmd
                break
        self._lock_list_cmd.unlock()

    def exec_list_cmd(self):
        self._lock_list_cmd.lock()
        for idx in range(len(self._list_cmd)):
            cur_cmd = self._list_cmd[idx]
            if not cur_cmd.isFinished and not cur_cmd.isRunning:
                cur_cmd.isRunning = True
                # execute command
                if cur_cmd.cmdKey == ML_APP_CMD_KEY_HELLO or \
                   cur_cmd.cmdKey == ML_APP_CMD_KEY_HANDSHAKE:
                    cur_cmd = self.process_execute_cmd(cur_cmd)
                    cur_cmd.isFinished = True
                elif cur_cmd.cmdKey == ML_APP_CMD_KEY_ANALYZE or cur_cmd.cmdKey == ML_APP_CMD_KEY_PHYSICAL_INVENTORY:
                    cur_cmd = self.process_start_analyze(cur_cmd)
                elif cur_cmd.cmdKey == ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY:
                    cur_cmd = self.process_stop_physical_inventory(cur_cmd)
                elif cur_cmd.cmdKey == ML_APP_CMD_KEY_STOP_ANALYZE:
                    cur_cmd = self.process_stop_analyze(cur_cmd)
                    cur_cmd.isFinished = True

                self._list_cmd[idx] = cur_cmd
            elif not cur_cmd.isFinished:
                # tracking
                pass
        self._lock_list_cmd.unlock()

    def check_list_cmd(self):
        self._lock_list_cmd.lock()
        for cmd in self._list_cmd[:]:
            if cmd.isFinished:
                self._list_cmd.remove(cmd)
                data_dict = dict()
                data_dict["cmdKey"] = cmd.cmdKey
                data_dict["uniqueID"] = cmd.uniqueID
                data_dict["exitCode"] = cmd.exitCode
                data_dict["mapCmdData"] = cmd.mapCmdData
                data_dict["isTimeoutRuntime"] = cmd.isTimeoutRuntime
                self.log.info("check_list_cmd remove cmd.uniqueID = {}".format(cmd.uniqueID))
                self.signal_send_data_to_server.emit(data_dict)
        self._lock_list_cmd.unlock()

    def process_start_analyze(self, cmd: ML_APP_COMMAND):
        if cmd.cmdKey == ML_APP_CMD_KEY_ANALYZE or cmd.cmdKey == ML_APP_CMD_KEY_PHYSICAL_INVENTORY:
            request = ML_CMD_ANALYZE_REQUEST()
            request.positionToCalib = cmd.mapCmdData.get("positionToCalib", str(""))
            request.positionToDetect = cmd.mapCmdData.get("positionToDetect", str(""))
            request.pathToImg = cmd.mapCmdData.get("pathToImg", str(""))
            request.pathToImg1 = cmd.mapCmdData.get("pathToImg1", str(""))
            request.pathToImg2 = cmd.mapCmdData.get("pathToImg2", str(""))
            request.pathToDebug = cmd.mapCmdData.get("pathToDebug", str(""))
            request.requestType = cmd.mapCmdData.get("requestType", str(""))
            request.pathToList = cmd.mapCmdData.get("pathToList", str(""))
            request.pathToResult = cmd.mapCmdData.get("pathToResult", str(""))
            request.transactionID = cmd.mapCmdData.get("transaction_id", str(""))
            request.x_position = cmd.mapCmdData.get("x_position", 0)
            request.z_position = cmd.mapCmdData.get("z_position", 0)
            request.x_min = cmd.mapCmdData.get("x_min", -1)
            request.x_max = cmd.mapCmdData.get("x_max", 1)
            request.z_min = cmd.mapCmdData.get("z_min", -1)
            request.z_max = cmd.mapCmdData.get("z_max", 1)
            request.numberSlotsDetector = cmd.mapCmdData.get("numberSlotsDetector", -1)
            analyze_info = STRUCT_ANALYZE_INFO()
            analyze_info.request = request
            analyze_info.cmdUniqueID = cmd.uniqueID
            self.log.info(
                "process_start_analyze analyze_info.cmdUniqueID = {}".format(
                    analyze_info.cmdUniqueID))
            if cmd.cmdKey == ML_APP_CMD_KEY_ANALYZE:
                self._analyze_controller.append_analyze_info(analyze_info)
            # elif cmd.cmdKey == ML_APP_CMD_KEY_PHYSICAL_INVENTORY:
            #     self._analyze_controller_PI.append_analyze_info(analyze_info)
        return cmd

    def process_stop_analyze(self, cmd: ML_APP_COMMAND):
        if cmd.cmdKey == ML_APP_CMD_KEY_STOP_ANALYZE:
            moduleIndex = cmd.mapCmdData.get("moduleIndex")
            self.log.info("process_stop_analyze moduleIndex = {}".format(moduleIndex))
            self._analyze_controller.stop_analyze(moduleIndex)
            #self._analyze_controller_PI.stop_analyze(moduleIndex)
            response_dict = dict()
            response_dict["result"] = 0
            cmd.mapCmdData = response_dict
        return cmd

    # def process_stop_physical_inventory(self, cmd: ML_APP_COMMAND):
    #     if cmd.cmdKey == ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY:
    #         self._analyze_controller_PI.stop_physical_inventory()
    #         response_dict = dict()
    #         response_dict["result"] = 0
    #         cmd.mapCmdData = response_dict
    #         self.log.info("Stopping physical inventory, in process_stop_physical_inventory()!!")
    #     return cmd

    def process_execute_cmd(self, cmd):
        if cmd.cmdKey == ML_APP_CMD_KEY_HELLO:
            response = ML_CMD_HELLO_RESPONSE()
            response.appName = ML_APP_NAME
            response_dict = dict()
            response_dict["appName"] = response.appName
            cmd.mapCmdData = response_dict
        elif cmd.cmdKey == ML_APP_CMD_KEY_HANDSHAKE:
            request = ML_CMD_HANDSHAKE_REQUEST()
            request.masterPid = cmd.mapCmdData.get("masterPid",-1)
            request.pathDebug = cmd.mapCmdData.get("pathDebug", str(""))
            request.pathRoot = cmd.mapCmdData.get("pathRoot", str(""))
            request.pathLog = cmd.mapCmdData.get("pathLog", str(""))
            self._analyze_controller.set_handshake_info(request)
            # self._analyze_controller_PI.set_handshake_info(request)
            response = ML_CMD_HANDSHAKE_RESPONSE()
            response.clientPid = os.getpid()
            response.result = 0
            response_dict = dict()
            response_dict["clientPid"] = response.clientPid
            response_dict["result"] = response.result
            cmd.mapCmdData = response_dict
        return cmd
