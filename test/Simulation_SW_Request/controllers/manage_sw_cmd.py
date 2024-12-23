import logging
import multiprocessing
import os
import time

from PyQt5.QtCore import QMutex, QThread, pyqtSignal, pyqtSlot

from globals import (ML_APP_CMD_KEY_ANALYZE, ML_APP_CMD_KEY_HANDSHAKE,
                     ML_APP_CMD_KEY_HELLO, ML_APP_CMD_KEY_PHYSICAL_INVENTORY,
                     ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY, ML_APP_COMMAND,
                     ML_CMD_ANALYZE_REQUEST, ML_CMD_HANDSHAKE_REQUEST,
                     ML_CMD_HANDSHAKE_RESPONSE)
from modules.general_model import GeneralModel
from modules.stream_image_and_offset_to_socket_UDP import \
    subprocess_stream_images_and_offset
from network.server_manager import ServerManger


class ManageSWCmd(QThread):
    
    signal_send_data_to_client = pyqtSignal(dict)
    slot_receive_data_from_client = pyqtSlot(dict)
    signal_start_server = pyqtSignal()

    def __init__(self, general_model: GeneralModel):
        
        # Init
        super().__init__()
        self._exit_flag = False
        self._exit_request = False
        self.log = logging.getLogger(__name__)
        self._general_model = general_model
        self._server_manager = ServerManger(self)
        self._list_cmd = list()
        self._lock_list_cmd = QMutex()
        
        # Create a server to connect subApp Analyze 
        self.signal_start_server.connect(self._server_manager.slot_start_server)
        self._server_manager.signal_receive_data_from_client.connect(self.slot_receive_data_from_client)
        self.signal_send_data_to_client.connect(self._server_manager.signal_send_data_to_client)
        self._server_manager.slot_start_server()
        
        video_file_path = "/home/greystone/StorageWall/image_debug/physical_inventory/recording_2024-05-11.avi"
        
        self.process_stream_image_and_offset = multiprocessing.Process(target = subprocess_stream_images_and_offset, args= (self._exit_flag, video_file_path,))
        self.process_stream_image_and_offset.daemon = True
        
        self.log.info('process_stream_image start!')
        
        # self.process_stream_image = multiprocessing.Process(target = subprocess_stream_images, args= (self._exit_flag, video_file_path,))
        # self.process_stream_image.daemon = True
        # self.process_stream_image.start()
        # self.log.info('process_stream_image start!')
        
        # self.process_stream_offset = multiprocessing.Process(target = subprocess_stream_offset, args= (self._exit_flag,))
        # self.process_stream_offset.daemon = True
        # self.process_stream_offset.start()
        # self.log.info('process_stream_offset start!')
        
    @property
    def exit_flag(self):
        return self._exit_flag
    
    @exit_flag.setter
    def exit_flag(self, exit_flag):
        self._exit_flag = exit_flag


    def start_analyze_command(self):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_HELLO
        command.uniqueID = "1"
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "hello"
        request.transactionID = "1"
        command.mapCmdData = request.__dict__
        return command
    
    def request_physical_inventory_left(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_PHYSICAL_INVENTORY
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "physical_inventory"
        request.positionToDetect = "left"
        request.pathToImg = ""
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count)
        command.mapCmdData = request.__dict__
        return command
    
    def request_physical_inventory_right(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_PHYSICAL_INVENTORY
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "physical_inventory"
        request.positionToDetect = "right"
        request.pathToImg = ""
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count)
        command.mapCmdData = request.__dict__
        return command

    def request_physical_inventory_finished(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = ""
        request.positionToDetect = ""
        request.pathToImg = ""
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count)
        command.mapCmdData = request.__dict__
        return command
    
    def request_analyze_scan_barcode(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "scan_barcode"
        request.pathToImg = "/home/greystone/StorageWall/image/scan_barcode/cam_1_box_side_1_square_11072024115429.jpg"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)

        command.mapCmdData = request.__dict__
        return command
    
    def request_analyze_detect_phone_slot(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "detect_phone_slot"
        request.positionToDetect = "right"
        request.numberSlotsDetector = 3
        request.pathToImg = "/home/greystone/StorageWall/debug/calibration_W01LSR58C31_34_calib_2.jpg"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        command.mapCmdData = request.__dict__
        return command
    
    def request_analyze_check_device_slot_status(self, count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "check_device_slot_status"
        request.numberSlotsDetector = 1
        request.positionToDetect = "right"
        request.pathToImg = "/home/greystone/StorageWall/debug/W01W01LSR49C39_20241011152555095.png"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        command.mapCmdData = request.__dict__
        return command   
    
    def request_analyze_check_phone_on_gripper(self, count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "check_phone_on_gripper"
        request.numberSlotsDetector = 1
        request.positionToDetect = "right"
        request.pathToImg = "/home/greystone/StorageWall/debug/W01RSR02C02_image.png"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        command.mapCmdData = request.__dict__
        return command    
    
    def request_analyze_detect_3phones_slot(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "detect_phone_slot"
        request.numberSlotsDetector = 3
        request.positionToDetect = "left"
        request.pathToImg = "/home/greystone/StorageWall/debug/W01W01LSR13C38_20240925192727012.png"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        command.mapCmdData = request.__dict__
        return command
    
    def request_analyze_detect_phone_slot_fast(self,count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "detect_phone_slot_fast"
        request.positionToDetect = "left"
        request.pathToImg = f"/home/greystone/StorageWall/image_debug/physical_inventory/image/image_{count}.jpg"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        request.x_position = int(15)
        request.z_position = int(count*3.45)
        command.mapCmdData = request.__dict__
        return command
    
    def request_calibrate(self, count):
        command = ML_APP_COMMAND()
        command.cmdKey = ML_APP_CMD_KEY_ANALYZE
        command.uniqueID = str(count)
        command.timeoutInQueue = 10000
        command.timeoutRuntime = 5000
        command.isPriority = True
        request = ML_CMD_ANALYZE_REQUEST()
        request.requestType = "calibrate"
        request.positionToCalib = "left"
        request.pathToImg1 = "/home/greystone/StorageWall/debug/calibration_W01LSR01C04_4_calib_1.jpg"
        request.pathToImg2 = "/home/greystone/StorageWall/debug/calibration_W01LSR58C31_34_calib_2.jpg"
        request.pathToDebug = "/home/greystone/StorageWall"
        request.transactionID = str(count + 10)
        command.mapCmdData = request.__dict__
        return command

    @pyqtSlot()
    def run(self):
        print("Start thread ManageSWCmd...")
        start_time = time.time()
        start_time_send_request = 0
        timeout_in_sec = 5
        timeout_send_request = 5
        start_analyze_command = None
        count_times_send_request = 0
        _exit_request_1 = False
        _exit_request_2 = False
        _exit_request_3 = False
        _exit_request_4 = False

        while not self._exit_flag:
            time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_in_sec:
                start_time = time.time()
                is_listening, number_of_client = self._server_manager.get_sever_status()
                self.log.info("ServerManger is running... is_listening = {} number_of_client = {}".format(is_listening, number_of_client))
                self.log.info("ServerManger port... is_listening = {} number_of_client = {}".format(self._server_manager._port_info, number_of_client))
                if not is_listening: self.signal_start_server.emit()

            if self._server_manager.get_sever_status()[1] > 0 and not self._exit_request:
                time.sleep(5)
                self.log.info("=================> CPP send start_analyze_command")
                start_analyze_command = self.start_analyze_command()
                self.signal_send_data_to_client.emit(start_analyze_command.__dict__)
                self._exit_request = True
                self.log.info(f"data send to client: {start_analyze_command}")
                time.sleep(5)

            elif self._server_manager.get_sever_status()[1] >0:
                for i in range(1,367):
                    self.log.info("=================> CPP send request_analyze_detect_phone_slot")
                    request_analyze_command = self.request_analyze_detect_phone_slot(i)
                    self.signal_send_data_to_client.emit(request_analyze_command.__dict__)
                    self.log.info(f"data send to client: {request_analyze_command}")
                    count_times_send_request += 1
                    time.sleep(1)

            if len(self._list_cmd) <= 0: continue
            self.exec_list_cmd()
            self.check_list_cmd()
        print("End thread ManageSWCmd... self._exit_flag = {}".format(self._exit_flag))
        
        #self.process_stream_image_and_offset.terminate()
        # self.process_stream_image.terminate()
        # self.process_stream_offset.terminate()
    
    @pyqtSlot(dict)
    def slot_receive_data_from_client(self, data_dict):
        cmd = ML_APP_COMMAND()
        cmd.cmdKey = data_dict.get("cmdKey", "")
        cmd.uniqueID = data_dict.get("uniqueID", "")
        cmd.mapCmdData = data_dict.get("mapCmdData", dict())
        cmd.timeoutRuntime = data_dict.get("timeoutRuntime", 0)
        cmd.timeoutInQueue = data_dict.get("timeoutInQueue", 0)
        cmd.isPriority = data_dict.get("isPriority", False)
        self._lock_list_cmd.lock()
        self.log.info("slot_receive_data_from_client cmd.uniqueID = {}; cmd.cmdKey = {}".format(cmd.uniqueID, cmd.cmdKey))
        self._list_cmd.append(cmd)
        self._lock_list_cmd.unlock()


    def exec_list_cmd(self):
        self._lock_list_cmd.lock()
        for idx in range(len(self._list_cmd)):
            cur_cmd = self._list_cmd[idx]
            if not cur_cmd.isFinished and not cur_cmd.isRunning:
                cur_cmd.isRunning = True
                if cur_cmd.cmdKey == ML_APP_CMD_KEY_HELLO or \
                    cur_cmd.cmdKey == ML_APP_CMD_KEY_HANDSHAKE:
                    cur_cmd = self.process_execute_cmd(cur_cmd)
                    cur_cmd.isFinished = True
                elif cur_cmd.cmdKey == ML_APP_CMD_KEY_ANALYZE:
                    pass
                self._list_cmd[idx] = cur_cmd
            elif not cur_cmd.isFinished:
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
                self.log.info("check_list_cmd remove cmd.cmdKey = {} cmd.uniqueID = {}".format(cmd.cmdKey, cmd.uniqueID))
                self.signal_send_data_to_client.emit(data_dict)
        self._lock_list_cmd.unlock()


    def process_execute_cmd(self, cmd):
        if cmd.cmdKey == ML_APP_CMD_KEY_HELLO:
            response_dict = dict()
            response_dict["cmdKey"] = ML_APP_CMD_KEY_HELLO
            response_dict["appName"] = "ML_SUBAPP_CPP"
            cmd.mapCmdData = response_dict
        elif cmd.cmdKey == ML_APP_CMD_KEY_HANDSHAKE:
            request = ML_CMD_HANDSHAKE_REQUEST()
            request.masterPid = cmd.mapCmdData.get("masterPid")
            request.pathDebug = cmd.mapCmdData.get("pathDebug")
            request.pathRoot = cmd.mapCmdData.get("pathRoot")
            request.pathLog = cmd.mapCmdData.get("pathLog")
            self._controller_analyze.set_handshake_info(request)
            response = ML_CMD_HANDSHAKE_RESPONSE()
            response.clientPid = os.getpid()
            response.result = 0
            response_dict = dict()
            response_dict["clientPid"] = response.clientPid
            response_dict["result"] = response.result
            cmd.mapCmdData = response_dict
        return cmd