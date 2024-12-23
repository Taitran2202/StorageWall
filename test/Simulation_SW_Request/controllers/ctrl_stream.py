import copy
import json
import logging
import os
import sys
import time

from PyQt5.QtCore import QMutex, QThread, pyqtSignal, pyqtSlot

from globals import (ML_CMD_HANDSHAKE_REQUEST, ML_CMD_TRACKING_RESPONSE,
                     STRUCT_TRACKING_INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt="%y-%m-%d %H:%M:%S:%f")
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

def dumps_json(json_path, json_data):
    pre_json_data = {'total': None, 'complete': None, 'remaining': None, 'message': None, 'timestamp': None, 'error_code': None}
    if os.path.exists(json_path) is False:
        pre_json_data.update(json_data)
        with open(json_path, 'w') as f:
            json.dump(pre_json_data, f)
            f.close()
    else:
        with open(json_path, 'r') as f: 
            pre_json_data = json.load(f)
            f.close()
        pre_json_data.update(json_data)
        with open(json_path, 'w') as f: 
            json.dump(pre_json_data, f)
            f.close()

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)  
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class StreamController(QThread):
    signal_open_camera = pyqtSignal(object)
    signal_analyze_finished = pyqtSignal(object)
    slot_analyze_finished = pyqtSlot(object)
    signal_analyze_update = pyqtSignal(object)
    slot_analyze_update = pyqtSlot(object)

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self._general_model = general_model
        self.log = logging.getLogger(__name__)
        self._handshake_info = ML_CMD_HANDSHAKE_REQUEST()
        self._analyze_list = list()
        self._lock_request_list = QMutex()
        self.processTrackingPhone = ProcessTrackingPhone()
        self.processTrackingPhone.signal_analyze_update.connect(self.slot_analyze_update)
        self.processTrackingPhone.signal_analyze_finished.connect(self.slot_analyze_finished)

    @property
    def exit_flag(self):
        return self._exit_flag
    
    @exit_flag.setter
    def exit_flag(self, exit_flag):
        self._exit_flag = exit_flag

    def append_analyze_info(self, _analyze_info: STRUCT_TRACKING_INFO):
        self._lock_request_list.lock()
        self.log.info(
            "append_track_info track_info.cmdUniqueID = {}".format(
                _analyze_info.cmdUniqueID))
        self._analyze_list.append(_analyze_info)
        self._lock_request_list.unlock()

    def stop_analyze(self):
        self._lock_request_list.lock()
        self.log.info("stop_analyze")
        self.processTrackingPhone.do_stop()
        for idx in range(len(self._analyze_list)):
            analyze_info = self._analyze_list[idx]
            analyze_info.cancelFlag = True
            self._analyze_list[idx] = analyze_info
        self._lock_request_list.unlock()

    def set_handshake_info(self, request: ML_CMD_HANDSHAKE_REQUEST):
        self._handshake_info = request

    def do_stop_all_process(self):
        self.processTrackingPhone.do_stop()
        self.processTrackingPhone.exit_flag = True

    @pyqtSlot(object)
    def slot_analyze_finished(self, response: ML_CMD_TRACKING_RESPONSE):
        self.log.info("slot_analyze_finish for uniqueID = {}".format(response.uniqueID))
        self._lock_request_list.lock()
        for info in self._analyze_list[:]:
            if info.cmdUniqueID == response.uniqueID:
                self.log.info("slot_analyze_finish found uniqueID = {}".format(response.uniqueID))
                info.response = response
                self.signal_analyze_finished.emit(info)
                break
        self._lock_request_list.unlock()
    
    @pyqtSlot(object)
    def slot_analyze_update(self, response: ML_CMD_TRACKING_RESPONSE):
        self.log.info("slot_analyze_update for uniqueID = {}".format(response.uniqueID))
        self._lock_request_list.lock()
        for info in self._analyze_list[:]:
            if info.cmdUniqueID == response.uniqueID:
                self.log.info("slot_analyze_update found uniqueID = {}".format(response.uniqueID))
                info.response = response
                self.signal_analyze_update.emit(info)
                break
        self._lock_request_list.unlock()

    @pyqtSlot()
    def run(self):
        
        self.log.info("Start thread Tracking MainController...")
        self.processTrackingPhone.initialize()
        
        while not self._exit_flag:
            time.sleep(0.1)
            
            module_info = None
            if len(self._analyze_list) > 0:
                self._lock_request_list.lock()
                for info in self._analyze_list[:]:
                    if not info.cancelFlag:
                        module_info = copy.deepcopy(info)
                        break
                    else:
                        info.response.result = 0
                        self.signal_analyze_finished.emit(info)
                        self._analyze_list.remove(info)
                self._lock_request_list.unlock()
            else: continue
            
            if module_info is None: continue
            # os.system("sudo docker stop docker_triton_server")
            self.log.info("Start model tracking for cmdUniqueID = {}".format(module_info.cmdUniqueID))
            # self.log.info("cuda: {}".format(torch.cuda.is_available()))
            pathHistory = module_info.request.pathHistory
            currentBoxID = module_info.request.currentBoxID
            total_phone = module_info.request.total
            
            self.log.info("total_phone = {}".format(total_phone))

            # Step 1: Reset data and set variables
            self.processTrackingPhone.do_reset()
            self.processTrackingPhone.set_variable(pathHistory, currentBoxID, total_phone, module_info.cmdUniqueID)
            
            # Step 2: Start process tracking
            self.processTrackingPhone.do_tracking_phone()
            
            # Step 3: Summary and end process tracking
            
            self._lock_request_list.lock()
            for info in self._analyze_list[:]:
                if info.cmdUniqueID == module_info.cmdUniqueID:
                    self._analyze_list.remove(info)
            self._lock_request_list.unlock()
            
        self.do_stop_all_process()
        self.log.info("End thread MainController...")