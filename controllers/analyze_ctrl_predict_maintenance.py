'''
23-Mar-24
'''
import os
import cv2
import time
import json
import copy
import logging
import numpy as np
from globals import GV
from modules.general_model import GeneralModel
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QMutex
from globals import ML_CMD_HANDSHAKE_REQUEST, STRUCT_ANALYZE_INFO
from modules.predictive_maintenance.predictive_maintenance_binary import Predictive_Maintenance



os.environ['CUDA_VISIBLE_DEVICES']='-1'

class AnalyzeController(QThread):
    signal_analyze_finished = pyqtSignal(object)

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self._general_model = general_model
        self.log = logging.getLogger(__name__)
        self._handshake_info = ML_CMD_HANDSHAKE_REQUEST()
        self._analyze_list = list()
        self._lock_request_list = QMutex()

    def append_analyze_info(self, _analyze_info: STRUCT_ANALYZE_INFO):
        self._lock_request_list.lock()
        self.log.info(
            "append_analyze_info analyze_info.cmdUniqueID = {}".format(
                _analyze_info.cmdUniqueID))
        self._analyze_list.append(_analyze_info)
        self._lock_request_list.unlock()

    def stop_analyze(self):
        self._lock_request_list.lock()
        self.log.info("stop_analyze")
        for idx in range(len(self._analyze_list)):
            analyze_info = self._analyze_list[idx]
            analyze_info.cancelFlag = True
            self._analyze_list[idx] = analyze_info
        self._lock_request_list.unlock()


    def set_handshake_info(self, request: ML_CMD_HANDSHAKE_REQUEST):
        self._handshake_info = request

    @pyqtSlot()
    def run(self):
        
        print("Start thread MainController...")
        
        # START LOAD MODEL

        # TODO Initalize
        model_classification_SVM_weight_path = None
        model_classification_SVM_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/model_object_detect_phone_slot.onnx'

        slot_id = None

        # load model detect phone slot
        try:
            phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.75)
            self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Load model phone_slot_object_detector successfuly!')
        except Exception as e:
            self.log.info(f'[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] FAIL to Load model detect phone slot, Error = {e}')
 
        # END LOAD MODEL
        while not self._exit_flag:
            time.sleep(0.001)
            # Get command from request list and execute
            
            # start analyze
            module_info = None
            if len(self._analyze_list) > 0:
                self._lock_request_list.lock()
                for info in self._analyze_list[:]:
                    if not info.cancelFlag:
                        module_info = copy.deepcopy(info)
                        break
                    else:
                        self.signal_analyze_finished.emit(info)
                        self._analyze_list.remove(info)
                self._lock_request_list.unlock()
            else:
                continue

            if module_info is None:
                continue


            cmdUniqueID = module_info.cmdUniqueID
            transaction_id = module_info.request.transactionID # TransactionID from SW
            request_type = module_info.request.requestType
            # for detect phone slot
            path_debug = module_info.request.pathToDebug # Path debug from SW
            path_image = module_info.request.pathToImg # Path image from SW
            positionToDetect = module_info.request.positionToDetect # Position to detect from SW
            x_position = module_info.request.x_position # Slot id from SW
            z_position = module_info.request.z_position # Slot id from SW
            
            cmdUniqueID_cur = cmdUniqueID
            
            if request_type == 'predict_maintenance':
                # Start execute ...
                self.log.info("\n\n\n")
                self.log.info("\tStart analyze for cmdUniqueID = {}".format(module_info.cmdUniqueID))
                # Create folder debug
                self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] -------------------------- INIT Version 1.04 --------------------------')
                self.log.info(f'[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] TransactionID = {transaction_id}, commandID = {cmdUniqueID}, request_type = {request_type}')

                # Create name for calibrate folder
                if request_type == 'calibrate':
                    path_parts = path_image_2.split("/")
                    last_part = path_parts[-1]
                    name_box = last_part.split(".")[0]
                    cmdUniqueID = name_box
                elif request_type == 'detect_phone_slot':
                    path_parts = path_image.split("/")
                    last_part = path_parts[-1]
                    name_box = last_part.split(".")[0]
                elif request_type == 'detect_phone_slot_fast':
                    path_parts = path_image.split("/")
                    last_part = path_parts[-1]
                    name_box = last_part.split(".")[0]
                
                position_of_phoneslot_cur = (x_position, z_position)
                    
                # init path debug by command id
                ml_debug_path = os.path.join(GV().path_to_debug_folder, cmdUniqueID)
                
                # if transaction_id
                if transaction_id != "":                
                    # Create transaction ID folder
                    transaction_id_path = os.path.join(GV().path_to_debug_folder,transaction_id)
                    if not os.path.exists(transaction_id_path):
                        os.makedirs(transaction_id_path)
                        self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Created Folder TransactionID sucessful ==> Go to update ml_debug_path!\n')
                    else:
                        self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Folder TransactionID debug already exits!\n')
                    
                    # update ml_debug_path 
                    ml_debug_path = os.path.join(GV().path_to_debug_folder,transaction_id,cmdUniqueID) # update ml_debug_path when transaction_ID exists!!
                
                # Create folder debug
                if not os.path.exists(ml_debug_path):
                    os.makedirs(ml_debug_path)
                    self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Created Folder debug sucessful!\n')
                else:
                    self.log.info('[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Folder debug already exits!\n')
                    
                self.log.info(f'[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Folder debug final= {ml_debug_path}')           

                # check request_type
                if request_type == "detect_phone_slot":
                    try:                  
                        self.log.info("[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] START cmdUniqueID = {} path_image= {} path_debug = {}".format(cmdUniqueID,path_image,path_debug))
                        path_log_id_phone_slot = os.path.join(ml_debug_path,"log_id_phone_slot.txt")
                        
                        # Write original image
                        image_detect_phone = cv2.imread(path_image)
                        if image_detect_phone is not None:
                            cv2.imwrite(os.path.join(ml_debug_path,f'{cmdUniqueID}_detect_phone_slot_org.jpg'), image_detect_phone)
                        else:
                            self.log.info(f"[ANALYZE_CONTROLER_PREDICT_MAINTENANCE] Image detect phone slot from SW is Null ==>ERROR \n")
                        
                        # Start detect phone slot
                        if positionToDetect == 'input_dock':
                            result, debug_img = phone_input_dock_object_detector.run(path_image, positionToDetect)
                        elif positionToDetect == 'buffer':
                            result, debug_img = phone_buffer_object_detector.run(path_image, positionToDetect)
                        else:
                            #result, debug_img = phone_slot_object_detector.run(path_image, positionToDetect)
                            result, debug_img = phone_slot_object_detector_triton.run(path_image, positionToDetect)

                        if debug_img is not None:
                            cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)
                        else:
                            self.log.info(f"[DETECT_PHONE_SLOT] Image detect phone slot is Null ==>ERROR \n")
                            
                        self.log.info(f"[DETECT_PHONE_SLOT] Result detect phone = {result}")
                        # Post process result
                        if 'have_phone' in result:
                            self.log.info("[DETECT_PHONE_SLOT] END cmdUniqueID = {} result = HAVE_PHONE path_image= {} ml_debug_path = {}".format(cmdUniqueID,path_image,ml_debug_path))
                            module_info.response.result_string = 'have_phone'
                        else:
                            module_info.response.result_string = 'none_phone'
                            self.log.info("[DETECT_PHONE_SLOT] END cmdUniqueID = {} result = NO_PHONE path_image= {} ml_debug_path = {}".format(cmdUniqueID,path_image,ml_debug_path))
                        
                        # Copy log file total to debug folder
                        path_backup_log_phone_slot = os.path.join(ml_debug_path,"log_phone_slot.log")
                        cmd = f"cp -rf {os.path.join(GV().path_to_logs_folder,'log_file.log')} {path_backup_log_phone_slot}"
                        self.log.info(f"[DETECT_PHONE_SLOT] cmd backup log: {cmd}")
                        os.system(cmd)        
                    except Exception as e:
                        self.log.info("[DETECT_PHONE_SLOT] FAIL to execute cmdUniqueID = {} path_image = {} path_debug = {} Error = {}".format(cmdUniqueID,path_image,path_debug,e))

            # Write down data after preprocess
            if request_type == 'detect_phone_slot' or request_type == 'detect_phone_slot_fast' or request_type == 'calibrate' or request_type == 'scan_barcode':
                self.log.info("Mode analyze = {}".format(request_type))
                module_info.response.result = 0
                module_info.response.cmdUniqueID = cmdUniqueID
                module_info.response.slot_id = slot_id
                self.log.info("module_info = {}".format(module_info))
                self.signal_analyze_finished.emit(module_info)
                self._lock_request_list.lock()

                for info in self._analyze_list[:]:
                    if info.cmdUniqueID == module_info.cmdUniqueID:
                        self._analyze_list.remove(info)
                self._lock_request_list.unlock()

        print("End thread MainController...")
