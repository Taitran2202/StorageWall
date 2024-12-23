'''
23-Mar-24
'''
import os
import cv2
import time
import json
import copy
import docker
import logging
import numpy as np
from globals import GV
from modules.general_model import GeneralModel
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QMutex
from globals import ML_CMD_HANDSHAKE_REQUEST, STRUCT_ANALYZE_INFO
from modules.scan_barcode.utils import is_container_running, draw_debug_image, log_msg
from modules.detect_phone_slot.model_detect_phone_slot import PhoneSlotDetector
from modules.detect_object_phone_slot.model_object_detect_phone_slot import PhoneSlotObjectDetector
from modules.detect_object_phone_slot.model_object_calibrate import CalibMagnetArea
from modules.realtime_detection.model_object_detect_phone_slot_triton import Object_Detection_YOLOv8



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
        current_cmd = ''
        current_image_path = ''
        phone_slot_detector = None
        phone_slot_object_detector = None
        container_scan_barcode_is_running = False
        model_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/model_detect_phone_slot.pt'
        model_object_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/model_object_detect_phone_slot.onnx'
        model_object_phone_input_dock_weight_path = '/home/greystone/StorageWall/model_template/InputDock/model_object_detect_phone_input_dock.onnx'
        model_object_detect_phone_in_buffer_weight_path = '/home/greystone/StorageWall/model_template/DetectBuffer/model_object_detect_phone_in_buffer.onnx'
        model_calibrate_weight_path = '/home/greystone/StorageWall/model_template/Calibrate/calibrate.onnx'
        model_object_detect_phone_slot_triton_weight_path = "./model_object_detect_phone_slot.onnx"
        container_scan_barcode_name = 'docker_barcode'
        script_scan_barcode_path = '/home/greystone/StorageWall/apps/ml-subapp/appPython/modules/scan_barcode/detect_barcode_list_path.py'
        labelmap_path_model_object_detect_phone_slot_triton = "/home/greystone/StorageWall/model_template/Model_Triton/model_object_detect_phone_slot/label.pbtxt"
        
        
        # load model detect phone slot
        try:
            phone_slot_detector = PhoneSlotDetector(model_phone_slot_weight_path, 0.85)
            self.log.info('[DETECT_PHONE_SLOT] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_PHONE_SLOT] FAIL to Load model detect phone slot, Error = {e}')
            
        # load model detect phone slot
        try:
            phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.75)
            self.log.info('[DETECT_OBJECT_PHONE_SLOT] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_PHONE_SLOT] FAIL to Load model detect phone slot, Error = {e}')
        
        # load model detect phone in input dock
        try:
            phone_input_dock_object_detector = PhoneSlotObjectDetector(model_object_phone_input_dock_weight_path, 0.75)
            self.log.info('[DETECT_OBJECT_PHONE_SLOT] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_PHONE_SLOT] FAIL to Load model detect phone in input dock, Error = {e}')
            
        # load model detect phone in buffer
        try:
            phone_buffer_object_detector = PhoneSlotObjectDetector(model_object_detect_phone_in_buffer_weight_path, 0.75)
            self.log.info('[DETECT_OBJECT_PHONE_SLOT] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_PHONE_SLOT] FAIL to Load model detect phone in buffer, Error = {e}')

        # load model detect calibrate    
        try:
            calib_magnet_detector = CalibMagnetArea(model_calibrate_weight_path, 0.80)
            self.log.info('[DETECT_OBJECT_MAGNET] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_OBJECT_MAGNET] FAIL to Load model detect magnet, Error = {e}')
            
        try:
            os.system('echo greystone | sudo -S docker run -id --gpus device=0 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/greystone/StorageWall/model_template/Model_Triton:/models --name triton_server nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --strict-model-config=false --model-control-mode=explicit')
            phone_slot_object_detector_triton = Object_Detection_YOLOv8(model_object_detect_phone_slot_triton_weight_path, labelmap_path_model_object_detect_phone_slot_triton, 0.75)
            self.log.info('[DETECT_OBJECT_PHONE_SLOT_TRITON] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[DETECT_OBJECT_PHONE_SLOT_TRITON] FAIL to Load model detect phone slot using triton, Error = {e}')
        # Start docker_barcode
        container_scan_barcode_is_running = is_container_running(container_scan_barcode_name)
        if not container_scan_barcode_is_running:
            self.log.info(f'[SCAN_BARCODE] {container_scan_barcode_name} is not running, start it now!')
            try:
                self.log.info(f'[SCAN_BARCODE] run command start docker {container_scan_barcode_name}')
                os.system(f"docker start {container_scan_barcode_name}")
            except Exception as e:
                self.log.info(f'[SCAN_BARCODE] Fail to start {container_scan_barcode_name}, Error={e}')

         # Check container is running
        if is_container_running(container_scan_barcode_name):
            self.log.info(f"[SCAN_BARCODE] The container {container_scan_barcode_name} is running.")
        else:
            self.log.info(f"[SCAN_BARCODE] The container {container_scan_barcode_name} is NOT running.")

        # Warm up to load triton models
        black_image = np.zeros((600,1200,3),np.uint8)
        ml_warm_up_path = 'home/greystone'
        if not os.path.exists(ml_debug_path):
            os.mkdir(ml_debug_path)
            self.log.info('[ANALYZE_CONTROLER] Created Folder debug sucessful!\n')
        else:
            self.log.info('[ANALYZE_CONTROLER] Folder debug already exits!\n')
        
        # END LOAD MODEL
        while not self._exit_flag:
            time.sleep(0.1)
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

            # Start execute ...
            self.log.info("\n\n\n")
            self.log.info("\tStart analyze for cmdUniqueID = {}".format(module_info.cmdUniqueID))
            cmdUniqueID = module_info.cmdUniqueID
            request_type = module_info.request.requestType
            # for detect phone slot
            path_debug = module_info.request.pathToDebug # Path debug from SW
            path_image = module_info.request.pathToImg # Path image from SW
            positionToDetect = module_info.request.positionToDetect # Position to detect from SW
            # for scan barcode
            path_to_list = module_info.request.pathToList # Path list of image from SW
            path_to_result = module_info.request.pathToResult # Path result from SW => ML return
            transaction_id = module_info.request.transactionID # TransactionID
            #for calibrate
            path_image_1 = module_info.request.pathToImg1 # Path image from SW
            path_image_2 = module_info.request.pathToImg2 # Path image from SW
            position_to_calib = module_info.request.positionToCalib # Position to calibration from SW
            
            # Create folder debug
            self.log.info('[ANALYZE_CONTROLER] -------------------------- INIT Version 1.04 --------------------------')
            self.log.info(f'[ANALYZE_CONTROLER] TransactionID = {transaction_id}, commandID = {cmdUniqueID}, request_type = {request_type}')

            # Create name for calibrate folder
            if request_type == 'calibrate':
                path_parts = path_image_2.split("/")
                last_part = path_parts[-1]
                name_box = last_part.split(".")[0]
                cmdUniqueID = name_box
            elif request_type == 'detect_phone_slot':
                path_parts = path_image.split("/")
                last_part = path_parts[-1]
                name_box = last_part.split("_")[0]
                cmdUniqueID = name_box
                
            # init path debug by command id
            ml_debug_path = os.path.join(GV().path_to_debug_folder, cmdUniqueID)
            
            # if transaction_id
            if transaction_id != "":                
                # Create transaction ID folder
                transaction_id_path = os.path.join(GV().path_to_debug_folder,transaction_id)
                if not os.path.exists(transaction_id_path):
                    os.mkdir(transaction_id_path)
                    self.log.info('[ANALYZE_CONTROLER] Created Folder TransactionID sucessful ==> Go to update ml_debug_path!\n')
                else:
                    self.log.info('[ANALYZE_CONTROLER] Folder TransactionID debug already exits!\n')
                
                # update ml_debug_path 
                ml_debug_path = os.path.join(GV().path_to_debug_folder,transaction_id,cmdUniqueID) # update ml_debug_path when transaction_ID exists!!
            
            # Create folder debug
            if not os.path.exists(ml_debug_path):
                os.mkdir(ml_debug_path)
                self.log.info('[ANALYZE_CONTROLER] Created Folder debug sucessful!\n')
            else:
                self.log.info('[ANALYZE_CONTROLER] Folder debug already exits!\n')
                
            self.log.info(f'[ANALYZE_CONTROLER] Folder debug final= {ml_debug_path}')           
            
            
            # check request_type
            if request_type == "detect_phone_slot":
                try:                  
                    self.log.info("[DETECT_PHONE_SLOT] START cmdUniqueID = {} path_image= {} path_debug = {}".format(cmdUniqueID,path_image,path_debug))
                    path_log_id_phone_slot = os.path.join(ml_debug_path,"log_id_phone_slot.txt")
                    
                    # Write original image
                    image_detect_phone = cv2.imread(path_image)
                    if image_detect_phone is not None:
                        cv2.imwrite(os.path.join(ml_debug_path,f'{cmdUniqueID}_detect_phone_slot_org.jpg'), image_detect_phone)
                    else:
                        self.log.info(f"[DETECT_PHONE_SLOT] Image detect phone slot from SW is Null ==>ERROR \n")
                    
                    # Start detect phone slot
                    if positionToDetect == 'input_dock':
                        result, debug_img = phone_input_dock_object_detector.run(path_image, path_log_id_phone_slot, positionToDetect)
                    elif positionToDetect == 'buffer':
                        result, debug_img = phone_buffer_object_detector.run(path_image, path_log_id_phone_slot, positionToDetect)
                    else:
                        #result, debug_img = phone_slot_object_detector.run(path_image, path_log_id_phone_slot, positionToDetect) # phone_slot_detector.run(path_image, path_log_id_phone_slot)
                        result, debug_img = phone_slot_object_detector_triton.run(path_image, path_log_id_phone_slot, positionToDetect) # phone_slot_detector.run(path_image, path_log_id_phone_slot)
                    # Start calibration
                    # try:
                    #     self.log.info(f"[CALIBRATE] Start calibration! \n")
                    #     result_calib, debug_img_calib =  segment_plastic_box.run(path_image, path_log_id_phone_slot)
                    #     self.log.info(f"[CALIBRATE] Line1 \n")
                    #     module_info.response.result_calib = result_calib
                    #     self.log.info(f"[CALIBRATE] Line2 \n")
                    #     self.log.info("[CALIBRATE] Position calibrate= {}".format(result_calib))
                    #     cv2.imwrite(os.path.join(ml_debug_path,'debug_img_calibrate.jpg'),debug_img_calib)
                    # except:
                    #     self.log.info(f"[CALIBRATE] Can't calibration! \n")
                    
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

            elif request_type == "calibrate":
                try:
                    self.log.info("[CALIBRATE] START cmdUniqueID = {} path_image_1= {} path_image_2= {} path_debug = {}".format(cmdUniqueID,path_image_1,path_image_2,path_debug))                  
                    path_log_id_phone_slot = os.path.join(ml_debug_path,"log_id_phone_slot.txt")
                    
                    # Write original image
                    image_calib_1 = cv2.imread(path_image_1)
                    image_calib_2 = cv2.imread(path_image_2)
                    if image_calib_1 is not None:
                        cv2.imwrite(os.path.join(ml_debug_path,f'{cmdUniqueID}_calib_1.jpg'), image_calib_1)
                    else:
                        self.log.info(f"[CALIBRATE] Image calib_1 from SW is Null ==>ERROR \n")

                    if image_calib_2 is not None:
                        cv2.imwrite(os.path.join(ml_debug_path,f'{cmdUniqueID}_calib_2.jpg'), image_calib_2)
                    else:
                        self.log.info(f"[CALIBRATE] Image calib_2 from SW is Null ==>ERROR \n")
                        
                    # Start detect magnet
                    if position_to_calib == 'buffer':
                        with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                            data = json.load(json_file)
                        path_image_home_of_kangaroo = data.get('path_image_home_of_kangaroo')
                        if path_image_home_of_kangaroo is None:
                            self.log.info(f"[CALIBRATE] Image origin of kangaroo from config is Null ==>ERROR \n")
                        path_image_1 = path_image_home_of_kangaroo
                    result_calib, image_pad_debug_1, image_pad_debug_2 = calib_magnet_detector.run(path_image_1, path_image_2, position_to_calib, path_log_id_phone_slot, ml_debug_path)
                    
                    if result_calib is not None:
                        cv2.imwrite(os.path.join(ml_debug_path,'debug_img_pad_1.jpg'),image_pad_debug_1)
                        cv2.imwrite(os.path.join(ml_debug_path,'debug_img_pad_2.jpg'),image_pad_debug_2)
                    else:
                        self.log.info(f"[CALIBRATE] Image debug calib is Null ==>ERROR \n")
                        
                    self.log.info(f"[CALIBRATE] Result calib = {result_calib}")
                    # Post process result
                    if result_calib is not None:
                        self.log.info("[CALIBRATE] END cmdUniqueID = {} result = {} ml_debug_path = {}".format(cmdUniqueID,result_calib,ml_debug_path))
                        module_info.response.result_calib = result_calib
                        module_info.response.result_string = 'calib_done'
                    else:
                        module_info.response.result_string = 'cant_calib'
                        self.log.info("[DETECT_MAGNET] END cmdUniqueID = {} result = {} ml_debug_path = {}".format(cmdUniqueID,result_calib,ml_debug_path))
                    
                    # Copy log file total to debug folder
                    path_backup_log_phone_slot = os.path.join(ml_debug_path,"log_phone_slot.log")
                    cmd = f"cp -rf {os.path.join(GV().path_to_logs_folder,'log_file.log')} {path_backup_log_phone_slot}"
                    self.log.info(f"[DETECT_MAGNET] cmd backup log: {cmd}")
                    os.system(cmd)        
                except Exception as e:
                    self.log.info("[DETECT_MAGNET] FAIL to execute cmdUniqueID = {} path_image = {} path_debug = {} Error = {}".format(cmdUniqueID,path_image,path_debug,e))
                    self.log.exception(e)

            
            # if request type is scan barcode
            elif request_type == "scan_barcode" and container_scan_barcode_is_running:
                
                # Create log file
                log_file_path = os.path.join(ml_debug_path,"log_id_scan_barcode.txt")
                text_log_file = open(log_file_path, 'w')
                
                msg = f"START cmdUniqueID = {cmdUniqueID} \n path_to_list= {path_to_list} \n path_debug = {path_debug}\n"
                log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                    
                try:
                    self.log.info("[SCAN_BARCODE] START cmdUniqueID = {} \n path_to_list= {} \n path_debug = {}".format(cmdUniqueID,path_to_list,path_debug))
                    
                    # initialize
                    image = None # Image to scan barcode
                    result_scan_barcode = None
                    isImageScanBarcode = False
                    output_path =  path_to_result #os.path.join(path_debug,'output_scan_barcode.json')
                    
                    # Read file path_to_list
                    with open(path_to_list,'r') as f:
                        current_image_path = f.readline()
                        current_image_path = current_image_path.strip()
                        msg = f"Read file path_to_list = {path_to_list} sucessful, current_image_path = {current_image_path}\n"
                        log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                        self.log.info("[SCAN_BARCODE] START cmdUniqueID = {} path_to_list= {} path_debug = {} current_image = {}".format(cmdUniqueID, path_to_list, path_debug, current_image_path))
                        # Read current image
                        image = cv2.imread(current_image_path)                       
                        if image is None:
                            msg = f"Read Image scan barcode from SW is Null ==>ERROR \n"
                            log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                            self.log.info(f"[SCAN_BARCODE] Image scan barcode from SW is Null ==>ERROR \n")
                        else:
                            # Write debug original image
                            isImageScanBarcode = True
                            cv2.imwrite(os.path.join(ml_debug_path,cmdUniqueID + '_scan_barcode.jpg'), image)
                            msg= f"Read Image scan barcode sucessful\n"
                            log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                            self.log.info(f"[SCAN_BARCODE] Read Image scan barcode sucessful\n")
                    
                    # if there is image start to scan barcode
                    if isImageScanBarcode == True:
                        current_cmd = '''docker exec docker_barcode python3.7 {} {} {}'''.format(script_scan_barcode_path, path_to_list, output_path)
                        msg = f"CMD= {current_cmd}\n"
                        log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                        self.log.info("[SCAN_BARCODE] CMD= {}".format(current_cmd))
                        
                        # run command scan barcode
                        os.system(current_cmd)
                        msg = f"Run command scan barcode sucessful\n"
                        log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                        self.log.info("[SCAN_BARCODE] END cmdUniqueID = {} path_to_list= {} path_debug = {}".format(cmdUniqueID,path_to_list,path_debug))
                        module_info.response.result_string = output_path # update result to result_string
                        
                        # Read file json and draw debug file
                        if os.path.exists(output_path) and image is not None:
                            # Read file json
                            with open(output_path,'r') as f:
                                result_scan_barcode = json.load(f)
                                msg= f"Result scan barcode= " + str(result_scan_barcode) + "\n"
                                log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                                
                            msg = f"Read file json and draw debug file sucessful\n"
                            log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                            
                            # Start draw debug
                            image = draw_debug_image(image,output_path)
                            
                            # Write debug image
                            cv2.imwrite(os.path.join(ml_debug_path,cmdUniqueID + '_scan_barcode_debug.jpg'), image)
                            self.log.info(f"[SCAN_BARCODE] Write debug Image scan barcode sucessfull\n")
                                    
                        else:
                            self.log.info(f"[SCAN_BARCODE] Read file json or image is not exists ==> ERROR from SW \n")
                            msg = f"Read file json or image is not exists ==> ERROR from SW \n"
                            log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                        
                        cmd = f"cp -rf {output_path} {ml_debug_path}"                    
                        self.log.info(f"[SCAN_BARCODE] cmd backup result: {cmd}")
                        os.system(cmd)                    
                    else:
                        self.log.info(f"[SCAN_BARCODE] isImageScanBarcode false ==> ERROR from SW \n")
                        module_info.response.result_string = 'error_scan_barcode' # update result to result_string
                    
                    path_backup_log_scan_barcode = os.path.join(ml_debug_path,"log_scan_barcode.log")
                    cmd = f"cp -rf {os.path.join(GV().path_to_logs_folder,'log_file.log')} {path_backup_log_scan_barcode}"                    
                    self.log.info(f"[SCAN_BARCODE] cmd backup log: {cmd}")
                    os.system(cmd)                                        
                except Exception as e:
                    module_info.response.result_string = 'error_scan_barcode' # update result to result_string
                    self.log.info("[SCAN_BARCODE] FAIL to cmdUniqueID = {} execute path_to_list = {} path_debug = {} Error = {}".format(
                        cmdUniqueID,path_to_list,path_debug,e))
                    
                    msg = f"FAIL to cmdUniqueID = {cmdUniqueID} execute path_to_list = {path_to_list} path_debug = {path_debug} Error = {e}\n"
                    log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                    
                    # Close log file
                    text_log_file.close()
                    
                # Close log file
                msg = f"END cmdUniqueID = {cmdUniqueID} \n path_to_list= {path_to_list} \n path_debug = {path_debug}\n"
                log_msg(msg, "[SCAN_BARCODE]", text_log_file)
                text_log_file.close()

            # Write down data after preprocess
            self.log.info("Mode analyze = {}".format(request_type))
            module_info.response.result = 0
            self.log.info("module_info.response.result = {}".format(module_info.response.result))
            self.signal_analyze_finished.emit(module_info)
            self._lock_request_list.lock()
            
            for info in self._analyze_list[:]:
                if info.cmdUniqueID == module_info.cmdUniqueID:
                    self._analyze_list.remove(info)
            self._lock_request_list.unlock()
            
        print("End thread MainController...")
