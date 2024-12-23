import copy
import json
import logging
import multiprocessing
import threading
import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QThread, pyqtSignal, pyqtSlot

from globals import GV, ML_CMD_HANDSHAKE_REQUEST, STRUCT_ANALYZE_INFO, ML_APP_CMD_KEY_HELLO
from modules.general_model import GeneralModel
from modules.physical_inventory_fast_mode.model_object_detect_phone_slot import PhoneSlotObjectDetector
from modules.physical_inventory_fast_mode.physical_inventory_detect_phone_slot import Object_Detection_Physical_Inventory_YOLOv8
from modules.physical_inventory_fast_mode.get_image_and_offset_from_socket_UDP import GetImageAndOffsetFromSocketUDP

os.environ['CUDA_VISIBLE_DEVICES']='-1'

class AnalyzeControllerPIFastmode(QThread):
    signal_analyze_finished = pyqtSignal(object)

    def __init__(self, general_model: GeneralModel):
        super().__init__()
        self._exit_flag = False
        self._exit_physical_inventory_flag = False
        self._handshake_physical_inventory_flag = False
        self._general_model = general_model
        self.log = logging.getLogger(__name__)
        self._handshake_info = ML_CMD_HANDSHAKE_REQUEST()
        self._analyze_list = list()
        self._lock_request_list = QMutex()

    def append_analyze_info(self, _analyze_info: STRUCT_ANALYZE_INFO):
        self._lock_request_list.lock()
        self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] append_analyze_info analyze_info.cmdUniqueID = {}".format(_analyze_info.cmdUniqueID))
        self._analyze_list.append(_analyze_info)
        self._lock_request_list.unlock()

    def stop_analyze(self):
        self._lock_request_list.lock()
        self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] stop_analyze")
        for idx in range(len(self._analyze_list)):
            analyze_info = self._analyze_list[idx]
            analyze_info.cancelFlag = True
            self._analyze_list[idx] = analyze_info
        self._exit_flag = True
        self._lock_request_list.unlock()

    def handshake_physical_inventory(self, module_info):
        module_info.response.result_string = ML_APP_CMD_KEY_HELLO
        module_info.response.result = 1
        self.log.info("module_info.response.result = {}".format(module_info.response.result))
        self.signal_analyze_finished.emit(module_info)
        self._handshake_physical_inventory_flag = True

    def stop_physical_inventory(self):
        self._lock_request_list.lock()
        for idx in range(len(self._analyze_list)):
            analyze_info = self._analyze_list[idx]
            analyze_info.cancelFlag = True
            self._analyze_list[idx] = analyze_info
        self._exit_physical_inventory_flag = True
        self.log.info(f"[ANALYZE_CONTROLER_PI_FASTMODE] signal stop_physical_inventory(self): {self._exit_physical_inventory_flag}")
        self._lock_request_list.unlock()

    def set_handshake_info(self, request: ML_CMD_HANDSHAKE_REQUEST):
        self._handshake_info = request

    @pyqtSlot()
    def run(self):
        
        print("Start thread MainController...")
        
        # START LOAD MODEL
        # TODO Initalize
        
        model_object_detect_phone_slot_triton_weight_path = "./model_object_detect_phone_slot.onnx"
        labelmap_path_model_object_detect_phone_slot_triton = "/home/greystone/StorageWall/model_template/Model_Triton/model_object_detect_phone_slot/label.pbtxt"
        model_object_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/model_object_detect_phone_slot.onnx'

        positionToDetect = 'left'
        
        # load model detect phone slot
        try:
            phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.65)
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Load model successfuly!')
        except Exception as e:
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] FAIL to Load model detect phone slot, Error = {e}')
        
        # load model detect phone slot triton
        try:
            os.system('echo greystone | sudo -S docker run -id --gpus device=0 --rm -p:8000:8000 -p:8001:8001 -p:8002:8002 -v /home/greystone/StorageWall/model_template/Model_Triton:/models --name triton_server nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --strict-model-config=false --model-control-mode=explicit')
            model_object_detect_phone_slot_triton_yolov8 = Object_Detection_Physical_Inventory_YOLOv8(model_object_detect_phone_slot_triton_weight_path, labelmap_path_model_object_detect_phone_slot_triton, 0.65)
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Load model model_object_detect_phone_slot_triton_yolov8 successfuly!')
        except Exception as e:
            self.log.exception(f'[ANALYZE_CONTROLER_PI_FASTMODE] FAIL to Load model_object_detect_phone_slot_triton_yolov8, Error = {e}')
    
        # Warm up to load triton models
        try:
            is_model_triton_ready = model_object_detect_phone_slot_triton_yolov8._load_model_ready()
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] is_model_triton_ready = {is_model_triton_ready}')
        except:
            self.log.error(f'[ANALYZE_CONTROLER_PI_FASTMODE] Failed to load model triton!, Exception = {e}')
        
        # All queue of subprocess
        # -- Share lock
        share_buffer_image_lock = multiprocessing.Lock()
        share_buffer_image_result_lock = multiprocessing.Lock()
        share_buffer_offset_lock = multiprocessing.Lock()
        # -- Share image
        share_buffer_image_queue = multiprocessing.Queue(2)
        share_buffer_image_result_queue = multiprocessing.Queue()
        # -- Share data
        share_buffer_offset_queue = multiprocessing.Queue(2)
        count_fail = multiprocessing.Value('i',0)
        count_total = multiprocessing.Value('i',0)
        fps_get_img = multiprocessing.Value('i', 0)
        position_to_detect = multiprocessing.Value('i', 0)
        # -- Event
        pause_task = multiprocessing.Value('i', 0)
        
        time_cur = int(time.time())
        time_in_ctrl_PI = multiprocessing.Value('i', time_cur)

        data = np.zeros((54*24),np.uint8)

        status_phone_slot_left_wall = multiprocessing.Array('i',data)
        status_phone_slot_right_wall = multiprocessing.Array('i',data)
        count_status_phone_slot_left_wall = multiprocessing.Array('i',data)
        count_status_phone_slot_right_wall = multiprocessing.Array('i',data)
        
        try:
            get_image_and_offset_from_socket_UDP = GetImageAndOffsetFromSocketUDP()
            process_get_image_and_offset = multiprocessing.Process(target = get_image_and_offset_from_socket_UDP.get_image_and_offset_streaming, args = (pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_queue, share_buffer_offset_queue, count_total, count_fail, fps_get_img,))
            process_get_image_and_offset.daemon = True #Allows the process to exit when the main program exits
            process_get_image_and_offset.start()
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Start process_get_image_and_offset from socket successfuly!')
        except Exception as e:
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] Error process_get_image_and_offset from socket, ERROR = {e}')
        
        #Start subprocess analyze image
        try:
            process_physical_inventory_fastmode = multiprocessing.Process(target = model_object_detect_phone_slot_triton_yolov8.run, args = (pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_result_lock, share_buffer_image_queue, position_to_detect, share_buffer_offset_queue,count_status_phone_slot_left_wall,
                                                                                                                                    count_status_phone_slot_right_wall, status_phone_slot_left_wall, status_phone_slot_right_wall, share_buffer_image_result_queue,))
            process_physical_inventory_fastmode.daemon = True
            process_physical_inventory_fastmode.start()
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Start subprocess physical inventory detect phone slots GPU successfuly!')
        except Exception as e:
            self.log.info(f'[ANALYZE_CONTROLER_PI_FAST_MODE] Error start subprocess physical inventory detect phone slots, ERROR = {e}')
        
        try:
            process_physical_inventory = multiprocessing.Process(target = phone_slot_object_detector.do_physical_inventory_fastmode, args = (pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_result_lock, share_buffer_image_queue, position_to_detect, share_buffer_offset_queue,count_status_phone_slot_left_wall,
                                                                                                                                    count_status_phone_slot_right_wall, status_phone_slot_left_wall, status_phone_slot_right_wall, share_buffer_image_result_queue,))
            process_physical_inventory.daemon = True
            #process_physical_inventory.start()
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Start subprocess physical inventory detect phone slots CPU successfuly!')
        except Exception as e:
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] Error start subprocess physical inventory detect phone slots, ERROR = {e}')

        # END LOAD MODEL
        while not self._exit_flag:
            
            time.sleep(0.001)
            pause_task.value = 1
            time_in_ctrl_PI.value = int(time.time())
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
            path_debug = module_info.request.pathToDebug # Path debug from SW
            positionToDetect = module_info.request.positionToDetect # Position to detect from SW, the left wall or the right wall
            transaction_id = module_info.request.transactionID # TransactionID
            
            if positionToDetect == 'left':
                position_to_detect.value = 0
            elif positionToDetect == 'right':
                position_to_detect.value = 1
            elif positionToDetect == 'buffer':
                position_to_detect.value = 2
            
            # Create folder debug
            self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] -------------------------- INIT Version 1.04 --------------------------')
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] TransactionID = {transaction_id}, commandID = {cmdUniqueID}, request_type = {request_type}, positionToDetect = {positionToDetect}')

                
            # init path debug by command id
            ml_debug_path = os.path.join(GV().path_to_debug_folder, cmdUniqueID)
            
            # if transaction_id
            if transaction_id != "":                
                # Create transaction ID folder
                transaction_id_path = os.path.join(GV().path_to_debug_folder,transaction_id)
                if not os.path.exists(transaction_id_path):
                    os.makedirs(transaction_id_path)
                    self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Created Folder TransactionID sucessful ==> Go to update ml_debug_path!\n')
                else:
                    self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Folder TransactionID debug already exits!\n')
                
                # update ml_debug_path 
                ml_debug_path = os.path.join(GV().path_to_debug_folder,transaction_id,cmdUniqueID) # update ml_debug_path when transaction_ID exists!!
            
            # Create folder debug
            if not os.path.exists(ml_debug_path):
                os.makedirs(ml_debug_path)
                self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Created Folder debug sucessful!\n')
            else:
                self.log.info('[ANALYZE_CONTROLER_PI_FASTMODE] Folder debug already exits!\n')
                
            self.log.info(f'[ANALYZE_CONTROLER_PI_FASTMODE] Folder debug final= {ml_debug_path}')           
            
            output_path = None
            positionToDetect_cur = positionToDetect
            status_phone_slot_left_wall_array = None
            status_phone_slot_right_wall_array = None
            status_phone_slot_left_wall_to_list = None
            status_phone_slot_right_wall_to_list = None
            
            if request_type == "physical_inventory":
                self.handshake_physical_inventory(module_info)
                pause_task.value = 0
                self._exit_physical_inventory_flag = False
                try:
                    self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] START cmdUniqueID = {} path_debug = {} _exit_flag = {}".format(cmdUniqueID,path_debug, self._exit_flag))
                    count = 0
                    
                    while not self._exit_physical_inventory_flag:    
                        time.sleep(0.001)
                        time_in_ctrl_PI.value = int(time.time())
                        #self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] time_in_ctrl_PI.value = {}".format(int(time.time())))
                        # Post process result
                        #self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] share_buffer_image_result_queue.empty() = {}".format(share_buffer_image_result_queue.empty()))
                        if not share_buffer_image_result_queue.empty():
                            print("[ANALYZE_CONTROLER_PI_FASTMODE] time_in_ctrl_PI = Have image in result queue")
                            self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] time_in_ctrl_PI = Have image in result queue")
                            count += 1
                            
                            share_buffer_image_result_lock.acquire()
                            image_debug = share_buffer_image_result_queue.get()
                            share_buffer_image_result_lock.release()
                            
                            image_debug_path = os.path.join(ml_debug_path, f'image_debug_{count}.jpg')
                            cv2.imwrite(image_debug_path, image_debug)
                            if positionToDetect_cur == 'left':
                                status_phone_slot_left_wall_array = np.array(status_phone_slot_left_wall).reshape((54,24))
                            elif positionToDetect_cur == 'right':
                                status_phone_slot_right_wall_array = np.array(status_phone_slot_right_wall).reshape((54,24))
                            self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] Update status phone slot to argument")
                    
                except Exception as e:
                    self.log.exception("[ANALYZE_CONTROLER_PI_FASTMODE] FAIL to execute cmdUniqueID = {} path_debug = {} Error = {}".format(cmdUniqueID,path_debug,e))
            
            self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] self._exit_flag = {}, self._exit_physical_inventory_flag = {}".format(self._exit_flag, self._exit_physical_inventory_flag))
            self._handshake_physical_inventory_flag = False
            
            try:
                if (status_phone_slot_left_wall_array is not None) or (status_phone_slot_right_wall_array is not None):
                    self.log.info(f"[ANALYZE_CONTROLER_PI_FASTMODE] positionToDetect_cur = {positionToDetect_cur}")
                    output_path = f'/home/greystone/StorageWall/ML_Analysis/detectPhoneOnSlots/result/physical_inventory_{positionToDetect_cur}.json'
                    with open(output_path, "w") as json_file:
                        if positionToDetect_cur == 'left':
                            status_phone_slot_left_wall_to_list = status_phone_slot_left_wall_array.tolist()
                            json.dump(status_phone_slot_left_wall_to_list, json_file)
                        elif positionToDetect_cur == 'right':
                            status_phone_slot_right_wall_to_list = status_phone_slot_right_wall_array.tolist()
                            json.dump(status_phone_slot_right_wall_to_list, json_file)
                    self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] Successed to dump status of phone slots")
            except Exception as e:
                self.log.exception("[ANALYZE_CONTROLER_PI_FASTMODE] Failed to dump status of phone slots, ERROR = {}".format(e))
                
            # Write down data after preprocess
            self.log.info("Mode analyze = {}".format(request_type))
            module_info.response.result_string = output_path
            module_info.response.result = 0
            self.log.info("module_info.response.result = {}".format(module_info.response.result))
            self.signal_analyze_finished.emit(module_info)
            self._lock_request_list.lock()
            
            for info in self._analyze_list[:]:
                if info.cmdUniqueID == module_info.cmdUniqueID:
                    self._analyze_list.remove(info)
            self._lock_request_list.unlock()
                                                                                                                                                                                                     
        try:
            process_get_image_and_offset.terminate()
            process_physical_inventory.terminate()
            process_physical_inventory_fastmode.terminate()
            self.log.info("[ANALYZE_CONTROLER_PI_FASTMODE] Success to terminate() process ")
        except Exception as e:
            self.log.exception("[ANALYZE_CONTROLER_PI_FASTMODE] Fail to terminate() process = {} ".format(e))
        print("End thread MainController...")
