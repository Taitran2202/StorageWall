import json
import logging
import multiprocessing
import os
import time

import cv2
import cv2.dnn
import numpy as np

from globals import GV

# Inits
CONF_THRESHOLD = 0.8
CLASSES = ['phone','gripper'] # yaml_load(check_yaml("coco128.yaml"))["names"]
images_tail = ['png','jpg','jpeg']

class PhoneSlotObjectDetector(multiprocessing.Process):
    
    def __init__(self, model_path, CONF_THRESHOLD) -> None:
        multiprocessing.Process.__init__(self)
        
        self.log = logging.getLogger('PhoneSlotObjectDetector')
        self.log.setLevel(logging.DEBUG)
        self.log_file_path = os.path.join(GV().path_to_logs_folder, "PhoneSlotObjectDetector.log")
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%y-%m-%d %H:%M:%S:%f")
        self.file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.file_handler)
        
        self.exit_flag = False
        self.W = 1200
        self.H = 600
        self.thickness = 5
        self.match_color = (0,255,100)
        self.GRIPPER_MIN_DIM = 100 # minimum dimension of gripper object
        self.GRIPPER_MAX_DIM = 400 # maximum dimension of gripper object
        self.PHONE_MIN_DIM = 20 # minimum dimension of phone object
        self.model_path = model_path
        self.CONF_THRESHOLD =CONF_THRESHOLD
        self.CLASSES = ['phone']
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.model = cv2.dnn.readNetFromONNX(model_path) # Load the ONNX model
        self.result_have_phone = 'have_phone'
        self.result_none_phone = 'none_phone'
        self.tag_log = '[model_object_detect_PI]'
        self.flag = 0
        self.quantity_row = 54
        self.quantity_column = 24
        self.offset_x_min = 0
        self.offset_y_min = 0
        self.offset_x_max = 3280
        self.offset_y_max = 2455
        self.side = ''
        self.path_log_txt = '/home/greystone/StorageWall/logs/ml_logs/model_object_detect_PI.txt'
        
    def find_top_left_and_bottom_right(self, points):
        points_array = np.array(points)
        top_left = np.min(points_array, axis=0)
        bottom_right = np.max(points_array, axis=0)
        return top_left, bottom_right
    
    def find_diagonal_line_of_box(self, points):
        points = np.array(points)
        x, y, w, h = points
        top_left = [x,y]
        bottom_right = [x + w, y + h]
        return top_left, bottom_right
    
    def find_top_left_and_top_right(self, points):
        # Find the top left point
        zero_point = (0,0)
        distances = [np.sqrt((p[0]-zero_point[0])**2 + (p[1]-zero_point[1])**2) for p in points]
        top_left_index = np.argmin(distances)
        top_left = points[top_left_index]
        # Find the top right point
        coord_point = (self.W, 0)
        distances = [np.sqrt((p[0]-coord_point[0])**2 + (p[1]-coord_point[1])**2) for p in points]
        top_right_index = np.argmin(distances)
        top_right = points[top_right_index]
        return top_left, top_right
        
    def find_min_max(self, dim1, dim2):
        if dim1 > dim2:
            return dim2, dim1
        else:
            return dim1, dim2
      
    def find_top_left_and_bottom_right(self, mask):
        # Find non-zero elements in the mask
        non_zero_points = cv2.findNonZero(mask)
        
        if non_zero_points is None:
            return None, None  # If no non-zero points found, return None for both
        
        # Extract x and y coordinates from the non-zero points
        x_coordinates = [point[0][0] for point in non_zero_points]
        y_coordinates = [point[0][1] for point in non_zero_points]
        
        # Find the minimum and maximum x and y coordinates
        min_x = min(x_coordinates)
        max_x = max(x_coordinates)
        min_y = min(y_coordinates)
        max_y = max(y_coordinates)
        
        # Create top-left and bottom-right points
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        
        return top_left, bottom_right

    def line_equation(self, x, y, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

    def bounding_box_crosses_line(self, bbox, line_start, line_end):
        # Extract the coordinates of the bounding box
        x, y, w, h = bbox

        # Define the four corners of the bounding box
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            if (self.line_equation(p1[0], p1[1], line_start, line_end) *
                self.line_equation(p2[0], p2[1], line_start, line_end) < 0):
                return True
        return False

    def crop_img(self, image, x1, y1, x2, y2):
        #image_crop = image[10:840, 90:1130, :]
        image_crop = image[x1:y1, x2:y2, :]
        #image_crop = cv2.resize(image_crop,(1200, 600))
        return image_crop

    def draw_arena_box(self, image, x_from, y_from, x_to, y_to, color_bgr, size=0.005, line_type=cv2.LINE_AA, is_copy=True):
        assert size > 0
        image = image.copy() if is_copy else image # copy/clone a new image
        # calculate thickness
        h, w = image.shape[:2]
        if size > 0:        
            short_edge = min(h, w)
            thickness = int(short_edge * size)
            thickness = 1 if thickness <= 0 else thickness
        else:
            thickness = -1
        # calc x,y in absolute coord
        cv2.rectangle(img=image, pt1=(x_from, y_from), pt2=(x_to, y_to), color=color_bgr, thickness=thickness, lineType=line_type, shift=0)
        #location = (x,y,w,h)
        location = (x_from, y_from, x_to - x_from, y_to - y_from)
        return image, location

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f"{CLASSES[class_id]} ({confidence:.2f})"
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
    
    def find_position_phone_slot(self, offset_x_cur, offset_y_cur):
        step_x = (self.offset_x_max - self.offset_x_min)/(self.quantity_column - 1)
        step_y = (self.offset_y_max - self.offset_y_min)/(self.quantity_row - 1)
        
        delta_slot_x = int(step_x / 3)
        delta_slot_y = int(step_y / 3)
        
        slot_x_cur = round((offset_x_cur + step_x)/step_x)
        slot_y_cur = round((offset_y_cur + step_y)/step_y)
        
        delta_slot_x_cur_in_mm = offset_x_cur - ((slot_x_cur-1)*step_x)
        delta_slot_y_cur_in_mm = offset_y_cur - ((slot_y_cur-1)*step_y)
        
        delta_slot_x_cur_in_pixel = round((delta_slot_x_cur_in_mm / step_x) * 805)
        delta_slot_y_cur_in_pixel = round((delta_slot_y_cur_in_mm / step_y) * 215)
        
        return slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel
    
    def check_area_detect(self, x_from, x_to, y_from, y_to, delta_slot_x_cur_in_pixel,delta_slot_y_cur_in_pixel):
        x_from = x_from + delta_slot_x_cur_in_pixel
        x_to = x_to + delta_slot_x_cur_in_pixel
        y_from = y_from + delta_slot_y_cur_in_pixel
        y_to = y_to + delta_slot_y_cur_in_pixel
        
        if x_from < 0:
            x_from = 0
        if x_to > 1200:
            x_to = 1200
        if y_from < 0:
            y_from = 0
        if y_to > 600:
            y_to = 600
        return x_from, x_to, y_from, y_to

    def detect(self, original_image):
        
        # Write the text to the file
        self.log.info('In function detect!')
        
        # Init
        boxes = []
        scores = []
        class_ids = []
        detections = [] # [{'class_id':, 'class_name':, 'confidence':, 'box':[], 'scale':},...]
        object_list = []
        
        # with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
        #     data = json.load(json_file)
        
        # x1 = data.get('x1', 10)
        # y1 = data.get('y1', 840)
        # x2 = data.get('x2', 90)
        # y2 = data.get('y2', 1130)
        #original_image = self.crop_img(original_image, x1, y1, x2, y2)
        
        if original_image is None:
            self.log.info('Error: Image not found!')
            return object_list
        
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()
        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        
        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.75:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            # init
            current_object = {}
            
            index = result_boxes[i]
            box = boxes[index]
            
            detection = {
                "class_id": class_ids[index],
                "class_name": self.CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            
            # get object
            x = round(box[0] * scale)
            y = round(box[1] * scale)
            x_plus_w = round((box[0] + box[2]) * scale)
            y_plus_h = round((box[1] + box[3]) * scale)
            w = x_plus_w - x
            h = y_plus_h - y
            
            current_object['label'] = self.CLASSES[class_ids[index]]
            current_object['conf'] = scores[index]
            current_object['x'] = x
            current_object['y'] = y
            current_object['w'] = w
            current_object['h'] = h
            
            if scores[index] > self.CONF_THRESHOLD:
                object_list.append(current_object)
            
                # draw debug
                self.draw_bounding_box(
                    original_image,
                    class_ids[index],
                    scores[index],
                    x,
                    y,
                    x_plus_w,
                    y_plus_h,
                )
        
        return object_list
    
    def post_process(self, object_list, debug_img, positionToDetect, position_x_cur, position_y_cur):
        
        # Write the text to the file
        self.log.info('In function post-proceess!')
        
        # init
        result = self.result_none_phone # 'have_phone', 'none_phone'
        
        # check gripper exist if not there is no phone
        phone_list = []     

        # check the postion of phone in slot
        if positionToDetect == 'left':
            self.log.info('Open file config for detect the phone in the left wall!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_left_wall_from', 350)
            y_from = data.get('y_detect_phone_left_wall_from', 180)
            x_to = data.get('x_detect_phone_left_wall_to', 900)
            y_to = data.get('y_detect_phone_left_wall_to', 320)
        elif positionToDetect == 'right':
            self.log.info('Open file config for detect the phone in the right wall!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_right_wall_from', 315)
            y_from = data.get('y_detect_phone_right_wall_from', 190)
            x_to = data.get('x_detect_phone_right_wall_to', 775)
            y_to = data.get('y_detect_phone_right_wall_to', 315)
        elif positionToDetect == 'buffer':
            self.log.info('Open file config for detect the phone in the buffer!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_kangaroo_from', 300)
            y_from = data.get('y_detect_phone_kangaroo_from', 30)
            x_to = data.get('x_detect_phone_kangaroo_to', 900)
            y_to = data.get('y_detect_phone_kangraroo_to', 455)
        elif positionToDetect == 'input_dock':
            self.log.info('Open file config for detect the phone in the Input dock!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_input_dock_from', 1450)
            y_from = data.get('y_detect_phone_input_dock_from', 515)
            x_to = data.get('x_detect_phone_input_dock_to', 2750)
            y_to = data.get('y_detect_phone_input_dock_to', 2815)
        else:
            self.log.info('ERROR: Require key positionToDetect from SW!')
        
        slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel = self.find_position_phone_slot(position_x_cur, position_y_cur)
        print(slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel)
        
        x_from, x_to, y_from, y_to = self.check_area_detect(x_from, x_to, y_from, y_to, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel)

        debug_img, location_arena_detect = self.draw_arena_box(debug_img, x_from, y_from, x_to, y_to, color_bgr = [0, 0, 255], is_copy = True)

        if object_list is None:
            self.log.info('object_list is empty. Return no_phone')
            return result, slot_x_cur, slot_y_cur, debug_img

        for current_object in object_list:
            # get name of the object
            object_label = current_object['label']
            w = current_object['w']
            h = current_object['h']
            y = current_object['y']
            x = current_object['x']
            min_dim, max_dim = self.find_min_max(w,h)
            
            # check object is phone or gripper
            if object_label == 'phone':
                # check minimum dimension of object with minimum dimension of phone
                if min_dim > self.PHONE_MIN_DIM:
                    phone_list.append(current_object)
        
        # check object detected
        
        print("phone_list",phone_list)
        if len(phone_list) == 0:
            self.log.info('phone_list is empty. Return no_phone')
            return result, slot_x_cur, slot_y_cur, debug_img

        is_has_phone = False
        overlap_percentage_target = 0.00
        img_check_area = np.zeros((debug_img.shape[0], debug_img.shape[1], 1), dtype=np.uint8)
        cv2.rectangle(img_check_area, (location_arena_detect[0], location_arena_detect[1]), (location_arena_detect[0] + location_arena_detect[2], location_arena_detect[1] + location_arena_detect[3]), [255], -1)
        
        for phone in phone_list:
            # check position of phone cross the line
            bbox = (phone['x'], phone['y'], phone['w'], phone['h'])
            cur_img = np.zeros((debug_img.shape[0], debug_img.shape[1], 1), dtype=np.uint8)
            cv2.rectangle(cur_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), [255], -1)
            img_dst = cv2.bitwise_and(img_check_area, cur_img)
            nonZeroVal = cv2.countNonZero(img_dst)
            overlap_percentage = (nonZeroVal * 100) / (location_arena_detect[2]*location_arena_detect[3])
            self.log.info("overlap_percentage = {}".format(overlap_percentage))
            if overlap_percentage > 25:
                overlap_percentage = np.round(overlap_percentage,2)
                overlap_percentage_target = overlap_percentage
                is_has_phone = True
                cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), [255, 0, 0], 5)
        msg = "overlap = " + str(overlap_percentage_target) + "%"
        cv2.putText(debug_img, msg, (0,200), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
        cv2.putText(debug_img, f'{self.side}R{slot_y_cur}C{slot_x_cur}', (0,250), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 2)
        
        if is_has_phone:
            result = self.result_have_phone
            self.log.info('Found overlapped phone. Return have_phone')
        else:
            result = self.result_none_phone
            self.log.info('Not found overlapped phone. Return no_phone')
            # x, y, w, h = location_arena_detect
            # line_start = [x,y]
            # line_end = [x+w,y+h]
            # top_left = (phone['x'], phone['y'])
            # bottom_right = (phone['x'] + phone['w'], phone['y'] + phone['h'])
            # is_cross_line = self.bounding_box_crosses_line(bbox, line_start, line_end)
            # if is_cross_line:
            #     cv2.rectangle(debug_img, top_left, bottom_right, self.match_color, self.thickness)
            #     result = self.result_have_phone
        
        return result, slot_x_cur, slot_y_cur, debug_img
    
    def do_physical_inventory_fastmode(self, pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_result_lock, share_buffer_image_queue, position_to_detect, share_buffer_offset_queue, count_status_phone_slot_left_wall, count_status_phone_slot_right_wall, status_phone_slot_left_wall, status_phone_slot_right_wall, share_buffer_image_result_queue):
        # init logger
        while not self.exit_flag:
        
            time_start = time.time()
            try:
                # time.sleep(0.001)
                time.sleep(0)
                time_now = int(time.time())
                delta_time_cur = time_now - time_in_ctrl_PI.value
                # self.log.info(f"[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] time_in_ctrl_PI = {time_in_ctrl_PI.value}")
                if delta_time_cur > 20:
                    self.exit_flag = True
                    self.log.info("[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Stop process get_offset_streaming, self.exit_flag  = {}, delta_time_cur = {}".format(self.exit_flag, delta_time_cur))
                
                if pause_task.value == 1:
                    continue
                # self.log.info(f'[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Pause signal = {pause_task.value}')
                
                fps = 0
                if share_buffer_image_queue.empty() is True:
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Error: share_buffer_image_queue empty!')
                    time.sleep(0.1)
                    continue
                if share_buffer_offset_queue.empty() is True:
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Error: share_buffer_offset_queue empty!')
                    time.sleep(0.1)
                    continue
                    
                share_buffer_offset_lock.acquire()
                position_x_cur, position_y_cur = share_buffer_offset_queue.get()
                share_buffer_offset_lock.release()
                
                share_buffer_image_lock.acquire()
                original_image = share_buffer_image_queue.get()
                share_buffer_image_lock.release()
                
                result = self.result_none_phone
                if original_image is None:
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Error: Image not found!')
                    continue
                else:
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Image exists!')
                
                if position_to_detect.value == 0:
                    positionToDetect = 'left'
                    self.side = 'LS'
                elif position_to_detect.value == 1:
                    positionToDetect = 'right'
                    self.side = 'RS'
                elif position_to_detect.value == 2:
                    positionToDetect = 'buffer'
                    self.side = 'BB01'
                
                # Detect
                self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Start run function detect!')
                object_list = self.detect(original_image)
                self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] End run function detect!')
                
                # Print out current object
                self.log.info(f'[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Current object list: {object_list}')
                
                # post process
                self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Start run function post-process!')
                result, slot_x_cur, slot_y_cur, debug_img = self.post_process(object_list, original_image, positionToDetect, position_x_cur, position_y_cur)
                self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] End run function post-process!')
                
                self.log.info(f'[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Result = {result}')
                
                if positionToDetect == 'left':
                    if result == self.result_have_phone:
                        count_status_phone_slot_left_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] += 1
                    if count_status_phone_slot_left_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] > 2:
                        status_phone_slot_left_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] = 1

                elif positionToDetect == 'right':
                    if result == self.result_have_phone:
                        count_status_phone_slot_right_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] += 1
                    if count_status_phone_slot_right_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] > 2:
                        status_phone_slot_right_wall[((slot_y_cur-1) * 24) + slot_x_cur - 1] = 1


                time_stop = time.time()
                time_elapsed = time_stop - time_start
                fps = round(1/time_elapsed)
                
                # draw debug
                self.log.info(f'[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] fps:{fps}!')
                cv2.putText(debug_img, f'fps:{fps}',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f'{result}', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"X:{position_x_cur}, Y:{position_y_cur}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"R:{slot_x_cur}, C:{slot_y_cur}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                
                self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] share_buffer_image_result_queue.full() is {}'.format(share_buffer_image_result_queue.full()))
                if not share_buffer_image_result_queue.full():
                    share_buffer_image_result_lock.acquire()
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] in function share_buffer_image_result_queue.put(debug_img)')
                    share_buffer_image_result_queue.put(debug_img)
                    self.log.info('[PHYSICAL INVENTORY DETECT PHONE SLOT CPU] Put debug_img to share_buffer_image_result_queue successful')
                    share_buffer_image_result_lock.release()
                    
            except Exception as e:
                self.log.exception(f'Error: {e}')
                continue
        


# if __name__ == '__main__':
#     positionToDetect = 'input_dock'
#     image_path = "/home/greystone/StorageWall/image_debug/1713746941049_5_scan_barcode.jpg"
#     model_object_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/InputDock/model_object_detect_phone_input_dock.onnx'
#     ml_debug_path = "/home/greystone/StorageWall/image_debug"
#     phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.70)
#     result, debug_img = phone_slot_object_detector.run(image_path, positionToDetect)
#     cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)
