
import datetime
import json
import math
import os
import traceback

import cv2
import cv2.dnn
import numpy as np

# Inits
CONF_THRESHOLD = 0.7
CLASSES = ['magnetic_dark', 'rectangle_hole']
images_tail = ['png','jpg','jpeg']

class CalibMagnetArea:
    
    def __init__(self, model_path, CONF_THRESHOLD):
        self.W = 1200
        self.H = 600
        self.thickness = 5
        self.match_color = (0,255,100)
        self.MAGNET_MIN_DIM = 1 # minimum dimension of magnet object
        self.model_path = model_path
        self.CONF_THRESHOLD =CONF_THRESHOLD
        self.CLASSES = ['magnetic_dark', 'rectangle_hole']
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.model = cv2.dnn.readNetFromONNX(model_path) # Load the ONNX model
        print('[CalibMagnetArea] Load model successful!')

        self.tag_log = '[CalibMagnetArea]'
    
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
        
    def init_csv_file(self, text_log_file):
        file_path = '/home/greystone/StorageWall/apps/ml-subapp/appPython/calib.csv'
        if os.path.exists(file_path):
            self.log_msg(f"File '{file_path}' exists", text_log_file)
        else:
            open(file_path, 'w').close()
            self.log_msg(f"File '{file_path}' do not exists, created a new file!", text_log_file)
            
        
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
    
    def find_bottom_left_min(self, object_list):
        x_from = []
        x_to = []
        y_from = []
        y_to = []
        for current_object in object_list:
            w = current_object['w']
            h = current_object['h']
            x = current_object['x']
            y = current_object['y']
            x_from.append(x)
            x_to.append(x+w)
            y_from.append(y)
            y_to.append(y+h)
        x_min = x_from[0]
        y_min = y_to[0]
        if len(x_from) > 1:
            for i in range(len(x_from)):
                if (x_to[i] < x_min):
                    x_min = x_to[i]
                    y_min = y_to[i]
        else:
            x_min = x_from[0]
            y_min = y_to[0]
        return x_min, y_min
        
    def find_bottom_right_max(self, object_list):
        x_from = []
        x_to = []
        y_from = []
        y_to = []
        for current_object in object_list:
            object_label = current_object['label']
            w = current_object['w']
            h = current_object['h']
            y = current_object['y']
            x = current_object['x']
            x_from.append(x)
            x_to.append(x + w)
            y_from.append(y)
            y_to.append(y + h)

        x_max = x_to[0]
        y_max = y_to[0]
        if len(x_to) > 1:
            for i in range(len(x_to)):
                if (x_to[i] > x_max):
                    x_max = x_to[i]
                    y_max = y_to[i]
        else:
            x_max = x_to[0]
            y_max = y_to[0]
        return x_max, y_max
    
    def find_bottom_right_max_v2(self, object_list):
        x_from = []
        x_to = []
        y_from = []
        y_to = []
        for current_object in object_list:
            object_label = current_object['label']
            w = current_object['w']
            h = current_object['h']
            y = current_object['y']
            x = current_object['x']
            center_x = x + w/2
            center_y = y + h/2
            
            x_from.append(x)
            x_to.append(x + w)
            y_from.append(y)
            y_to.append(y + h)

        x_max = x_to[0]
        y_max = y_to[0]
        if len(x_to) > 1:
            for i in range(len(x_to)):
                if (x_to[i] > x_max):
                    x_max = x_to[i]
                    y_max = y_to[i]
        else:
            x_max = x_to[0]
            y_max = y_to[0]
        return x_max, y_max
    
    def load_coordinates_to_crop(self, position):
        with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
            data = json.load(json_file)
        if position == 'left':
            x_from_1 = 36
            x_to_1 = data.get('x_to_wall_left_1', 396)
            y_from_1 = data.get('y_from_wall_left_1', 260)
            y_to_1 = data.get('y_to_wall_left_1', 420)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_wall_left_2', 800) 
            x_to_2 = data.get('x_to_wall_left_2', 1120)
            y_from_2 = data.get('y_from_wall_left_2', 260)
            y_to_2 = data.get('y_to_wall_left_2', 420)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]

        elif position == 'right':
            x_from_1 = 10
            x_to_1 = 330
            y_from_1 = data.get('y_from_wall_right_1', 207)
            y_to_1 = data.get('y_to_wall_right_1', 367)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = 737
            x_to_2 = 1057
            y_from_2 = data.get('y_from_wall_right_2', 191)
            y_to_2 = data.get('y_to_wall_right_2', 351)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            
        elif position == 'buffer':
            x_from_1 = data.get('x_from_kangaroo_1',100)
            x_to_1 = data.get('x_to_kangaroo_1',420)
            y_from_1 = data.get('y_from_kangaroo_1',420)
            y_to_1 = data.get('y_to_kangaroo_1',580)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_kangaroo_2',800)
            x_to_2 = data.get('x_to_kangaroo_2',1120)
            y_from_2 = data.get('y_from_kangaroo_2',415)
            y_to_2 = data.get('y_to_kangaroo_2',575)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            
        return position_crop_magnet_left,position_crop_magnet_right
              
        
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

    def crop_img(self, image_path, position_crop_magnet_left, position_crop_magnet_right):
        #image_crop = image[10:840, 90:1130, :]
        x_from_left, x_to_left, y_from_left, y_to_left = position_crop_magnet_left
        x_from_right, x_to_right, y_from_right, y_to_right = position_crop_magnet_right
        image = cv2.imread(image_path)
        #image_crop_left = image[x_from_left:x_to_left, y_from_left:y_to_left, :]
        image_crop_left = image[y_from_left:y_to_left, x_from_left:x_to_left, :]
        #image_crop_right = image[x_from_right:x_to_right, y_from_right:y_to_right, :]
        image_crop_right = image[y_from_right:y_to_right, x_from_right:x_to_right, :]
        return image_crop_left, image_crop_right
    
    def padding_img_debug(self, image_path_1, image_path_2, image_crop_1, image_crop_2, position_crop_magnet_left):
        image_pad_1 = cv2.imread(image_path_1)
        image_pad_2 = cv2.imread(image_path_2)
        x_from, x_to, y_from, y_to = position_crop_magnet_left
        image_pad_1[y_from:y_to, x_from:x_to, :] = image_crop_1
        image_pad_2[y_from:y_to, x_from:x_to, :] = image_crop_2
        return image_pad_1, image_pad_2
    
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
        x_from_abs = int(x_from * w)
        y_from_abs = int(y_from * h)
        x_to_abs = int(x_to * w)
        y_to_abs = int(y_to * h)
        cv2.rectangle(img=image, pt1=(x_from_abs, y_from_abs), pt2=(x_to_abs, y_to_abs), color=color_bgr, thickness=thickness, lineType=line_type, shift=0)
        #location = (x,y,w,h)
        location = (x_from_abs, y_from_abs, x_to_abs - x_from_abs, y_to_abs - y_from_abs)
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
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    def detect(self, original_image, text_log_file):
        
        # Write the text to the file
        msg = 'In function detect!\n'
        self.log_msg(msg, text_log_file)
        
        # Init
        boxes = []
        scores = []
        class_ids = []
        detections = [] # [{'class_id':, 'class_name':, 'confidence':, 'box':[], 'scale':},...]
        object_list = []
        
        # Read the input image
        
        #original_image = cv2.imread(image_path)
        
        # with open('config.json', 'r') as json_file:
        #     data = json.load(json_file)
        
        # x1 = data.get('x1', 10)
        # y1 = data.get('y1', 840)
        # x2 = data.get('x2', 90)
        # y2 = data.get('y2', 1130)
        #original_image = self.crop_img(original_image, x1, y1, x2, y2)
        
        if original_image is None:
            msg = 'Error: Image not found!\n'
            self.log_msg(msg, text_log_file)
            return object_list, None
        
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
            if maxScore >= 0.25:
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
            
            if (scores[index] > self.CONF_THRESHOLD) and (current_object['label'] == 'magnetic_dark'):
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
        
        return object_list, original_image
    
    # def post_process(self, object_list_1, object_list_2, debug_img_1, debug_img_2, text_log_file):
        
    #     # Write the text to the file
    #     msg = 'In function post-proceess!\n'
    #     self.log_msg(msg, text_log_file)
    #     return result, image_pad
    
    def run(self, image_path_1, image_path_2, position_to_calib, log_file_path, ml_debug_path, offset_reference):
        # init logger
        text_log_file = open(log_file_path, 'w')
        result_dict, std_image_debug, cur_image_debug = {}, None, None
        try:
            # Crop images
            position_crop_magnet_left, position_crop_magnet_right = self.load_coordinates_to_crop(position_to_calib)
            print(type(position_crop_magnet_left))
            print(position_crop_magnet_left)
            print(position_crop_magnet_right)
            image_1_left, image_1_right = self.crop_img(image_path_1, position_crop_magnet_left, position_crop_magnet_right)
            image_2_left, image_2_right = self.crop_img(image_path_2, position_crop_magnet_left, position_crop_magnet_right)
            self.log_msg('End run function crop!\n', text_log_file)
            
            std_image_debug = cv2.imread(image_path_1)
            cur_image_debug = cv2.imread(image_path_2)
            cv2.rectangle(std_image_debug, (position_crop_magnet_left[0], position_crop_magnet_left[2]), (position_crop_magnet_left[1], position_crop_magnet_left[3]), (0, 255, 0), 2)
            cv2.rectangle(std_image_debug, (position_crop_magnet_right[0], position_crop_magnet_right[2]), (position_crop_magnet_right[1], position_crop_magnet_right[3]), (0, 255, 0), 2)
            cv2.rectangle(cur_image_debug, (position_crop_magnet_left[0], position_crop_magnet_left[2]), (position_crop_magnet_left[1], position_crop_magnet_left[3]), (0, 255, 0), 2)
            cv2.rectangle(cur_image_debug, (position_crop_magnet_right[0], position_crop_magnet_right[2]), (position_crop_magnet_right[1], position_crop_magnet_right[3]), (0, 255, 0), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_0.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_0.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)
            
            # detect
            self.log_msg('Start run function detect!\n', text_log_file)
            object_list_1_left, debug_img_1_left = self.detect(image_1_left, text_log_file)
            object_list_1_right, debug_img_1_right = self.detect(image_1_right, text_log_file)
            
            object_list_2_left, debug_img_2_left = self.detect(image_2_left, text_log_file)
            object_list_2_right, debug_img_2_right = self.detect(image_2_right, text_log_file)
            
            std_box_left_magnetic = dict()
            x_max = 0
            for box in object_list_1_left:
                x = box['x']
                if x > x_max:
                    x_max = x
                    std_box_left_magnetic = box
                    
            std_box_right_magnetic = dict()
            x_min = image_1_right.shape[1]
            for box in object_list_1_right:
                x = box['x']
                if x < x_min:
                    x_min = x
                    std_box_right_magnetic = box
            
            cur_box_left_magnetic = dict()
            x_max = 0
            for box in object_list_2_left:
                x = box['x']
                if x > x_max:
                    x_max = x
                    cur_box_left_magnetic = box
            
            cur_box_right_magnetic = dict()
            x_min = image_2_right.shape[1]
            for box in object_list_2_right:
                x = box['x']
                if x < x_min:
                    x_min = x
                    cur_box_right_magnetic = box
            
            std_image_debug_crop_left = std_image_debug[position_crop_magnet_left[2]:position_crop_magnet_left[3], position_crop_magnet_left[0]:position_crop_magnet_left[1]]
            cv2.rectangle(std_image_debug_crop_left, (std_box_left_magnetic['x'], std_box_left_magnetic['y']), (std_box_left_magnetic['x'] + std_box_left_magnetic['w'], std_box_left_magnetic['y'] + std_box_left_magnetic['h']), (255, 0, 0), 2)
            std_image_debug_crop_right = std_image_debug[position_crop_magnet_right[2]:position_crop_magnet_right[3], position_crop_magnet_right[0]:position_crop_magnet_right[1]]
            cv2.rectangle(std_image_debug_crop_right, (std_box_right_magnetic['x'], std_box_right_magnetic['y']), (std_box_right_magnetic['x'] + std_box_right_magnetic['w'], std_box_right_magnetic['y'] + std_box_right_magnetic['h']), (0, 255, 0), 2)
            
            cur_image_debug_crop_left = cur_image_debug[position_crop_magnet_left[2]:position_crop_magnet_left[3], position_crop_magnet_left[0]:position_crop_magnet_left[1]]
            cv2.rectangle(cur_image_debug_crop_left, (cur_box_left_magnetic['x'], cur_box_left_magnetic['y']), (cur_box_left_magnetic['x'] + cur_box_left_magnetic['w'], cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']), (255, 0, 0), 2)
            cur_image_debug_crop_right = cur_image_debug[position_crop_magnet_right[2]:position_crop_magnet_right[3], position_crop_magnet_right[0]:position_crop_magnet_right[1]]
            cv2.rectangle(cur_image_debug_crop_right, (cur_box_right_magnetic['x'], cur_box_right_magnetic['y']), (cur_box_right_magnetic['x'] + cur_box_right_magnetic['w'], cur_box_right_magnetic['y'] + cur_box_right_magnetic['h']), (0, 255, 0), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_crop_left_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug_crop_left)
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_crop_right_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug_crop_right)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_1.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)
            
            std_mm_per_pixel_ratio = math.fabs(113/((position_crop_magnet_right[0] + std_box_right_magnetic['x']) - (position_crop_magnet_left[0] + std_box_left_magnetic['x'])))
            cur_mm_per_pixel_ratio = math.fabs(113/((position_crop_magnet_right[0] + cur_box_right_magnetic['x']) - (position_crop_magnet_left[0] + cur_box_left_magnetic['x'])))
            std_mm_per_pixel_ratio_rounded = np.round(std_mm_per_pixel_ratio, 6)
            cur_mm_per_pixel_ratio_rounded = np.round(cur_mm_per_pixel_ratio, 6)
            
            y_putText = int(cur_image_debug.shape[0]/3)
            x_putText = int(cur_image_debug.shape[1]/4)
            
            std_left_magnetic_center_x_in_pixel = int(std_box_left_magnetic['x'] + std_box_left_magnetic['w']/2)
            std_left_magnetic_center_y_in_pixel = int(std_box_left_magnetic['y'] + std_box_left_magnetic['h']/2)
            std_right_magnetic_center_x_in_pixel = int(std_box_right_magnetic['x'] + std_box_right_magnetic['w']/2)
            std_right_magnetic_center_y_in_pixel = int(std_box_right_magnetic['y'] + std_box_right_magnetic['h']/2)
            
            std_center_left = (std_left_magnetic_center_x_in_pixel + position_crop_magnet_left[0] , std_left_magnetic_center_y_in_pixel + position_crop_magnet_left[2])
            std_center_right = (std_right_magnetic_center_x_in_pixel + position_crop_magnet_right[0] , std_right_magnetic_center_y_in_pixel + position_crop_magnet_right[2])
            
            cv2.circle(std_image_debug, std_center_left, 5, (0, 0, 255), -1)
            cv2.circle(std_image_debug, std_center_right, 5, (0, 0, 255), -1)
            
            std_left_magnetic_center_x_in_mm = (std_box_left_magnetic['x'] + std_box_left_magnetic['w']/2) * std_mm_per_pixel_ratio
            std_left_magnetic_center_y_in_mm = (std_box_left_magnetic['y'] + std_box_left_magnetic['h']/2) * std_mm_per_pixel_ratio
            
            cur_left_magnetic_center_x_in_pixel = int(cur_box_left_magnetic['x'] + cur_box_left_magnetic['w']/2)
            cur_left_magnetic_center_y_in_pixel = int(cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']/2)
            cur_right_magnetic_center_x_in_pixel = int(cur_box_right_magnetic['x'] + cur_box_right_magnetic['w']/2)
            cur_right_magnetic_center_y_in_pixel = int(cur_box_right_magnetic['y'] + cur_box_right_magnetic['h']/2)

            cur_center_left = (cur_left_magnetic_center_x_in_pixel + position_crop_magnet_left[0], cur_left_magnetic_center_y_in_pixel + position_crop_magnet_left[2])
            cur_center_right = (cur_right_magnetic_center_x_in_pixel + position_crop_magnet_right[0], cur_right_magnetic_center_y_in_pixel + position_crop_magnet_right[2])

            cv2.circle(cur_image_debug, cur_center_left, 5, (0, 0, 255), -1)
            cv2.circle(cur_image_debug, cur_center_right, 5, (0, 0, 255), -1)

            std_center = (round((std_center_left[0] + std_center_right[0])/2) , round((std_center_left[1] + std_center_right[1])/2))
            cur_center = (round((cur_center_left[0] + cur_center_right[0])/2) , round((cur_center_left[1] + cur_center_right[1])/2))
            cv2.circle(cur_image_debug, cur_center, 5, (0, 0, 255), -1)
            cv2.circle(std_image_debug, std_center, 5, (0, 0, 255), -1)

            cv2.putText(std_image_debug, "x: {}, y: {}, r: {}".format(std_center[0], std_center[1], std_mm_per_pixel_ratio_rounded), (x_putText, y_putText), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(cur_image_debug, "x: {}, y: {}, r: {}".format(cur_center[0], cur_center[1], cur_mm_per_pixel_ratio_rounded), (x_putText, y_putText), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cur_left_magnetic_center_x_in_mm = (cur_box_left_magnetic['x'] + cur_box_left_magnetic['w']/2) * cur_mm_per_pixel_ratio
            cur_left_magnetic_center_y_in_mm = (cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']/2) * cur_mm_per_pixel_ratio

            std_center_x_in_mm = np.round(std_center[0] * std_mm_per_pixel_ratio, 6)
            std_center_y_in_mm = np.round(std_center[1] * std_mm_per_pixel_ratio, 6)
            cur_center_x_in_mm = np.round(cur_center[0] * cur_mm_per_pixel_ratio, 6)
            cur_center_y_in_mm = np.round(cur_center[1] * cur_mm_per_pixel_ratio, 6)
            flag = 0
            delta_center_x_in_mm = np.round(cur_center_x_in_mm - std_center_x_in_mm, 4)
            delta_center_y_in_mm = -np.round(cur_center_y_in_mm - std_center_y_in_mm, 4)
            if(abs(delta_center_x_in_mm) > 50 or abs(delta_center_y_in_mm) > 50):
                self.log_msg('[WARNING ON CALIBRATE TOOL] This Position has anomoly offset value: {}'.format(os.path.join(ml_debug_path, image_path_2.split('/')[-1].split('.')[0])), text_log_file)
            if(abs(delta_center_x_in_mm) > 50):
                delta_center_x_in_mm = offset_reference[0]
            if(abs(delta_center_y_in_mm) > 50):
                delta_center_y_in_mm = offset_reference[1]
                flag = 1
            # delta_center_x_in_mm = np.round(cur_left_magnetic_center_x_in_mm - std_left_magnetic_center_x_in_mm, 6)
            # delta_center_y_in_mm = -np.round(cur_left_magnetic_center_y_in_mm - std_left_magnetic_center_y_in_mm, 6)
            
            focal_length_in_mm = 8
            pixel_size_in_mm = 0.0045
            magnetic_distance_in_mm = 113
            
            cur_magnetic_distance_in_pixel = math.fabs(cur_center_left[0] - cur_center_right[0])
            std_magnetic_distance_in_pixel = math.fabs(std_center_left[0] - std_center_right[0])
            
            cur_hypotenuse_distance_between_camera_and_magnetic = (focal_length_in_mm / (pixel_size_in_mm * cur_magnetic_distance_in_pixel)) * magnetic_distance_in_mm
            std_hypotenuse_distance_between_camera_and_magnetic = (focal_length_in_mm / (pixel_size_in_mm * std_magnetic_distance_in_pixel)) * magnetic_distance_in_mm
            
            cur_line_distance_between_camera_and_magnetic = round(cur_hypotenuse_distance_between_camera_and_magnetic * math.cos(40))
            std_line_distance_between_camera_and_magnetic = round(std_hypotenuse_distance_between_camera_and_magnetic * math.cos(40))
            
            delta_center_l_in_mm = np.round(cur_line_distance_between_camera_and_magnetic - std_line_distance_between_camera_and_magnetic, 4)
            
            if position_to_calib == 'right':
                if flag == 0:
                    delta_center_x_in_mm = -delta_center_x_in_mm
                    delta_center_y_in_mm = delta_center_y_in_mm
                cv2.putText(cur_image_debug, "Right: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif position_to_calib == 'left':
                cv2.putText(cur_image_debug, "Left: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif position_to_calib == 'buffer':
                cv2.putText(cur_image_debug, "Buffer: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_2.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_2.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)
            offset_reference = [delta_center_x_in_mm, delta_center_y_in_mm]
            # draw debug
            result_dict = dict()
            # result_dict["offset_x"] = offset_x
            # result_dict["offset_y"] = offset_y
            result_dict["offset_x"] = delta_center_x_in_mm
            result_dict["offset_y"] = delta_center_y_in_mm
            print("result_dict", result_dict)
            self.log_msg('Draw debug image successful\n', text_log_file)
            
        except Exception as e:
            self.log_msg('Error: ' + str(e) + '\n', text_log_file)
            #write traceback to logfile
            traceback.print_exc()
            traceback.print_exc(file=text_log_file)
            
            text_log_file.close()
            return result_dict, std_image_debug, cur_image_debug, offset_reference
        
        # Close file
        text_log_file.close()
        return result_dict, std_image_debug, cur_image_debug, offset_reference

if __name__ == '__main__':
    position_to_calib = 'left'
    image_path_1 = "/home/greystone/StorageWall/image_debug/calibrate/image_org/W01W01LSR01C01_20240923195633211.png"
    image_path_2 = "/home/greystone/StorageWall/image_debug/calibrate/image_org/W01W01LSR49C18_20240923201644031.png"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/calibrate/image_org/log_id_phone_slot.txt"
    model_weight_path = '/home/greystone/StorageWall/model_template/Calibrate/calibrate.onnx'
    ml_debug_path = "/home/greystone/StorageWall/image_debug/calibrate/image_org/debug"
    object_detector = CalibMagnetArea(model_weight_path, 0.70)
    result, image_pad_debug_1, image_pad_debug_2 = object_detector.run(image_path_1, image_path_2, position_to_calib, path_log_id_phone_slot, ml_debug_path)
