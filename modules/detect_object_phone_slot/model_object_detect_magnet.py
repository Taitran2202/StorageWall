import os
import cv2.dnn
import cv2
import numpy as np
import datetime
import json
    
# Inits
CONF_THRESHOLD = 0.8
CLASSES = ['magnet'] # yaml_load(check_yaml("coco128.yaml"))["names"]
images_tail = ['png','jpg','jpeg']

class MagnetObjectDetector:
    
    def __init__(self, model_path, CONF_THRESHOLD):
        self.W = 1200
        self.H = 600
        self.thickness = 5
        self.match_color = (0,255,100)
        self.GRIPPER_MIN_DIM = 100 # minimum dimension of gripper object
        self.GRIPPER_MAX_DIM = 400 # maximum dimension of gripper object
        self.PHONE_MIN_DIM = 20 # minimum dimension of phone object
        self.MAGNET_MIN_DIM = 1 # minimum dimension of magnet object
        self.model_path = model_path
        self.CONF_THRESHOLD =CONF_THRESHOLD
        self.CLASSES = ['magnet']
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.model = cv2.dnn.readNetFromONNX(model_path) # Load the ONNX model
        print('[MagnetObjectDetector] Load model successful!')
        self.result_have_magnet = 'have_magnet'
        self.result_none_magnet = 'none_magnet'
        self.result_have_phone = 'have_phone'
        self.result_none_phone = 'none_phone'
        self.tag_log = '[MagnetObjectDetector]'
    
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
        
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
        image_crop = cv2.resize(image_crop,(1200, 600))
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
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def detect(self, image_path, text_log_file):
        
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
        
        original_image = cv2.imread(image_path)
        
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
        
        return object_list, original_image
    
    def post_process(self, object_list, debug_img, text_log_file):
        
        # Write the text to the file
        msg = 'In function post-proceess!\n'
        self.log_msg(msg, text_log_file)
        
        # init
        result = self.result_none_magnet # 'have_magnet', 'none_magnet'
        
        is_magnet = False
        is_cross_line = False
        
        # check object detected
        if len(object_list) == 0:
            return result + '_object_list_empty', debug_img
        
        # check gripper exist if not there is no phone
        magnet_list = []
        magnet_list_in_arena = []

        for current_object in object_list:
            # get name of the object
            object_label = current_object['label']
            w = current_object['w']
            h = current_object['h']
            y = current_object['y']
            x = current_object['x']
            min_dim, max_dim = self.find_min_max(w,h)
            
            # check object
            if object_label == 'magnet':
                # check minimum dimension of object with minimum dimension of phone
                if min_dim > self.MAGNET_MIN_DIM:
                    magnet_list.append(current_object)
                    
        
            
        # check object detected 
        print("magnet_list",magnet_list)
        if len(magnet_list) == 0:
            return result + '_magnet_list_empty', debug_img       
        else:
            is_magnet = True
          
        # check the postion of magnet in slot      
        if is_magnet:
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            
            x_from_magnet = data.get('x_from_magnet', 0.2) #range 0.0 to 1.0
            y_from_magnet = data.get('y_from_magnet', 0.3)
            x_to_magnet = data.get('x_to_magnet', 0.8)
            y_to_magnet = data.get('y_to_magnet', 0.5)
            debug_img, location_arena_detect = self.draw_arena_box(debug_img, x_from_magnet, y_from_magnet, x_to_magnet, y_to_magnet, color_bgr = [0, 0, 255], is_copy = True)
            i = 0
            for magnet in magnet_list:
                # check position of phone cross the line
                bbox = (magnet['x'], magnet['y'], magnet['w'], magnet['h'])
                x, y, w, h = location_arena_detect
                if(magnet['x'] > x and magnet['x'] < x + w and magnet['y'] > y) and magnet['y'] < y + h:
                    magnet_list_in_arena.append(bbox)
                    result = self.result_have_magnet
            self.log_msg('magnet_list_in_arena: {magnet_list_in_arena}', text_log_file)
            print("magnet_list_in_arena: ",magnet_list_in_arena)
        return result, debug_img
    
    def run(self, image_path, log_file_path):
        # init logger
        text_log_file = open(log_file_path, 'w')
        try:
            # detect
            self.log_msg('Start run function detect!\n', text_log_file)
            object_list, debug_img = self.detect(image_path, text_log_file)
            self.log_msg('End run function detect!\n', text_log_file)
            
            # Print out current object
            msg  = 'Current object list: ' + str(object_list) + '\n'
            self.log_msg(msg, text_log_file)
            
            # post process
            self.log_msg('Start run function post-process!\n', text_log_file)
            result, debug_img = self.post_process(object_list, debug_img, text_log_file)
            self.log_msg('End run function post-process!\n', text_log_file)
            
            msg  = 'Result=' + result + '\n'
            self.log_msg(msg, text_log_file)
            
            # draw debug
            cv2.putText(debug_img, result, (50,250), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 1, cv2.LINE_AA)
            self.log_msg('Draw debug image successful\n', text_log_file)
            
            # Close file
            text_log_file.close()
            
            return result, debug_img
        
        except Exception as e:
            self.log_msg('Error: ' + str(e) + '\n', text_log_file)
            text_log_file.close()
            return self.result_none_phone, None


if __name__ == '__main__':
    image_path = "/home/greystone/StorageWall/image_debug/W01LSR02C01_20240404212321512_crop.jpg"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/log_id_phone_slot.txt"
    model_object_magnet_weight_path = '/home/greystone/StorageWall/model_template/Magnet/model_object_detect_magnet.onnx'
    ml_debug_path = "/home/greystone/StorageWall/image_debug"
    magnet_object_detector = MagnetObjectDetector(model_object_magnet_weight_path, 0.70)
    result, debug_img = magnet_object_detector.run(image_path, path_log_id_phone_slot)
    cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_magnet.jpg'),debug_img)
