'''
python modules/detect_object_phone_slot/script_batch_test.py \
    --model dataset/model_object_detect_phone_slot/models/v3/best.onnx \
    --imgs dataset/model_object_detect_phone_slot/data/unseen
'''
import os
import cv2.dnn
import argparse
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
parser.add_argument("--imgs", default=str(ASSETS / "bus.jpg"), help="Path to input imagew.")
parser.add_argument("--debug",default="dst",help="folder debug path")
args = parser.parse_args()
    
# Inits
onnx_model = args.model
input_images = args.imgs
debug_path = args.debug
CONF_THRESHOLD = 0.8
CLASSES = ['phone','gripper'] # yaml_load(check_yaml("coco128.yaml"))["names"]
images_tail = ['png','jpg','jpeg']

class PhoneSlotObjectDetector:
    
    def __init__(self, model_path, CONF_THRESHOLD):
        self.W = 1600
        self.H = 1200
        self.match_color = (0,255,100)
        self.GRIPPER_MIN_DIM = 100 # minimum dimension of gripper object
        self.GRIPPER_MAX_DIM = 400 # maximum dimension of gripper object
        self.PHONE_MIN_DIM = 200 # minimum dimension of phone object
        self.thickness = 5
        self.model_path = model_path
        self.CONF_THRESHOLD =CONF_THRESHOLD
        self.CLASSES = ['phone','gripper']
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.model = cv2.dnn.readNetFromONNX(model_path) # Load the ONNX model
        print('[PhoneSlotObjectDetector] Load model successful!')
        self.result_have_phone = 'have_phone'
        self.result_none_phone = 'none_phone'
        
    def find_top_left_and_bottom_right(self, points):
        points_array = np.array(points)
        top_left = np.min(points_array, axis=0)
        bottom_right = np.max(points_array, axis=0)
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
    
    def detect(self, image_path):
        # Read the input image
        original_image: np.ndarray = cv2.imread(image_path)
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

        boxes = []
        scores = []
        class_ids = []
        detections = [] # [{'class_id':, 'class_name':, 'confidence':, 'box':[], 'scale':},...]
        object_list = []
        
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
    
    def post_process(self, object_list, debug_img):
        # init
        result = self.result_none_phone # 'have_phone', 'none_phone'
        
        is_gripper = False
        is_phone = False
        is_cross_line = False
        check_line = []
        
        # check object detected
        if len(object_list) == 0:
            return result + '_object_list_empty', debug_img
        
        # check gripper exist if not there is no phone
        gripper_list = []
        phone_list = []
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
                    
            elif object_label == 'gripper':
                # check minimum dimension of object with minimum dimension of gripper
                if min_dim > self.GRIPPER_MIN_DIM \
                    and max_dim < self.GRIPPER_MAX_DIM \
                        and y > int(self.H/2):
                            gripper_list.append(current_object)
        
        # check object detected
        if len(gripper_list) == 0:
            return result + '_gripper_list_empty', debug_img
        
        elif len(gripper_list) == 1:
            # draw line by top_left
            # find top left point
            current_gripper = gripper_list[0]
            line_start = (0, current_gripper['y'])
            line_end = (self.W, current_gripper['y'])
            check_line = [line_start, line_end]
            
            # draw line by most top_left and most top_right
            cv2.line(debug_img, line_start, line_end, self.match_color, self.thickness-2)
            
            is_gripper = True
        
        elif len(gripper_list) >= 2:
            list_of_point = []
            for current_gripper in gripper_list:
                x = current_gripper['x']
                y = current_gripper['y']
                w = current_gripper['w']
                h = current_gripper['h']
                current_top_left = (x, y)
                current_top_right = (x+w, y)
                current_bottom_right = (x+w, y+h)
                current_bottom_left = (x, y+h)
                list_of_point += [current_top_left,current_top_right,current_bottom_right,current_bottom_left]
                
            line_start, line_end = self.find_top_left_and_top_right(list_of_point)
            check_line = [line_start, line_end]

            # draw line by most top_left and most top_right
            cv2.line(debug_img, line_start, line_end, self.match_color, self.thickness-2)
            
            is_gripper = True
        
        if len(phone_list) == 0:
            return result + '_phone_list_empty', debug_img       
        else:
            is_phone = True
                    
        # check the postion of phone in slot
        if is_gripper and is_phone:
            for phone in phone_list:
                # check position of phone cross the line
                bbox = (phone['x'], phone['y'], phone['w'], phone['h'])
                line_start = check_line[0]
                line_end = check_line[1]
                top_left = (phone['x'], phone['y'])
                bottom_right = (phone['x'] + phone['w'], phone['y'] + phone['h'])
                is_cross_line = self.bounding_box_crosses_line(bbox, line_start, line_end)
                if is_cross_line:
                    cv2.rectangle(debug_img, top_left, bottom_right, self.match_color, self.thickness)
                    result = self.result_have_phone
                    break

        return result, debug_img
    
    def run(self, image_path):
        # detect
        object_list, debug_img = self.detect(image_path)
        # post proceess
        result, debug_img = self.post_process(object_list, debug_img)
        # draw debug
        cv2.putText(debug_img, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
        return result, debug_img

if __name__ == "__main__":
  
    # Create debug path   
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # Check all files in the folder
    images = os.listdir(input_images)
    
    # Instance model object
    model_object_detect_phone_slot = PhoneSlotObjectDetector(onnx_model, CONF_THRESHOLD)
    
    for image in images:
        
        image_tail = image.split('.')[-1]
        
        if image_tail in images_tail:
            image_path = os.path.join(input_images,image)
            image_dst_path = os.path.join(debug_path, image)
            result, debug_image = model_object_detect_phone_slot.run(image_path)
            cv2.imwrite(image_dst_path, debug_image)
            print('Predicting image=',image,' Result = ',result)
            