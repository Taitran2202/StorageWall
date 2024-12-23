'''
python3 modules/detect_phone_slot/script_detect_phone_slot.py \
    --pathToImg=dataset/model_detect_phone_slot/phone_slot/example_phone_slot.png \
        --pathToDebug=dst
10-Mar-24 9h36m
'''
import os
import cv2
import time
import math
import logging
import argparse
import numpy as np
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES']='-1'

print('Import successfully!')

# 1. Parse arguments
parser = argparse.ArgumentParser(description = 'Detect phone in slot')
parser.add_argument('--pathToImg',type = str, help = 'directory of the image need to predict', required = True)
parser.add_argument('--pathToDebug',type = str, help = 'directory of the debug folder',required = True)
args = parser.parse_args()

# 2.TODO Initialize
CONF_THRESHOLD = 0.8
log_tag = '[DETECT_PHONE]'
image_path = args.pathToImg
debug_path = args.pathToDebug
log_file_path = os.path.join(debug_path,'detect_phone.log')
model_path = 'dataset/model_detect_phone_slot/phone_slot/runs/segment/train/weights/best.pt'

# 3. Define code blocks
class PhoneSlotDetector:
    
    def __init__(self, model_path, CONF_THRESHOLD):
        self.model_path = model_path
        self.yolov8seg = YOLO(self.model_path)
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.W = 1600
        self.H = 1200
        self.match_color = (0,255,100)
        self.GRIPPER_MIN_DIM = 100 # minimum dimension of gripper object
        self.PHONE_MIN_DIM = 200 # minimum dimension of phone object
        self.thickness = 5
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

    def detect(self, image_path):
        # Init
        object_list = []
        # Predict
        img = cv2.imread(image_path)
        img_org = img.copy()
        H,W,_ = img.shape
        debug_mask = np.zeros((H,W))
        debug_mask = debug_mask.astype(np.uint8)
        result = self.yolov8seg([img])
        result = result[0]

        # Exact result
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        names = result.names
        
        # Post process
        if masks:
            label_list = [names[int(c)] for c in boxes.cls]
            conf_list = [round(float(f),2) for f in boxes.conf]
            for i,mask in enumerate(masks):
                # init
                current_object = {}
                # get current label
                current_label = label_list[i]
                current_conf = conf_list[i]
                current_object['label'] = current_label
                current_object['conf'] = current_conf
                # process tensor mask to numpy array
                mask = mask.cpu()
                mask = mask.data.numpy()
                mask = mask[0]
                mask = mask * 255
                mask = mask.astype(np.uint8)
                mask_resize = cv2.resize(mask,(W,H),interpolation=cv2.INTER_AREA) # resize mask
                # find coordinate
                top_left, bottom_right = self.find_top_left_and_bottom_right(mask_resize)
                width = int(bottom_right[0] - top_left[0])
                height = int(bottom_right[1] - top_left[1])
                current_object['x'] = top_left[0]
                current_object['y'] = top_left[1]
                current_object['w'] = width
                current_object['h'] = height
                # check condition of confident score
                if current_conf >= self.CONF_THRESHOLD:
                    color = (0,255,0)
                    object_list.append(current_object)
                else:
                    color = (0,0,255)
                print(f'{i}-label={label_list[i]}-conf={str(conf_list[i])}-top_left={top_left}-bottom_right={bottom_right}')
                # draw debug
                debug_mask = cv2.bitwise_or(debug_mask, mask_resize)
                cv2.rectangle(img, top_left, bottom_right, color, 1)
                cv2.putText(img, label_list[i], top_left, cv2.FONT_HERSHEY_SIMPLEX , 1, color, 1, cv2.LINE_AA)
                cv2.putText(img,str(conf_list[i]), (top_left[0],top_left[1]+30), cv2.FONT_HERSHEY_SIMPLEX , 1, color, 1, cv2.LINE_AA) 

            debug_img = cv2.addWeighted(img, 0.8, cv2.cvtColor(debug_mask,cv2.COLOR_GRAY2RGB), 0.2, 0)
            return object_list , debug_img
        else:
            return object_list , img_org
        
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
            min_dim, max_dim = self.find_min_max(w,h)
            
            # check object is phone or gripper
            if object_label == 'phone':
                # check minimum dimension of object with minimum dimension of phone
                if min_dim > self.PHONE_MIN_DIM:
                    phone_list.append(current_object)
                    
            elif object_label == 'gripper':
                # check minimum dimension of object with minimum dimension of gripper
                if min_dim > self.GRIPPER_MIN_DIM:
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
        return result, debug_img


if __name__ == "__main__":

    start_time = time.time() # Start counting time
    msg = '---Detect phone in slot---'
    print(msg)
    
    # 4. Create debug folder
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
        print(log_tag," Create debug folder: ",debug_path)
    else:
        print(log_tag,"Debug folder existed: ",debug_path)

    # 5. Create logger
    logging.basicConfig(filename=log_file_path,format='%(asctime)s %(message)s',filemode='w')
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold zof logger to DEBUG
    logger.setLevel(logging.INFO)
    # log message
    logger.info(msg)

    # 6. Check input files
    files = [image_path, model_path]
    msg = "Checking files:\n"
    enough_file = True
    
    for file in files:
      file_exist = os.path.exists(file)
      enough_file &= file_exist
      msg += f'{file}: {file_exist}\n'
    logger.info(msg)
    
    if enough_file:
        msg = log_tag + 'All files exist!'
        logger.info(msg)
    else:
        msg = log_tag + 'Finish detect phone slot, Not enough files to analyze!'
        logger.info(msg)
        exit(-1)

    # 7. Load model
    model = PhoneSlotDetector(model_path, CONF_THRESHOLD) # pretrained YOLOv8n model
    print('Instantiate model successful!')

    # 8.Predict
    result, debug_img = model.run(image_path)
    if 'have_phone' in result:
        print('Result=HavePhone')
        logger.info(f'{log_tag} Result=HavePhone')
        cv2.imwrite(os.path.join(debug_path,'debug_img.jpg'),debug_img)
        logger.info(f'{log_tag} Predict model successful!')
    else:
        print('Result=NonPhone')
        logger.info(f'{log_tag} Result=NonPhone')

    