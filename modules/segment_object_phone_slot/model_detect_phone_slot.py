import os
import cv2
import time
import math
import numpy as np
import datetime

class PhoneSlotDetector:
    
    def __init__(self, model_path, CONF_THRESHOLD):
        self.model_path = model_path
        #self.model = cv2.dnn.readNetFromONNX(model_path)
        print('[PlasticBoxDetector] Load model successful!')
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.W = 1200
        self.H = 600
        self.match_color = (0,255,100)
        self.PLASTIC_BOX_MIN_DIM = 5 # minimum dimension of phone object
        self.thickness = 5
        self.result_have_plastic_box = 'have_plastic_box'
        self.result_none_plastic_box = 'none_plastic_box'
        self.tag_log = '[PlasticBoxDetector]'
        
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ': ' + msg
        # Log message
        text_log_file.write(msg)
    
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
    
    def process_mask_upsample(self, output, image_width, image_height, threshold=0.5):
        num_classes = output.shape[1] - 1
        mask = np.zeros((image_height, image_width, num_classes), dtype=np.uint8)

        for class_index in range(num_classes):
            class_output = output[0, class_index, :, :]
            class_output = cv2.resize(class_output, (image_width, image_height))

            mask[:, :, class_index] = (class_output > threshold).astype(np.uint8)

        return mask

    def detect(self, image_path, text_log_file):
        
        # Write the text to the file        
        msg = 'In function detect!\n'
        self.log_msg(msg, text_log_file)
        
        # Init
        object_list = []
        
        # Predict
        img = cv2.imread(image_path)
        if img is None:
            msg = 'Error: Image not found!\n'
            self.log_msg(msg, text_log_file)
            return object_list, None
        
        ov_model = ov.Core().read_model(self.model_path)
        
        
        
        # Exact result
        boxes = outputs.boxes  # Boxes object for bbox outputs
        masks = outputs.masks  # Masks object for segmentation masks outputs
        keypoints = outputs.keypoints  # Keypoints object for pose outputs
        probs = outputs.probs  # Probs object for classification outputs
        names = outputs.names
        
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
        
    def post_process(self, object_list, debug_img, text_log_file):
        
        # Write the text to the file
        msg = 'In function post-process!\n'
        self.log_msg(msg, text_log_file)
        
        # init
        result = self.result_none_plastic_box # 'have_plastic_box', 'none_plastic_box'
        
        is_plastic_box = False
        is_cross_line = False
        check_line = []
        
        # check object detected
        if len(object_list) == 0:
            return result + '_object_list_empty', debug_img
        
        box_list = []
        
        for current_object in object_list:
            # get name of the object
            object_label = current_object['label']
            w = current_object['w']
            h = current_object['h']
            y = current_object['y']
            x = current_object['x']
            min_dim, max_dim = self.find_min_max(w,h)
            
            # check object
            if object_label == 'plastic_box':
                # check minimum dimension of object with minimum dimension of phone
                if min_dim > self.PLASTIC_BOX_MIN_DIM:
                    box_list.append(current_object)
        
        if len(box_list) == 0:
            return result + '_box_list_empty', debug_img       
        else:
            is_plastic_box = True
                    
        # check the postion of phone in slot
        if is_plastic_box:
            for plastic_box in box_list:
                # check position of phone cross the line
                bbox = (plastic_box['x'], plastic_box['y'], plastic_box['w'], plastic_box['h'])
                line_start = check_line[0]
                line_end = check_line[1]
                top_left = (plastic_box['x'], plastic_box['y'])
                bottom_right = (plastic_box['x'] + plastic_box['w'], plastic_box['y'] + plastic_box['h'])
                is_cross_line = self.bounding_box_crosses_line(bbox, line_start, line_end)
                if is_cross_line:
                    cv2.rectangle(debug_img, top_left, bottom_right, self.match_color, self.thickness)
                    result = self.result_have_plastic_box
                    break

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
            cv2.putText(debug_img, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
            self.log_msg('Draw debug image successful\n', text_log_file)
            
            # Close file
            text_log_file.close()
            
            return result, debug_img
        
        except Exception as e:
            self.log_msg('Error: ' + str(e) + '\n', text_log_file)
            text_log_file.close()
            return self.result_none_plastic_box, None
        
if __name__ == '__main__':   
    image_path = "/home/greystone/StorageWall/image_debug/1712717122312_261_detect_phone_slot_org.jpg"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/log_id_phone_slot.txt"
    model_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/best.pt'
    ml_debug_path = "/home/greystone/StorageWall/image_debug"
    phone_slot_object_detector = PhoneSlotDetector(model_phone_slot_weight_path, 0.70)
    result, debug_img = phone_slot_object_detector.run(image_path, path_log_id_phone_slot)
    cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)