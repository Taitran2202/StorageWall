'''
python3 modules/detect_phone_slot/script_batch_predict.py\
    --pathToImgs=dataset/model_detect_phone_slot/phone_slot/images1\
        --pathToModel=dataset/model_detect_phone_slot/phone_slot/runs/segment/train/weights/best.pt\
            --pathToDebug=dst
'''
import os
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES']='-1'

print('Import successfully!')

# 1. Parse arguments
parser = argparse.ArgumentParser(description = 'Detect phone in slot')
parser.add_argument('--pathToImgs',type = str, help = 'directory of the image need to predict', required = True)
parser.add_argument('--pathToModel',type = str, help = 'directory of model',required = True)
parser.add_argument('--pathToDebug',type = str, help = 'directory of the debug folder',required = True)
args = parser.parse_args()

# 2.TODO Initialize
CONF_THRESHOLD = 0.9
images_path = args.pathToImgs
debug_path = args.pathToDebug
model_path = args.pathToModel


class PhoneSlotDetector:
    def __init__(self, model_path, CONF_THRESHOLD):
        self.model_path = model_path
        self.yolov8seg = YOLO(self.model_path)
        self.CONF_THRESHOLD = CONF_THRESHOLD
        
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
            return object_list , None
        
    def post_process(self, object_list):
        # init
        result = "none_phone" # 'have_phone', 'none_phone'
        
        # check gripper exist if not there is no phone
        
        
        # check the postion of phone in slot
        
        return result



if __name__ == "__main__":
    
    start_time = time.time() # Start counting time
    msg = '---Detect phone in slot---'
    print(msg)

    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # 7. Load model
    model = PhoneSlotDetector(model_path, CONF_THRESHOLD)  # pretrained YOLOv8n model
    print('Instantiate model successful!')

    images = os.listdir(images_path)
    
    for k, image in enumerate(images):
        
        image_path = os.path.join(images_path, image)

        # 8.Predict
        object_list, debug_img = model.detect(image_path)

        if len(object_list) > 0:
            cv2.imwrite(os.path.join(debug_path, image), debug_img)
        else:
            print('model can not detect anything!')
            
        print(f'({k+1}/{len(images)}) {image}-{len(object_list)}\n {object_list}')