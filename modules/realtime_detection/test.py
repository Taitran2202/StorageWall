import cv2
import numpy as np
import os
from modules.detect_object_phone_slot.model_object_detect_phone_slot_copy import PhoneSlotObjectDetector

quantity_row = 54
quantity_column = 24
offset_x_min = 0
offset_y_min = 0
offset_x_max = 3280
offset_y_max = 2455

def find_position_phone_slot(offset_x_cur, offset_y_cur):

    step_x = (offset_x_max - offset_x_min)/(quantity_column - 1)
    step_y = (offset_y_max - offset_y_min)/(quantity_row - 1)
    
    delta_slot_x = int(step_x / 3)
    delta_slot_y = int(step_y / 3)
    
    slot_x_cur = round((offset_x_cur + step_x)/step_x)
    slot_y_cur = round((offset_y_cur + step_y)/step_y)
    
    delta_slot_x_cur_in_mm = offset_x_cur - ((slot_x_cur-1)*step_x)
    delta_slot_y_cur_in_mm = offset_y_cur - ((slot_y_cur-1)*step_y)
    
    delta_slot_x_cur_in_pixel = round((delta_slot_x_cur_in_mm / step_x) * 805)
    delta_slot_y_cur_in_pixel = round((delta_slot_y_cur_in_mm / step_y) * 215)
    
    return slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel


if __name__ == '__main__':
    
    status_phone_slot_left_wall = np.zeros((54,24),np.uint8)
    status_phone_slot_right_wall = np.zeros((54,24),np.uint8)
    
    slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel = find_position_phone_slot(0,75)
    print(f"slot_x: {slot_x_cur}, slot_y: {slot_y_cur}, delta_slot_x_cur_in_pixel: {delta_slot_x_cur_in_pixel}, delta_slot_y_cur_in_pixel: {delta_slot_y_cur_in_pixel},")
    
    positionToDetect = 'left'
    image_path = "/home/greystone/StorageWall/image_debug/W01W01LSR07C04_20240507141854765.png"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/log_id_phone_slot.txt"
    model_object_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PhoneSlots/model_object_detect_phone_slot.onnx'
    ml_debug_path = "/home/greystone/StorageWall/image_debug"
    phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.70)
    result, debug_img = phone_slot_object_detector.run(image_path, path_log_id_phone_slot, positionToDetect)
    cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)





