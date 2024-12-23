import cv2
import numpy as np
import os
import multiprocessing
from modules.detect_object_phone_slot.model_object_detect_phone_slot_PI_normal import PhoneSlotObjectDetector
import time

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
    # phone_slot_object_detector = PhoneSlotObjectDetector(model_object_phone_slot_weight_path, 0.70)
    # result, debug_img = phone_slot_object_detector.run(image_path, path_log_id_phone_slot, positionToDetect)
    # cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)
    data = np.zeros((54*24),np.uint8)
    flattened_data = data.flatten()
    
    count_fail = multiprocessing.Value('i',0)
    count_total = multiprocessing.Value('i',0)
    fps_get_img = multiprocessing.Value('i', 0)

    data = np.zeros((53*24),np.uint8)
    data_1 = np.zeros((54,24),np.uint8)
    
    status_phone_slot_left_wall = multiprocessing.Array('i',data)
    count_status_phone_slot_left_wall = multiprocessing.Array('i',data)



    # count_status_phone_slot_left_wall_array[1,0] = 5
    # if count_status_phone_slot_left_wall_array[1,0] > 3:
    #     status_phone_slot_left_wall_array[1,0] = 1
    # print('status_phone_slot_left_wall', status_phone_slot_left_wall)
    # print('count_status_phone_slot_left_wall_array',count_status_phone_slot_left_wall_array)
    
    # status_phone_slot_left_wall.Array = status_phone_slot_left_wall_array.flatten()
    # count_status_phone_slot_left_wall.Array = count_status_phone_slot_left_wall_array.flatten()
    # print('status_phone_slot_left_wall', status_phone_slot_left_wall.Array)
    # print('status_phone_slot_left_wall', count_status_phone_slot_left_wall.Array)
    
    # count_status_phone_slot_left_wall_array = np.array(count_status_phone_slot_left_wall.get_obj(), dtype=np.uint8).reshape((54, 24))
    # status_phone_slot_left_wall = np.array(status_phone_slot_left_wall.get_obj(), dtype=np.uint8).reshape((54, 24))
    # print('status_phone_slot_left_wall', status_phone_slot_left_wall)
    # print('count_status_phone_slot_left_wall_array',count_status_phone_slot_left_wall_array)
time_1 = int(time.time())
time_in_ctrl_PIa = multiprocessing.Value('i', time_1)

print("1:", time_in_ctrl_PIa.value)
time.sleep(3)

time_2 = int(time.time())
time_in_ctrl_PIa.value =  time_2# Cập nhật lại giá trị
print("2:", time_in_ctrl_PIa.value)


import cv2
import numpy as np
import socket
import struct

# Calculate CRC32 checksum
def calculate_crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF

# Send data over UDP
def send_udp(sock, data, addr):
    sock.sendto(data, addr)
    time.sleep(0.2)

# Convert C++ code to Python
image_count = 0
fps_time_obj = time.time()
image_height = 0
image_width = 0
package_size = 0
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = None

fps = (image_count * 1000) / (time.time() - fps_time_obj)
print(f"fps: {fps:.2f}, count: {image_count}")

new_image = np.ndarray((image_height, image_width, 4), dtype=np.uint8, buffer=image.Data, order='C')
_, encoded_frame = cv2.imencode('.jpg', new_image)

total_package_count = 1 + (len(encoded_frame) - 1) // package_size
buffer = struct.pack('IIIIII',
                    total_package_count,
                    len(encoded_frame),
                    calculate_crc32(encoded_frame),
                    shmSnapshortObj.getSnapshotPositionX(),
                    shmSnapshortObj.getSnapshotPositionZ(),
                    calculate_crc32(buffer))

send_udp(sock, buffer, server_address)

for i in range(total_package_count):
    send_udp(sock, encoded_frame[i*package_size:(i+1)*package_size], server_address)




