import os
import cv2
import json
import subprocess
import datetime
import numpy as np
import logging

def is_container_running(container_name):
    # client = docker.from_env()

    # try:
    #     container = client.containers.get(container_name)
    #     return container.status == "running"
    # except docker.errors.NotFound:
    
    #     return False
    
    try:
        cmd_check = "docker container inspect -f '{{.State.Running}}' " + "{}".format(container_name)
        result = subprocess.check_output(cmd_check, shell=True, text=True)
        # os.system(cmd_check_container_running)
        if 'true' in result:
            return True
    except Exception as e:
        logging.exception("Failed to execute command cmd_check = {}".format(cmd_check))
    return False

def rotate_point_90(x, y, image_width, image_height):
    # Convert (x, y) to be centered at (0, 0)
    x_rotated = int(y)
    y_rotated = int(image_width - x)
    return x_rotated, y_rotated

def find_top_left_point(localization):
    top_left_point = (localization[0][0], localization[0][1])
    for i in range(1, 4):
        if localization[i][0] + localization[i][1] < top_left_point[0] + top_left_point[1]:
            top_left_point = (localization[i][0], localization[i][1])
    return top_left_point

def detect_most_right_bottom_corner(points):
    if not points:
        return None

    # Initialize variables to store the maximum x-coordinate and y-coordinate
    max_x = points[0][0]
    max_y = points[0][1]

    # Iterate through the points to find the maximum x and y coordinates
    for point in points:
        x, y = point
        if x > max_x:
            max_x = x
            max_y = y
        elif x == max_x and y > max_y:
            max_y = y

    # Return the most right bottom corner point
    return (max_x, max_y)

def draw_debug_image(img,json_path):
    # init
    H,W,_ = img.shape
    list_of_text = []
    list_of_localization = []
    
    # Open json path
    if os.path.exists(json_path) :
        
        # Read json
        with open(json_path, 'r') as f:
            # Read result scan barcode
            result_scan_barcode = json.load(f)
            
        path_image = list(result_scan_barcode.keys())
        path_image = path_image[0]
        json_results = result_scan_barcode[path_image]
        
        if len(json_results) == 0:
            # Put text
            cv2.putText(img, 'Found No barcode', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 155), 2)
            return img
        
        for i, json_result in enumerate(json_results):
            current_text = json_result['text']
            current_localization = json_result['localization']
            
            # Draw polygon
            pts = np.array(current_localization, np.int32)
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            bottom_right_point = detect_most_right_bottom_corner(current_localization)
            
            # Append to list
            list_of_localization.append(bottom_right_point)
            list_of_text.append(current_text)
            
        # Draw text
        img_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        for j in range(len(list_of_localization)):
            x_rotated, y_rotated = rotate_point_90(list_of_localization[j][0], list_of_localization[j][1], W, H)
            cv2.putText(img_90, list_of_text[j], (x_rotated, y_rotated), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 2)
        
        return img_90 
    else:
        # Put text
        cv2.putText(img, 'Found No file', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 155), 2)
        return img
    
def log_msg(msg, tag_log, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
    
