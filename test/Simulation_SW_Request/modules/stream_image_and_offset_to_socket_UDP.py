import socket
import time
import cv2
import os
import zlib
import logging
import struct
import numpy as np
import binascii

exit_flag = False

def calculate_crc32_buffer(buffer):
    crc32 = binascii.crc32(buffer[:]) & 0xFFFFFFFF
    return crc32

def subprocess_stream_images_and_offset(exit_flag, video_file_path):
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        #host_ip = '127.0.0.1'
        host_ip = 'localhost'
        host_port = 2678
        client_port = 1234
        server_address = (host_ip,host_port)
        client_address = (host_ip,client_port)
        server_socket.bind(server_address)
        
        PACKAGE_SIZE = 2048
        
        cap = cv2.VideoCapture(video_file_path)
        count = 0
        while exit_flag is False:
            time.sleep(0.001)
            time_start = time.time()
            ret, frame = cap.read()
            if ret is False:
                break;
            count += 1
            retval, frame_encoder = cv2.imencode('.jpg', frame)
            frame_len = len(frame_encoder)
            
            total_package_count_int = int(1 + (frame_len / PACKAGE_SIZE))
            
            longbuf = bytearray(PACKAGE_SIZE * (total_package_count_int + 1))
            longbuf = frame_encoder.tobytes()
            crc32_img = calculate_crc32_buffer(frame_encoder)
            
            offset_x = 0
            offset_y = int(3.42*count)
            
            buffer_header = bytearray(24)
            struct.pack_into("i", buffer_header, 0, total_package_count_int)
            struct.pack_into("i", buffer_header, 4, frame_len)
            struct.pack_into("I", buffer_header, 8, crc32_img)
            struct.pack_into("i", buffer_header, 12, offset_x)
            struct.pack_into("i", buffer_header, 16, offset_y)
            
            crc32_offset = calculate_crc32_buffer(buffer_header[:20])
            
            struct.pack_into("I", buffer_header, 20, crc32_offset)
            
            server_socket.sendto(buffer_header, client_address)
            
            print("frame_len", frame_len)
            print("crc32_offset", crc32_offset)
            print("crc32_img", crc32_img)
            # print("longbuf: ", longbuf)
            
            time.sleep(0.001)
            for i in range(total_package_count_int):
                time.sleep(0.001)
                start_index = i * PACKAGE_SIZE
                end_index = start_index + PACKAGE_SIZE
                server_socket.sendto(longbuf[start_index:end_index], client_address)
            time.sleep(0.001)
            
            # server_socket.sendto(crc32, server_address)
            logging.info(f"Image have been sent: {frame.shape}")
            logging.info(f"Quantity image have been sent: {frame.shape}")
            time_stop = time.time()
            elapsed_time = time_stop - time_start
            fps = round(1/elapsed_time)
            logging.info(f"subprocess_stream_images_and_offset Fps: {fps}")
            
        server_socket.close()
    except Exception as e:
        logging.exception(f"Exception in subprocess_stream_images: {e}")
    
if __name__ == "__main__":
    exit_flag = False
    video_file_path = "/home/greystone/StorageWall/image_debug/physical_inventory/recording_2024-05-11.avi"
    # subprocess_stream_images_and_offset(exit_flag, video_file_path)

    
        
        
    