import socket
import time
import cv2
import binascii
import logging
import struct
import numpy as np

exit_flag = False

def calculate_crc32_buffer(buffer, size):
    crc32 = binascii.crc32(buffer[:size]) & 0xFFFFFFFF
    return crc32

# def calculate_crc32_buffer(buffer, size):
#     count = 0
#     crc = 0
#     mask = 0
#     table = np.zeros(256, dtype=np.uint32)

#     # Setup the Lookup table
#     for count in range(256):
#         crc = count
#         for i in range(8):
#             mask = -(crc & 1)
#             crc = (crc >> 1) ^ (0xEDB88320 & mask)
#         table[count] = crc

#     crc = 0xFFFFFFFF   # Initial Value

#     for i in range(size):
#         crc = (crc >> 8) ^ table[(crc ^ buffer[i]) & 0xFF]

#     return (~crc) # crc^0xFFFFFFFF : Final XOR Value

def subprocess_stream_images(exit_flag, video_file_path):
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        #host_ip = '127.0.0.1'
        host_ip = 'localhost'
        host_port = 2678
        client_port = 1789
        server_address = (host_ip,host_port)
        client_address = (host_ip,client_port)
        server_socket.bind(server_address)
        
        PACKAGE_SIZE = 512
        
        cap = cv2.VideoCapture(video_file_path)
        count = 0
        while exit_flag is False:
            time.sleep(1)
            ret, frame = cap.read()
            if ret is False:
                break;
            count += 1
            retval, frame_encoder = cv2.imencode('.jpg', frame)
            frame_len = len(frame_encoder)
            
            total_package_count_int = int(1 + (frame_len / PACKAGE_SIZE))
            
            longbuf = bytearray(PACKAGE_SIZE * (total_package_count_int + 1))
            longbuf = frame_encoder.tobytes()
            crc32 = calculate_crc32_buffer(frame_encoder, total_package_count_int)
            
            
            buffer_header = bytearray(12)
            struct.pack_into("i", buffer_header, 0, total_package_count_int)
            struct.pack_into("i", buffer_header, 4, frame_len)
            struct.pack_into("I", buffer_header, 8, crc32)
            server_socket.sendto(buffer_header, client_address)
            
            print("total_package_count_int: ",total_package_count_int)
            print("frame_len", frame_len)
            print("crc32", crc32)
            # print("longbuf: ", longbuf)
            
            time.sleep(0.01)
            for i in range(total_package_count_int):
                time.sleep(0.01)
                start_index = i * PACKAGE_SIZE
                end_index = start_index + PACKAGE_SIZE
                server_socket.sendto(longbuf[start_index:end_index], client_address)
            time.sleep(0.01)
            
            # server_socket.sendto(crc32, server_address)
            print(f"Image have been sent: {frame.shape}")
            logging.info(f"Image have been sent: {frame.shape}")
        server_socket.close()
    except Exception as e:
        logging.exception(f"Exception in subprocess_stream_images: {e}")
    