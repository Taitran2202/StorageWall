import socket
import struct
import cv2
import numpy as np
import binascii
import numpy as np
import threading
from multiprocessing import Process, Queue, Event, Lock
import time

class getImageStream:
    def __init__(self):
        self.HOST = '172.16.20.218'
        #self.HOST = 'localhost'
        self.RECEIVER_PORT = 1234  # Replace with the appropriate port number
        self.BUF_LEN = 4096 
        self.isExitApp = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.RECEIVER_PORT))

    def calculate_crc32_buffer(self, buffer, size):
        crc32 = binascii.crc32(buffer[:size]) & 0xFFFFFFFF
        return crc32

    def get_image_streaming(self, frame, lock):
        
        while not self.isExitApp:
            time.sleep(0.001)
            time_start = time.time()
            buffer = bytearray(self.BUF_LEN)
            received_bytes, client_address = self.sock.recvfrom_into(buffer)
            if received_bytes == 12:
                total_pack = struct.unpack("i", buffer[:4])[0]
                size = struct.unpack("i", buffer[4:8])[0]
                crc32 = struct.unpack("I", buffer[8:12])[0]
                print(f"Expecting length of packs: {total_pack}")

                longbuf = bytearray(self.BUF_LEN * total_pack)
                for i in range(total_pack):
                    received_bytes, client_address = self.sock.recvfrom_into(buffer)
                    if received_bytes == self.BUF_LEN:
                        longbuf[i * self.BUF_LEN:(i + 1) * self.BUF_LEN] = buffer

                print(f"Received packet, length = {size}")
                crc32cal = self.calculate_crc32_buffer(longbuf, size)
                if crc32cal != crc32:
                    print(f"CRC failed, CRC receive = {crc32}, CRC cal = {crc32cal}")
                    time_stop = time.time()
                    time_elapsed = time_stop - time_start
                    print("time_elapsed get image", time_elapsed)
                    continue
                else:
                    rawData = np.frombuffer(longbuf[:size], dtype=np.uint8)
                    dataFinal = cv2.imdecode(rawData, cv2.IMREAD_COLOR)
                    lock.acquire()
                    frame.put(dataFinal)
                    lock.release()
                    print(dataFinal.shape)
                    time_stop = time.time()
                    time_elapsed = time_stop - time_start
                    print("time_elapsed get image", time_elapsed)
                    break

            # return frame, failedCount
    def show_image_streaming(self, frame, lock, count):
        time_start = time.time()
        image = frame.get()
        if image is None:
            print("Get image failure, Queue is None!")
        else:
            print("Get image success!")
            cv2.imwrite(f"/home/greystone/StorageWall/image_debug/image_debug/img_{count}_.jpg", image)
            # cv2.imshow("frame receiving" ,image)

            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     get_image.sock.close()
        time_stop = time.time()
        time_elapsed = time_stop - time_start
        print("time_elapsed show image:",time_elapsed)
            
            
        
    
if __name__ == "__main__":
    get_image = getImageStream()
    count = 0
    event = Event()
    lock = Lock()
    frame = Queue()
    while True:
        count += 1
        process_get_img = Process(target = get_image.get_image_streaming, args = (frame, lock))
        process_show_img = Process(target = get_image.show_image_streaming, args = (frame, lock, count,))
        #image, failedCount = get_image.get_image_streaming(failedCount)
        try:
            process_get_img.start()
            process_show_img.start()
            process_show_img.join()
            print(f"Total image: {count}")
        except Exception as e:
            print(f"Error: {e}")
        
        # cv2.imshow("data receiving", image)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     get_image.sock.close()
        #     break

#  /home/greystone/StorageWall/miniconda3/envs/ml_subapp/bin/python /home/greystone/StorageWall/apps/ml-subapp/appPython/modules/realtime_detection/get_image_streaming_UDP.py
