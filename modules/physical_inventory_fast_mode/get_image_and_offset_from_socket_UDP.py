import binascii
import logging
import multiprocessing
import os
import socket
import struct
import time

import cv2
import numpy as np

from globals import GV


class GetImageAndOffsetFromSocketUDP(multiprocessing.Process):
    def __init__(self) -> None:
        super().__init__()
        multiprocessing.Process.__init__(self)
        
        self.log = logging.getLogger('GetImageAndOffsetFromSocketUDP')
        self.log.setLevel(logging.DEBUG)
        self.log_file_path = os.path.join(GV().path_to_logs_folder, "GetImageAndOffsetFromSocketUDP.log")
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%y-%m-%d %H:%M:%S:%f")
        self.file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.file_handler)
        
        
        #self.HOST = '172.16.20.37'
        self.HOST = 'localhost'
        self.RECEIVER_PORT = 1234  # Replace with the appropriate port number
        self.BUF_LEN = 2048
        self.isExitApp = False
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.RECEIVER_PORT))

    def calculate_crc32_buffer(self, buffer, size):
        crc32 = binascii.crc32(buffer[:size]) & 0xFFFFFFFF
        return crc32

    def get_image_and_offset_streaming(self, pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_queue, share_buffer_offset_queue, count_total, count_fail, fps_get_img):
        try:
            time_start = time.time()
            while not self.isExitApp:
                #time.sleep(0.001)
                time.sleep(0)
                time_now = int(time.time())
                delta_time_cur = time_now - time_in_ctrl_PI.value

                if delta_time_cur > 20:
                    self.isExitApp = True
                    self.log.info("[GET IMAGE AND OFFSET] Stop process get_image_and_offset_streaming, self.isExitApp  = {}, delta_time_cur = {}".format(self.isExitApp, delta_time_cur))

                if pause_task.value == 1:
                    continue
                # self.log.info(f'[get_image_and_offset_streaming] Pause signal = {pause_task.value}')

                buffer_header = bytearray(self.BUF_LEN)
                buffer_image = bytearray(self.BUF_LEN)
                received_bytes, client_address = self.sock.recvfrom_into(buffer_header)
                #self.log.info("[get_image_and_offset_streaming] Received_bytes = {}".format(received_bytes))
                #print("received_bytes", received_bytes)
                if received_bytes == 24:
                    total_pack = struct.unpack("i", buffer_header[:4])[0]
                    size = struct.unpack("i", buffer_header[4:8])[0]
                    crc32_image = struct.unpack("I", buffer_header[8:12])[0]
                    offset_x = struct.unpack("i", buffer_header[12:16])[0]
                    offset_y = struct.unpack("i", buffer_header[16:20])[0]
                    crc32_offset = struct.unpack("I", buffer_header[20:24])[0]

                    crc32_offset_cal = self.calculate_crc32_buffer(buffer_header[0:20],20)
                    if crc32_offset_cal != crc32_offset:
                        count_fail.value += 1
                        count_total.value += 1
                        self.log.info(f"CRC get offset failed, CRC receive = {crc32_offset}, CRC cal = {crc32_offset_cal}")
                        print(f"CRC get offset failed, CRC receive = {crc32_offset}, CRC cal = {crc32_offset_cal}")
                        continue
                    
                    longbuf = bytearray(self.BUF_LEN * total_pack)
                    count_pack = 0
                    for i in range(total_pack):
                        received_bytes, client_address = self.sock.recvfrom_into(buffer_image)
                        if received_bytes == self.BUF_LEN:
                            longbuf[i * self.BUF_LEN:(i + 1) * self.BUF_LEN] = buffer_image
                            count_pack += 1
                            # print("count_pack: ",count_pack)
                    
                    # print(f"Received packet, length = {size}")
                    crc32_image_cal = self.calculate_crc32_buffer(longbuf, size)
                    if crc32_image_cal != crc32_image:
                        self.log.info(f"[GET IMAGE AND OFFSET] CRC get image failed, CRC receive = {crc32_image}, CRC cal = {crc32_image_cal}")
                        print(f"[GET IMAGE AND OFFSET] CRC get image failed, CRC receive = {crc32_image}, CRC cal = {crc32_image_cal}")
                        count_fail.value += 1
                        count_total.value += 1
                        #continue
                    dataFinal = None
                    rawData = np.frombuffer(longbuf[:size], dtype=np.uint8)
                    dataFinal = cv2.imdecode(rawData, cv2.IMREAD_COLOR)

                    offset = (offset_x, offset_y)
                    self.log.info("[GET IMAGE AND OFFSET] share_buffer_offset_queue.full() = {}".format(share_buffer_offset_queue.full()))
                    if not share_buffer_offset_queue.full():
                        share_buffer_offset_lock.acquire()
                        share_buffer_offset_queue.put(offset)
                        self.log.info(f"[GET IMAGE AND OFFSET] OFFSET has been put into the queue")
                        share_buffer_offset_lock.release()

                    # if dataFinal is not None:
                    #     print("Received image:", dataFinal.shape)
                    self.log.info("[GET IMAGE AND OFFSET] share_buffer_image_queue.full() = {}".format(share_buffer_image_queue.full()))
                    if not share_buffer_image_queue.full():
                        share_buffer_image_lock.acquire()
                        share_buffer_image_queue.put(dataFinal)
                        self.log.info("[GET IMAGE AND OFFSET] IMAGE has been put into the queue")
                        share_buffer_image_lock.release()
                        count_total.value += 1
                time_stop = time.time()
                time_elapsed = time_stop - time_start
                fps_get_img.value = round(count_total.value/time_elapsed)
            
            self.sock.close()
        except Exception as e:
            self.log.exception(f"GET IMAGE AND OFFSET Exception = {e}")

    def show_image_streaming(self, share_buffer_image_queue, share_buffer_offset_queue, count, fps_get_img):
        time_start = time.time()
        while not self.isExitApp:
            if (not share_buffer_image_queue.empty()) and (not share_buffer_offset_queue.empty()):
                offset_x, offset_y = share_buffer_offset_queue.get()
                image = share_buffer_image_queue.get()
                count.value += 1
                print("Cur image:", count.value)
                if image is None:
                    print("Get image failure, Queue is None!")
                else:
                    print("Get image success!")
                    # cv2.imwrite(f"/home/greystone/StorageWall/image_debug/image_debug/img_{count.value}_.jpg", image)
                    time_stop = time.time()
                    time_elapsed = time_stop - time_start
                    print("time_elapsed show image:",time_elapsed)
                    cv2.putText(image, f"fps_get_img:{fps_get_img.value}", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                    cv2.putText(image, f"X:{offset_x}, Y:{offset_y}", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                    cv2.imshow("image", image)
                    key = cv2.waitKey(1) % 0xFF
                    if key == ord('q'):
                        self.isExitApp = True
                        break

              
if __name__ == "__main__":
    
    share_buffer_image_lock = multiprocessing.Lock()
    share_buffer_offset_lock = multiprocessing.Lock()
    
    get_image = GetImageAndOffsetFromSocketUDP()
    count_fail = multiprocessing.Value('i',0)
    time_now = int(time.time())
    time_in_ctrl_PI = multiprocessing.Value("i",time_now)
    count_total = multiprocessing.Value('i', 0)
    count = multiprocessing.Value('i', 0)
    fps_get_img = multiprocessing.Value('i', 0)
    
    share_buffer_queue = multiprocessing.Queue(2)
    share_buffer_image_queue = multiprocessing.Queue(2)
    share_buffer_offset_queue = multiprocessing.Queue(2)
    
    pause_task = multiprocessing.Value('i', 0)
    
    process_get_img = multiprocessing.Process(target = get_image.get_image_and_offset_streaming, args = (pause_task, time_in_ctrl_PI, share_buffer_image_lock, share_buffer_offset_lock, share_buffer_image_queue, share_buffer_offset_queue, count_total, count_fail, fps_get_img))
    process_show_img = multiprocessing.Process(target = get_image.show_image_streaming, args = (share_buffer_image_queue, share_buffer_offset_queue, count, fps_get_img))
    
    process_get_img.start()
    process_show_img.start()
    
    while not get_image.isExitApp:
        time.sleep(1)
        time_in_ctrl_PI.value = int(time.time())
        # if count.value == 50:
        #     break

    get_image.sock.close()
    # process_show_img.join()
    # process_get_img.join()
    
    process_show_img.terminate()
    process_get_img.terminate()
#  /home/greystone/StorageWall/miniconda3/envs/ml_subapp/bin/python /home/greystone/StorageWall/apps/ml-subapp/appPython/modules/realtime_detection/get_image_streaming_UDP.py
