import socket
import struct
import cv2
import numpy as np
import binascii
import numpy as np
import multiprocessing
import time

class GetImageFromSocketUDP:
    def __init__(self):
        #self.HOST = '127.0.0.1'
        self.HOST = 'localhost'
        self.RECEIVER_PORT = 1789  # Replace with the appropriate port number
        self.BUF_LEN = 2048 
        self.isExitApp = False
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.RECEIVER_PORT))

    def calculate_crc32_buffer(self, buffer, size):
        crc32 = binascii.crc32(buffer[:size]) & 0xFFFFFFFF
        return crc32

    def get_image_streaming(self, share_buffer_queue, count_total, count_fail, fps_get_img):
        time_start = time.time()
        while not self.isExitApp:
            buffer = bytearray(self.BUF_LEN)
            received_bytes, client_address = self.sock.recvfrom_into(buffer)
            print("received_bytes", received_bytes)
            if received_bytes == 12:
                total_pack = struct.unpack("i", buffer[:4])[0]
                size = struct.unpack("i", buffer[4:8])[0]
                crc32 = struct.unpack("I", buffer[8:12])[0]
                print(f"Expecting length of packs: {total_pack}")

                longbuf = bytearray(self.BUF_LEN * total_pack)
                count_pack = 0
                for i in range(total_pack):
                    received_bytes, client_address = self.sock.recvfrom_into(buffer)
                    if received_bytes == self.BUF_LEN:
                        longbuf[i * self.BUF_LEN:(i + 1) * self.BUF_LEN] = buffer
                        count_pack += 1
                        print("count_pack: ",count_pack)

                print(f"Received packet, length = {size}")
                crc32cal = self.calculate_crc32_buffer(longbuf, size)
                print("longbuf: ", longbuf)
                if crc32cal != crc32:
                    print(f"CRC failed, CRC receive = {crc32}, CRC cal = {crc32cal}")
                    count_fail.value += 1
                    count_total.value += 1
                    print(f"count_fail: {count_fail.value}")
                    #continue

                rawData = np.frombuffer(longbuf[:size], dtype=np.uint8)
                dataFinal = cv2.imdecode(rawData, cv2.IMREAD_COLOR)

                print(dataFinal.shape)
                
                share_buffer_queue.put(dataFinal)
                count_total.value += 1
                print("count_total = ", count_total.value)
            time_stop = time.time()
            time_elapsed = time_stop - time_start
            fps_get_img.value = round(count_total.value/time_elapsed)
                    

            # return frame, failedCount
    def show_image_streaming(self, share_buffer_queue, count_total, count, fps_get_img):
        time_start = time.time()
        while not self.isExitApp:
            if share_buffer_queue.empty() is False:
                image = share_buffer_queue.get()
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
                    fps_show_img = count_total.value/time_elapsed
                    cv2.putText(image, f"fps_show_img:{fps_show_img}", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                    cv2.putText(image, f"fps_get_img:{fps_get_img.value}", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                    cv2.imshow("image", image)
                    key = cv2.waitKey(1) % 0xFF
                    if key == ord('q'):
                        self.isExitApp = True
                        break
                    #cv2.waitKey(0)

              
if __name__ == "__main__":
    
    get_image = GetImageFromSocketUDP()
    count_fail = multiprocessing.Value('i',0)
    count_total = multiprocessing.Value("i",0)
    count = multiprocessing.Value('i', 0)
    fps_get_img = multiprocessing.Value('i', 0)
    share_buffer_queue = multiprocessing.Queue()
    share_buffer_condition = multiprocessing.Condition()
    process_get_img = multiprocessing.Process(target = get_image.get_image_streaming, args = (share_buffer_queue, count_total, count_fail, fps_get_img))
    process_show_img = multiprocessing.Process(target = get_image.show_image_streaming, args = (share_buffer_queue, count_total, count, fps_get_img))
    
    process_get_img.start()
    process_show_img.start()
    
    while not get_image.isExitApp:
        time.sleep(1)
        # if count.value == 50:
        #     break

    get_image.sock.close()
    # process_show_img.join()
    # process_get_img.join()
    
    process_show_img.terminate()
    process_get_img.terminate()
#  /home/greystone/StorageWall/miniconda3/envs/ml_subapp/bin/python /home/greystone/StorageWall/apps/ml-subapp/appPython/modules/realtime_detection/get_image_streaming_UDP.py
