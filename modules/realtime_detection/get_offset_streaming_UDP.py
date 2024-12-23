import socket
import struct
import cv2
import numpy as np
import binascii
import numpy as np


class getOffsetStream:
    def __init__(self):
        #self.HOST = '172.16.20.235'
        self.HOST = 'localhost'
        self.RECEIVER_PORT = 1790  # Replace with the appropriate port number
        self.BUF_LEN = 11
        self.isExitApp = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.RECEIVER_PORT))

    def calculateCRC(self, packet, size):
        ck = 0
        for i in range(size):
            ck += packet[i]
        return (~ck + 1) & 0xFF

    def get_offset_streaming(self):
        failedCount = 0
        while not self.isExitApp:
            
            buffer = bytearray(self.BUF_LEN)
            received_bytes, client_address = self.sock.recvfrom_into(buffer)
            print("here!!", buffer)
            print("received_bytes:", received_bytes)
            if received_bytes <12:
                start_first = struct.unpack("B", buffer[:1])[0]             #0x55
                print(start_first)
                start_second = struct.unpack("B", buffer[1:2])[0]           #0xFF
                print(start_second)

                if start_first == 0x55 and start_second == 0xFF:
                    offset_x = struct.unpack("i", buffer[2:6])[0]
                    print(offset_x)
                    offset_y = struct.unpack("i", buffer[6:10])[0]
                    print(offset_y)
                    crc32 = struct.unpack("B", buffer[10:11])[0]
                    print(crc32)
                else:
                    continue

                #crc32cal = self.calculate_crc32_buffer(buffer[0:10], 10)
                crc32cal = self.calculateCRC(buffer[0:10], 10)
                # print("crc32cal", crc32cal)
                if crc32cal != crc32:
                    print(f"CRC failed, CRC receive = {crc32}, CRC cal = {crc32cal}, failed count: {failedCount}")
                    failedCount += 1
                    return None, None
            else:
                continue
            return offset_x, offset_y
if __name__ == "__main__":
    get_data = getOffsetStream()
    while True:
        offset_x, offset_y = get_data.get_offset_streaming()
        if offset_x is None or offset_y is None:
            print("Get data failure!")
            continue
        print(f"offset_x: {offset_x}, offset_y: {offset_y}")

#/home/greystone/StorageWall/miniconda3/envs/ml_subapp/bin/python /home/greystone/StorageWall/apps/ml-subapp/appPython/modules/realtime_detection/get_data_streaming_UDP.py
