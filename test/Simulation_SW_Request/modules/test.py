import socket
import struct
import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QElapsedTimer

BUF_LEN = 4096
RECEIVER_PORT = 12345
isExitApp = False

def thread_append_excel_debug():
    print("[threadAppendExcelDebug]  START >>>")

    display = QDialog()
    grid = QGridLayout(display)
    lb = QLabel(display)
    grid.addWidget(lb)
    lb.setUpdatesEnabled(True)
    display.show()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as e:
        print("Socket creation error:", e)
        return

    server_addr = ('', RECEIVER_PORT)
    try:
        sock.bind(server_addr)
    except socket.error as e:
        print("Socket bind error:", e)
        return

    elaped = QElapsedTimer()
    elaped.start()

    while not isExitApp:
        buffer = bytearray(BUF_LEN)
        while True:
            recv_msg_size, client_addr = sock.recvfrom_into(buffer)
            if recv_msg_size <= struct.calcsize('i'):
                break

        total_pack = struct.unpack('i', buffer[:struct.calcsize('i')])[0]
        print("Expecting length of packs:", total_pack)

        longbuf = bytearray(4096 * total_pack)
        for i in range(total_pack):
            recv_msg_size, client_addr = sock.recvfrom_into(buffer)
            if recv_msg_size == 4096:
                longbuf[i * 4096:(i + 1) * 4096] = buffer[:4096]

        print("Received packet")

        raw_data = np.frombuffer(longbuf, dtype=np.uint8)
        frame = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        if elaped.elapsed() > 10000:
            cv2.imwrite("/home/greystone/imageRc.jpg", frame)
            elaped.restart()

        if frame.size == 0:
            print("Decode failure!")
            continue

        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixel = QPixmap.fromImage(img)
        lb.setPixmap(pixel)
        display.repaint()

    print("[threadAppendExcelDebug]  END >>>")

class Controller:
    def __init__(self):
        self.port = None
        self.address = None
        self.sock = None
        self.server_address = None
        self.is_streaming = False
        self.shm_snapshort_obj = None  # Placeholder for the actual object

    def start_streaming(self):
        self.port = self.shm_snapshort_obj.get_udp_port()
        self.address = self.shm_snapshort_obj.get_udp_address()
        print(f"[startStreaming] Start streaming with IP = {self.address}, port = {self.port}")

        if self.port < 0:
            print("[startStreaming] Invalid port. port < 0. Start stream [FAILED]")
            return False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            print("[startStreaming] Socket creation error. Start stream [FAILED]", e)
            return False

        self.server_address = (self.address, self.port)
        try:
            socket.inet_pton(socket.AF_INET, self.address)
        except socket.error as e:
            print("[startStreaming] Invalid address. Address not supported. Start stream [FAILED]", e)
            return False

        self.is_streaming = True
        return True

def send_image(sock, server_address, image, image_height, image_width, package_size):
    new_image = cv2.Mat(image_height, image_width, cv2.CV_8UC4, image.data, cv2.Mat.AUTO_STEP)
    encoded_frame = cv2.imencode(".jpg", new_image)[1].tobytes()
    total_package_count = 1 + (len(encoded_frame) - 1) // package_size

    sock.sendto(struct.pack('i', total_package_count), server_address)
    for i in range(total_package_count):
        sock.sendto(encoded_frame[i * package_size:(i + 1) * package_size], server_address)


