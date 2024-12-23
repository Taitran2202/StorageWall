import socket
import time
import struct
import logging

exit_flag = False

def calculateCRC(packet, size):
        ck = 0
        for i in range(size):
            ck += packet[i]
        return (~ck + 1) & 0xFF

def subprocess_stream_offset(exit_flag):
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #host_ip = '127.0.0.1'
        host_ip = 'localhost'
        host_port = 2679
        client_port = 1790
        server_address = (host_ip,host_port)
        client_address = (host_ip,client_port)
        
        server_socket.bind(server_address)
        count = 0
        total_bytes = 11
        while exit_flag is False:
            time.sleep(1)
            count += 1
            # if share_offset_to_stream_queue.empty():
            #     continue  # Wait for frames to become available
            # offset_x, offset_y = share_offset_to_stream_queue.get()
            start_first = 0x55
            start_second = 0xFF
            offset_x = 0
            offset_y = int(3.42*count)
            
            buffer = bytearray(total_bytes)
            struct.pack_into("B", buffer, 0, start_first)
            struct.pack_into("B", buffer, 1, start_second)
            struct.pack_into("i", buffer, 2, offset_x)
            struct.pack_into("i", buffer, 6, offset_y)
            
            crc32 = calculateCRC(buffer,10)
            
            struct.pack_into("B", buffer, 10, crc32)
            
            server_socket.sendto(buffer, client_address)
            logging.info(f"Offset have been sent: {offset_x}, {offset_y}")
            print(f"Offset have been sent: {offset_x}, {offset_y}")
            
        server_socket.close()
    except Exception as e:
        logging.exception(f"Exception in subprocess_stream_offset: {e}")
    