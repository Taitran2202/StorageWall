import logging
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtNetwork import QTcpServer
from network.server_socket import ServerSocket
from network.server_socket import HOST_DEFINE, PORT_DEFINE


class ServerManger(QTcpServer):
    slot_receive_hello_cmd_from_client = pyqtSlot(str)
    signal_send_data_to_client = pyqtSignal(dict)
    signal_receive_data_from_client = pyqtSignal(dict)
    slot_start_server = pyqtSlot()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(__name__)
        self._host_info = HOST_DEFINE
        self._port_info = PORT_DEFINE
        self._clients = []

    @pyqtSlot()
    def slot_start_server(self):
        self.log.info("start_server")
        is_start_ok = self.listen(self._host_info, self._port_info)
        self.log.info("start_server is_start_ok = {}".format(is_start_ok))
        if is_start_ok is False:
            self.log.info("self.errorString = {}".format(self.errorString()))
        return is_start_ok

    def get_sever_status(self):
        is_listening = self.isListening()
        number_of_client = len(self._clients)
        return is_listening, number_of_client

    def incomingConnection(self, socketDescriptor):
        self.log.info("New connection")
        socket = ServerSocket()
        socket.setSocketDescriptor(socketDescriptor)
        socket.signal_receive_hello_cmd_from_client.connect(self.slot_receive_hello_cmd_from_client)
        socket.disconnected.connect(self.on_disconnected)
        socket.send_hello_cmd_to_client()
        self._clients.append(socket)

    def on_disconnected(self):
        self.log.info("Disconnected")
        socket = self.sender()
        if socket in self._clients:
            self._clients.remove(socket)
            socket.deleteLater()

    @pyqtSlot(str)
    def slot_receive_hello_cmd_from_client(self, app_name):
        self.log.info("slot_receive_hello_cmd_from_client app_name = {}".format(app_name))
        socket: ServerSocket = self.sender()
        if isinstance(socket, ServerSocket) is False:
            self.log.info("slot_receive_hello_cmd_from_client unknown sender")
            return
        socket.signal_receive_hello_cmd_from_client.disconnect(self.slot_receive_hello_cmd_from_client)
        socket.signal_receive_data_from_client.connect(self.signal_receive_data_from_client)
        self.signal_send_data_to_client.connect(socket.slot_send_data_to_client)

