import logging
from PyQt5.QtCore import QObject, QDataStream, qChecksum, QIODevice, \
    pyqtSlot, pyqtSignal, QByteArray
from PyQt5.QtNetwork import QTcpSocket, QHostAddress
import json
from globals import ML_APP_CMD_KEY_HELLO


HOST_DEFINE = QHostAddress.LocalHost
PORT_DEFINE = 1236
START_FLAG_DEFINE = 0xFF55
START_FLAG_SIZE_DEFINE = 2
HEADER_SIZE_DEFINE = 4
APP_ID_SIZE_DEFINE = 1
CHECKSUM_SIZE_DEFINE = 2
APP_ID_DEFINE = 1


class ServerSocket(QTcpSocket):
    slot_send_data_to_client = pyqtSlot(dict)
    signal_receive_hello_cmd_from_client = pyqtSignal(str)
    signal_receive_data_from_client = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(__name__)
        self._socket_setup()
        self._data_buffer = bytes()
        self._next_block_size = 0
        self._host_info = HOST_DEFINE
        self._port_info = PORT_DEFINE
        self._is_connected = False
        self._app_name = ""

    @property
    def host_info(self):
        return self._host_info

    @host_info.setter
    def host_info(self, address):
        self._host_info = address

    @property
    def port_info(self):
        return self._port_info

    @port_info.setter
    def port_info(self, port_num):
        self._port_info = port_num

    @property
    def is_connected(self):
        return self._is_connected

    @pyqtSlot(dict)
    def slot_send_data_to_client(self, data_dict):
        data = json.dumps(data_dict).encode()
        self.write_to_client(data)

    def send_hello_cmd_to_client(self):
        data_dict = dict()
        data_dict["cmdKey"] = ML_APP_CMD_KEY_HELLO
        data = json.dumps(data_dict).encode()
        self.write_to_client(data)

    def _socket_setup(self):
        self.setSocketOption(QTcpSocket.KeepAliveOption, 1)
        self.connected.connect(self.on_connected)
        self.disconnected.connect(self.on_disconnected)
        self.readyRead.connect(self.on_readyRead)
        self.error.connect(self.on_error)
        self.stateChanged.connect(self.on_stateChanged)

    def disconnect(self):
        self.log.info("Disconnecting")
        self.disconnectFromHost()

    def on_readyRead(self):
        if self.bytesAvailable():
            self._data_buffer += self.readAll()

        self.log.debug(self._data_buffer)
        self.processClientData()

    def processClientData(self):
        if len(self._data_buffer) < HEADER_SIZE_DEFINE:
            return
        reader = QDataStream(self._data_buffer)
        reader.setVersion(QDataStream.Qt_5_9)
        if self._next_block_size == 0:
            if len(self._data_buffer) < HEADER_SIZE_DEFINE:
                return
            is_found_start_point = False
            for idx in range(0, len(self._data_buffer) -
                             START_FLAG_SIZE_DEFINE):
                reader.device().seek(idx)
                start_flag = reader.readUInt16()
                if start_flag == START_FLAG_DEFINE:
                    reader.device().seek(0)
                    reader.skipRawData(idx + START_FLAG_SIZE_DEFINE)
                    self._next_block_size = reader.readUInt16()
                    self._data_buffer = self._data_buffer[idx:]
                    is_found_start_point = True
                    break
            if not is_found_start_point:
                self.log.info("readFromClient cannot found start flag, \
                              do clear buffer, leave two final bytes to \
                              continue checking new flag")
                self._data_buffer = self._data_buffer[-START_FLAG_SIZE_DEFINE:]

        if len(self._data_buffer) < self._next_block_size + HEADER_SIZE_DEFINE:
            return
        readerPayload = QDataStream(self._data_buffer)
        readerPayload.setVersion(QDataStream.Qt_5_9)
        readerPayload.skipRawData(HEADER_SIZE_DEFINE)
        app_id = reader.readUInt8()
        self.log.info("app_id = {}".format(app_id))
        payload_size = self._next_block_size - APP_ID_SIZE_DEFINE - \
            CHECKSUM_SIZE_DEFINE
        payload_data = reader.readRawData(payload_size)
        checksum_by_server = reader.readUInt16()
        checksum_on_data = qChecksum(payload_data)
        print(checksum_by_server)
        print(checksum_on_data)
        if checksum_by_server == checksum_on_data:
            data_dict = json.loads(payload_data.decode("UTF8"))
            print(data_dict)
            if "cmdKey" in data_dict.keys():
                print(data_dict["cmdKey"])
            self.new_data_from_client(payload_data)
        else:
            self.log.info("readFromClient invalid checksum! \
                           checksumOnData = {}; checksumByClient = {}"
                          .format(checksum_on_data, checksum_by_server))
        self._data_buffer = self._data_buffer[self._next_block_size +
                                              HEADER_SIZE_DEFINE:]
        if len(self._data_buffer) > 0:
            self.log.info("_data_buffer remaining = {}"
                          .format(self._data_buffer))
        self._next_block_size = 0
        if len(self._data_buffer) > 0:
            self.processClientData()

    def new_data_from_client(self, data: bytes):
        self.log.debug("new_data_from_client {}".format(data))
        data_dict = json.loads(data.decode("UTF8"))
        if "cmdKey" not in data_dict.keys():
            self.log.info("new_data_from_client invalid data {}".format(data))
            return
        cmd_key = data_dict["cmdKey"]
        if cmd_key == "hello":
            mapCmdData = data_dict.get("mapCmdData", {})
            self._app_name = mapCmdData.get("appName", "Unknown")
            self.log.info("Receive hello from server from app_name = {}".format(self._app_name))
            self.signal_receive_hello_cmd_from_client.emit(self._app_name)
        else:
            self.log.info("Receive cmdKey {} from ml app"
                          .format(data_dict["cmdKey"]))
            self.signal_receive_data_from_client.emit(data_dict)

    def write_to_client(self, data: bytes):
        self.log.debug("data to client = {}".format(data))
        data_buffer = QByteArray()
        writer = QDataStream(data_buffer, QIODevice.WriteOnly)
        writer.setVersion(QDataStream.Qt_5_9)
        writer.writeUInt16(START_FLAG_DEFINE)
        writer.writeUInt16(0)
        writer.writeUInt8(APP_ID_DEFINE)
        writer.writeRawData(data)
        checksum_on_data = qChecksum(data)
        writer.writeUInt16(checksum_on_data)
        writer.device().seek(START_FLAG_SIZE_DEFINE)
        writer.writeUInt16(len(data_buffer) - HEADER_SIZE_DEFINE)
        byte_written = self.write(data_buffer)
        self.log.info("byte_written = {}".format(byte_written))
        self.log.debug("data_buffer = {}".format(data_buffer))

    def on_connected(self):
        self.log.info("Connected")
        self._is_connected = True

    def on_disconnected(self):
        self.log.info("Disconnected")
        self._is_connected = False

    def on_error(self, error):
        self.log.error("Error {}:{}".format(error, self.errorString()))

    def on_stateChanged(self, state):
        states = ["Unconnected", "HostLookup", "Connecting", "Connected",
                  "Bound", "Closing", "Listening"]
        self.log.debug("State changed to {}".format(states[state]))
