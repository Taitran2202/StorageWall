
import logging
import os


class GV(object):
    '''    Global variables    '''
    def __init__(self):
        basedir = os.path.dirname(__file__)
        self._version = '1.00.01'
        #self._path_root = "/home/greystone/StorageWall/image_debug/LongVu_debug"
        self._path_root = "/home/greystone/StorageWall/logs"
        self._path_to_base_dir = basedir
        self._path_to_debug_folder = os.path.join(self._path_root, "ml_debug")
        self._path_to_logs_folder = os.path.join(self._path_root, "ml_logs")
        self._path_ml_model_folder = os.path.join(self._path_root, "ML_MODULE/ml_models")
        self._path_config_file = "/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json"

    @property
    def version(self):
        return self._version

    @property
    def path_to_base_dir(self):
        return self._path_to_base_dir

    @property
    def path_to_debug_folder(self):
        return self._path_to_debug_folder

    @property
    def path_to_logs_folder(self):
        return self._path_to_logs_folder

    @property
    def path_root(self):
        return self._path_root

    @property
    def path_ml_model_folder(self):
        return self._path_ml_model_folder
    
    @property
    def path_config_file(self):
        return self._path_config_file

    def write_log(message, scope="other", level=logging.INFO):
        """Write formatted log message to stderr."""
        logger = logging.getLogger(scope)
        logger.log(level, message)


ML_APP_CMD_KEY_HELLO = "hello"
ML_APP_CMD_KEY_HANDSHAKE = "handshake"
ML_APP_CMD_KEY_ANALYZE = "analyze"
ML_APP_CMD_KEY_PHYSICAL_INVENTORY = "physical_inventory"
ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY = "stop_physical_inventory"
ML_APP_CMD_KEY_STOP_ANALYZE = "stop_analyze"
ML_APP_NAME = "ML_APP"
ML_ANALYZE_APP_PORT = 1236 # 1236=deploy, 1234=test

ML_CONFIG_QUANTITY_COLUMNS = 39
ML_CONFIG_QUANTITY_ROWS = 76

class ML_APP_COMMAND:
    cmdKey = ""
    uniqueID = ""
    slot_id = ""
    isTimeoutInQueue = False
    isFinished = False
    isTimeoutRuntime = False
    isRunning = False
    isPriority = False
    isCancelled = False
    isFromClient = False
    isAddedInQueue = False
    timeoutInQueue = 40*1000
    timeoutRuntime = 20*1000
    exitCode = 0
    mapCmdData = dict()


class ML_CMD_HANDSHAKE_REQUEST:
    masterPid = -1
    pathDebug = ""
    pathRoot = ""
    pathLog = ""


class ML_CMD_HANDSHAKE_RESPONSE:
    clientPid = -1
    result = -1


class ML_CMD_ANALYZE_REQUEST:
    positionToDetect = '' # position of phone slot
    pathToImg = '' # path to image for detect phone slot
    pathToList = '' # path to text file for scan barcode
    pathToDebug = '' # path to debug
    requestType = '' # define request is scan barcode or detect phone slot
    pathToResult = '' # path to result json file of docker_scanbarcode
    transactionID = '' # transactionID of current phone
    pathToImg1 = '' # path to image for calib
    pathToImg2 = '' # path to image for calib
    positionToCalib = '' # wall left or wall right
    x_position = '' # x position of current phone slot
    z_position = '' # z position of current phone slot
    x_min = -1 # x_min from SW
    x_max = 1 # x_max from SW
    z_min = -1 # y_min from SW
    z_max = 1 # y_max from SW
    numberSlotsDetector = -1 # number of slots detected from SW

class ML_CMD_ANALYZE_RESPONSE:
    result_string = ''
    result_calib = ''
    result_transactionID = ''
    slot_id = '' # slot id of current phone slot
    cmdUniqueID = ''
    result = 0


class STRUCT_ANALYZE_INFO:
    moduleIndex = -1
    cmdUniqueID = ""
    isFinished = False
    cancelFlag = False
    request = ML_CMD_ANALYZE_REQUEST()
    response = ML_CMD_ANALYZE_RESPONSE()


class ML_CMD_HELLO_RESPONSE:
    appName = ML_APP_NAME
