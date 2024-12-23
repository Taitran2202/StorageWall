import os
import logging


class GV(object):
    '''    Global variables    '''
    def __init__(self):
        basedir = os.path.dirname(__file__)
        self._version = '1.00.01'
        self._path_root = "/home/greystone/StorageWall"
        self._path_to_base_dir = basedir
        self._path_to_debug_folder = os.path.join(self._path_root, "debug")
        self._path_to_logs_folder = os.path.join(self._path_root, "logs")
        self._path_ml_models_folder = os.path.join(self._path_root, "ml_models")
        self._path_ml_environment_folder = os.path.join(self._path_root, "ml_environment")
        self._path_to_config_folder = os.path.join(self._path_root, "config")
        self._path_to_models_config = os.path.join(self._path_root, "modules","manage_models.json")

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
    def path_ml_models_folder(self):
        return self._path_ml_models_folder

    @property
    def path_ml_environment_folder(self):
        return self._path_ml_environment_folder
    
    @property
    def path_to_config_folder(self):
        return self._path_to_config_folder

    def write_log(message, scope="other", level=logging.INFO):
        """Write formatted log message to stderr."""
        logger = logging.getLogger(scope)
        logger.log(level, message)


SW_APP_NAME = "SW_APP"
CPP_SUB_APP_NAME = "CPP_SUB_APP"
ML_APP_CMD_KEY_HELLO = "hello"
ML_APP_CMD_KEY_HANDSHAKE = "handshake"
ML_APP_CMD_KEY_INFERENCE_MODEL = "inference_model"
ML_APP_CMD_KEY_ANALYZE_DEFECT = "analyze_defect"
SW_APP_NAME = "SW_APP"
CPP_SUB_APP_NAME = "CPP_SUB_APP"
ML_APP_CMD_KEY_ANALYZE = "analyze"
ML_APP_CMD_KEY_PHYSICAL_INVENTORY = "physical_inventory"
ML_APP_CMD_KEY_STOP_ANALYZE = "stop_analyze"
ML_APP_CMD_KEY_STOP_PHYSICAL_INVENTORY = "stop_physical_inventory"
ML_SUBAPP_NAME = "AnalyzeTop"


class ML_CMD_HANDSHAKE_REQUEST:
    masterPid = -1
    pathDebug = ""
    pathRoot = ""
    pathLog = ""


class ML_CMD_HANDSHAKE_RESPONSE:
    clientPid = -1
    result = -1


class ML_APP_COMMAND:
    cmdKey = ""
    uniqueID = ""
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
    x_position = 0 # x position
    z_position = 0 # z position
    numberSlotsDetector = -1


class ML_CMD_ANALYZE_RESPONSE:
    result_string = ''
    result_calib = ''
    result_transactionID = ''
    slot_id = '' # slot id of current phone slot
    result = 0


class STRUCT_ANALYZE_INFO:
    moduleIndex = -1
    cmdUniqueID = ""
    isFinished = False
    cancelFlag = False
    request = ML_CMD_ANALYZE_REQUEST()
    response = ML_CMD_ANALYZE_RESPONSE()


class ML_CMD_INFERENCE_MODEL_REQUEST:
    model_name = ""
    image_list = ""
    debug_dir  = ""


class ML_CMD_INFERENCE_MODEL_RESPONSE:
    mapRes = dict()
    result = -1


class STRUCT_INFERENCE_MODEL_INFO:
    moduleIndex = -1
    cmdUniqueID = ""
    isFinished = False
    cancelFlag = False
    request = ML_CMD_INFERENCE_MODEL_REQUEST()
    response = ML_CMD_INFERENCE_MODEL_RESPONSE()