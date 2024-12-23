from PyQt5.QtCore import QObject, QProcess
from globals import GV


class GeneralModel(QObject):
    def __init__(self):
        super().__init__()

    def exec_system_command(self, command):
        GV().write_log("Execute command: {}".format(command))
        process = QProcess()
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.start("bash -c \"{}\"".format(command))
        isFinish = process.waitForFinished()
        print("isFinish = {}".format(isFinish))
        exitCode = process.exitCode()
        print("exitCode = {}".format(exitCode))
        output = process.readAll().data().decode("utf-8").strip('\n')
        # print("process.readAll() = {}".format(output))
        return exitCode, output
