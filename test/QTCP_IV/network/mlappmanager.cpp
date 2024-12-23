#include "mlappmanager.h"
#include "globalvariable.h"
#include <QDateTime>
#include <QMutexLocker>
#include <QVariantMap>

using namespace QLogger;

MLAppManager *MLAppManager::INSTANCE = NULL;

MLAppManager *MLAppManager::getInstance()
{
    if(!INSTANCE) {
        INSTANCE = new MLAppManager();
    }
    return INSTANCE;
}

MLAppManager::~MLAppManager()
{

}

int MLAppManager::initialize()
{
    QLogger::QLog_Info(moduleLogFile, QString("MLAppManager::initialize()"));
    //TODO: Need to provide function to check sub app running status
    return 0;
}

void MLAppManager::slotReceiveDataFromMLApp(const ML_APP_COMMAND &cmd)
{
    QMutexLocker locker(&this->lockListCmd);
    QMutableListIterator<ML_APP_COMMAND> mutableListIterator(this->listCmd);
    while(mutableListIterator.hasNext()) {
        ML_APP_COMMAND curCmd = mutableListIterator.next();
        if(cmd.uniqueID == curCmd.uniqueID) {
            curCmd = cmd;
            curCmd.isFinished = true;
            mutableListIterator.setValue(curCmd);
            break;
        }
    }
}

int MLAppManager::handshakeWithMLApp(ML_CMD_HANDSHAKE_REQUEST request, ML_CMD_HANDSHAKE_RESPONSE &response)
{
    QVariantMap mapRequest;
    mapRequest["masterPid"] = request.masterPid;
    mapRequest["pathDebug"] = request.pathDebug;
    mapRequest["pathRoot"] = request.pathRoot;
    mapRequest["pathLog"] = request.pathLog;
    ML_APP_COMMAND mlAppCmd;
    mlAppCmd.cmdKey = ML_APP_CMD_KEY_HANDSHAKE;
    mlAppCmd.mapCmdData = mapRequest;
    this->appendCommand(mlAppCmd);
    int res = this->waitCommandResponse(mlAppCmd);
    if(res == 0) {
        response.clientPid = mlAppCmd.mapCmdData.value("clientPid", -1).toInt();
        response.result = mlAppCmd.mapCmdData.value("result", -1).toInt();
    }
    return res;
}

int MLAppManager::analyzeSerialAndModel(ML_CMD_ANALYZE_REQUEST request, ML_CMD_ANALYZE_RESPONSE &response)
{
    QVariantMap mapRequest;
    mapRequest["pathDebug"] = request.pathDebug;
    mapRequest["fixtureType"] = request.fixtureType;
    mapRequest["moduleIndex"] = request.moduleIndex;
    QVariantList mapListImgPath;
    QList<QString>::iterator i;
    for(i = request.listImgPath.begin(); i != request.listImgPath.end(); ++i) {
        mapListImgPath.append(*i);
    }
    mapRequest["listImgPath"] = mapListImgPath;
    ML_APP_COMMAND mlAppCmd;
    mlAppCmd.cmdKey = ML_APP_CMD_KEY_ANALYZE;
    mlAppCmd.mapCmdData = mapRequest;
    this->appendCommand(mlAppCmd);
    int res = this->waitCommandResponse(mlAppCmd);
    if(res == 0) {
        response.serialNumber = mlAppCmd.mapCmdData.value("serialNumber", "").toString();
        response.modelNumber = mlAppCmd.mapCmdData.value("modelNumber", "").toString();
        response.result = mlAppCmd.mapCmdData.value("result", -1).toInt();
    }
    return res;
}

int MLAppManager::analyzeDentAndModel(QVariantMap mapRequest)
{
    QLogger::QLog_Info(moduleLogFile, QString("analyzeDentAndModel: ------------- begin ------------------"));

    ML_APP_COMMAND mlAppCmd;
//    QVariantMap mapRequest;
//    mapRequest["stationId"] = station_id;
//    mapRequest["phoneId"] = phone_id;
//    mapRequest["pathHistory"] = path_history;
//    mapRequest["detectId"] = detect_id;
//    mapRequest["tensorScriptCmd"] = tensorCommand;

    mlAppCmd.cmdKey = ML_APP_CMD_KEY_ANALYZE;
    mlAppCmd.mapCmdData = mapRequest;
    int res = executeCommand(mlAppCmd);
    QLogger::QLog_Info(moduleLogFile, QString("analyzeDentAndModel: res: %1").arg(res));
    QLogger::QLog_Info(moduleLogFile, QString("analyzeDentAndModel: ------------- begin ------------------"));
    return res;
}

int MLAppManager::executeCommand(ML_APP_COMMAND &subAppCmd)
{
    subAppCmd.isFinished = false;
    subAppCmd.isRunning = false;
    subAppCmd.isTimeoutInQueue = false;
    subAppCmd.isTimeoutRuntime = false;
    QTime elapsedTime;
    elapsedTime.start();
    QLogger::QLog_Info(moduleLogFile, QString("executeCommand: %1 - %2").arg(subAppCmd.cmdKey, subAppCmd.uniqueID));
    if(this->cancelFlag) {
        QLogger::QLog_Info(moduleLogFile, QString("executeCommand: %1 CANCELLED").arg(subAppCmd.cmdKey, subAppCmd.uniqueID));
        return -1;
    }
    this->appendCommand(subAppCmd);
    QLogger::QLog_Info(moduleLogFile, QString("Added commandID: %1; uniqueID: %2").arg(subAppCmd.cmdKey, subAppCmd.uniqueID));
    bool res = this->waitCommandResponse(subAppCmd);
    QLogger::QLog_Info(moduleLogFile, QString("commandID: %1 waitCommandResponse END elapsedTime = %2").arg(subAppCmd.uniqueID).arg((float)elapsedTime.elapsed()/1000));
    return res;
}

void MLAppManager::appendCommand(ML_APP_COMMAND &subAppCmd)
{
    QString uniqueID = this->createUniqueID();
    subAppCmd.uniqueID = uniqueID;
    QMutexLocker locker(&this->lockListCmd);
    this->listCmd.append(subAppCmd);
    QLogger::QLog_Info(moduleLogFile, QString("appendCommand: cmdKey = %1; commandID %2").arg(subAppCmd.cmdKey, uniqueID));
    QLogger::QLog_Info(moduleLogFile, QString("appendCommand: this->listCmd.count() = %1").arg(this->listCmd.count()));
}

void MLAppManager::clearCommandInQueue()
{
    QMutexLocker locker(&this->lockListCmd);
    QMutableListIterator<ML_APP_COMMAND> mutableListIterator(this->listCmd);
    while(mutableListIterator.hasNext()) {
        ML_APP_COMMAND curCmd = mutableListIterator.next();
        if(curCmd.isRunning == false) {
            mutableListIterator.remove();
        }
    }
}

QString MLAppManager::createUniqueID()
{
    this->countCmd++;
    QString uniqueID = QString("%1_%2").arg(QDateTime::currentMSecsSinceEpoch()).arg(this->countCmd);
    if(this->countCmd >= UINT32_MAX) {
        this->countCmd = 0;
    }
    return uniqueID;
}

int MLAppManager::waitCommandResponse(ML_APP_COMMAND &subAppCmd)
{
    int res = -1;
    int countTime = 0;
    QTime elapsedTime;
    elapsedTime.start();
    while(this->exitFlag == false) {
        bool isFoundCmd = false;
        bool isCmdFinished = false;
        bool isCmdCancelled = false;
        countTime++;
        if(countTime > 10000) {
            countTime = 0;
        }
        this->lockListCmd.lock();
        QMutableListIterator<ML_APP_COMMAND> mutableListIterator(this->listCmd);
        while(mutableListIterator.hasNext()) {
            ML_APP_COMMAND curCmd = mutableListIterator.next();
            if(countTime % 100 == 0) {
                QLogger::QLog_Info(moduleLogFile, QString("commandID: %1 waiting elapsedTime = %2s").arg(subAppCmd.uniqueID).arg((float)elapsedTime.elapsed()/1000));
            }
            if(curCmd.uniqueID == subAppCmd.uniqueID) {
                if(curCmd.isCancelled == true) {
                    subAppCmd = curCmd;
                    mutableListIterator.remove();
                    isCmdCancelled = true;
                } else if (curCmd.isFinished == true) {
                    subAppCmd = curCmd;
                    mutableListIterator.remove();
                    isCmdFinished = true;
                }
                isFoundCmd = true;
                break;
            }
        }
        this->lockListCmd.unlock();
        if(isCmdCancelled ==  true) {
            res = 0;
            QLogger::QLog_Info(moduleLogFile, QString("commandID: %1 waitCommandResponse cancelled! isAppRunning = %2")
                               .arg(subAppCmd.uniqueID).arg(this->isAppRunning?"true":"false"));
            break;
        } else if(isCmdFinished ==  true) {
            res = 0;
            QLogger::QLog_Info(moduleLogFile, QString("commandID: %1 waitCommandResponse OK").arg(subAppCmd.uniqueID));
            break;
        } else if(isFoundCmd == false) {
            res = -1;
            QLogger::QLog_Info(moduleLogFile, QString("commandID: %1 waitCommandResponse in queue NOT FOUND").arg(subAppCmd.uniqueID));
            break;
        }
        usleep(500000);
    }
    return res;
}

MLAppManager::MLAppManager() : QThread()
{
    qRegisterMetaType<ML_APP_COMMAND>();
    this->exitFlag = false;
    this->cancelFlag = false;
    this->countCmd = 0;
    this->isAppRunning = false;
    moduleLogFile = g_app_name;
}

void MLAppManager::run()
{
    QLogger::QLog_Info(moduleLogFile, QString("############### BEGIN MLAppManager ################"));
    QTime elapsedTime;
    elapsedTime.start();
    int countTime = 0;
    while(this->exitFlag == false) {
        usleep(10000);
        countTime++;
        if(countTime > 10000) {
            countTime = 0;
        }
//        if(countTime % 10 == 0) {
//            QLogger::QLog_Info(moduleLogFile, QString("Total command in the list: %1").arg(this->listCmd.count()));
//        }
        if(this->listCmd.count() == 0)
            continue;
        if(this->isAppRunning == false) {
            QLogger::QLog_Info(moduleLogFile, QString("this->isAppRunning == false"));
            this->lockListCmd.lock();
            for(int i = 0; i < this->listCmd.count(); i++) {
                ML_APP_COMMAND curCmd = this->listCmd.at(i);
                curCmd.isCancelled = true;
                curCmd.isFinished = true;
                this->listCmd.replace(i, curCmd);
            }
            this->lockListCmd.unlock();
            continue;
        }

        this->lockListCmd.lock();
        QList<ML_APP_COMMAND> listCmdCopy;
        QList<ML_APP_COMMAND> listCmdPriority;
        QList<ML_APP_COMMAND> listCmdNormal;
        QListIterator<ML_APP_COMMAND> listIterator(this->listCmd);
        while(listIterator.hasNext()) {
            ML_APP_COMMAND curCmd = listIterator.next();
            if(curCmd.isPriority == true) {
                listCmdPriority.append(curCmd);
            } else {
                listCmdNormal.append(curCmd);
            }
        }
        listCmdPriority.append(listCmdNormal);
        listCmdCopy = listCmdPriority;
        this->lockListCmd.unlock();
//        if(countTime % 100 == 0) {
//            for(int i = 0; i < listCmdCopy.count(); i++) {
//                ML_APP_COMMAND curCmd = listCmd.at(i);
//                QLogger::QLog_Info(moduleLogFile, QString("cmdKey = %1; cmdID").arg(curCmd.cmdKey).arg(curCmd.cmdID));
//            }
//        }
        QMutableListIterator<ML_APP_COMMAND> mutableListIterator(listCmdCopy);
        while(mutableListIterator.hasNext()) {
            ML_APP_COMMAND curCmd = mutableListIterator.next();
            if(curCmd.isFinished == false && curCmd.isRunning == false) {
                if(this->cancelFlag == false) {
                    QLogger::QLog_Info(moduleLogFile, QString("Add commandID %1 to Subapp").arg(curCmd.cmdKey));
                    curCmd.isRunning = true;
                    mutableListIterator.setValue(curCmd);
                    //emit signal to send command to sub app via network
                    emit this->signalSendDataToMLApp(curCmd);
                } else if(this->cancelFlag == true) {
                    QLogger::QLog_Info(moduleLogFile, QString("Cancel cmdKey %1; cmdID = %2").arg(curCmd.cmdKey, curCmd.uniqueID));
                    curCmd.isFinished = true;
                    curCmd.isRunning = false;
                    curCmd.isCancelled = true;
                    mutableListIterator.setValue(curCmd);
                }
            } else if(curCmd.isFinished == false) {
                //tracking running command
                //detect timeoutInQueue
                //detect cancel and set force stop to commands in sub app
                if(this->cancelFlag == false) {
                    if(elapsedTime.elapsed() > curCmd.timeoutInQueue) {
                        curCmd.isTimeoutInQueue = true;
                        curCmd.isFinished = true;
                        curCmd.isRunning = false;
                        mutableListIterator.setValue(curCmd);
                        //emit signal to send force stop command to sub app via network
                    } else {
//                    QLogger::QLog_Info(moduleLogFile, QString("Running commandID %1").arg(curCmd.cmdKey));
                    }
                } else if(this->cancelFlag == true) {
                    QLogger::QLog_Info(moduleLogFile, QString("Force stop running command cmdKey %1; cmdID = %2").arg(curCmd.cmdKey, curCmd.uniqueID));
                    curCmd.isFinished = true;
                    curCmd.isRunning = false;
                    curCmd.isCancelled = true;
                    mutableListIterator.setValue(curCmd);
                    //emit signal to send force stop command to sub app via network
                }
            }
        }
        this->lockListCmd.lock();
        mutableListIterator.toFront();
        while(mutableListIterator.hasNext()) {
            ML_APP_COMMAND curCmd = mutableListIterator.next();
            for(int i = 0; i < this->listCmd.count(); i++) {
                ML_APP_COMMAND oldCmd = this->listCmd.at(i);
                if(curCmd.uniqueID == oldCmd.uniqueID) {
                    this->listCmd.replace(i, curCmd);
                    break;
                }
            }
        }
        this->lockListCmd.unlock();
    }
}
