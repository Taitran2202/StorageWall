#include "clientsocket.h"
#include <QDataStream>
#include <QDate>
#include <QTime>
#include "logger/qlogger.h"
#include "globalvariable.h"
#include "json/jsondata.h"
#include "mlappmanager.h"

using namespace QLogger;

ClientSocket::ClientSocket(QObject *parent)
    : QTcpSocket(parent)
{
    connect(this, SIGNAL(readyRead()), this, SLOT(readFromClient()));
    nextBlockSize = 0;
    chunkSize = 128;
    moduleLogFile = g_app_name;
}

void ClientSocket::sendHelloCommmand()
{
    QVariantMap mapData;
    mapData["cmdKey"] = ML_APP_CMD_KEY_HELLO;
    mapData["uniqueID"] = "";
    mapData["mapCmdData"] = QVariantMap();
    mapData["timeoutRuntime"] = 30*1000;
    mapData["timeoutInQueue"] = 3*60*1000;
    mapData["isPriority"] = false;
    JsonData jsonData;
    QByteArray data = jsonData.encodeJsonString(mapData);
    this->writeToClient(data);
}

void ClientSocket::readFromClient()
{
//    QLogger::QLog_Info(moduleLogFile, QString("readFromClient byteAvailable = %1").arg(bytesAvailable()));
    if(bytesAvailable()) {
        this->dataBuffer.append(this->readAll());
    } else {
        return;
    }
    this->processClientData();
}

void ClientSocket::processClientData()
{
    QDataStream in(this->dataBuffer);
    in.setVersion(QDataStream::Qt_5_9);

    quint16 startFlagValue = 0xFF55;
    quint8 headerSize = sizeof(startFlagValue) + sizeof(this->nextBlockSize);
    if (this->nextBlockSize == 0) {
        if (this->dataBuffer.size() < headerSize)
            return;
        bool isFoundStartPoint = false;
        for(uint idx = 0; idx < this->dataBuffer.size() - sizeof(startFlagValue); idx ++) {
            in.device()->seek(idx);
            quint16 curValue;
            in >> curValue;
            if(curValue == startFlagValue) {
                in.device()->seek(0);
                in.skipRawData(idx + sizeof(startFlagValue));
                in >> this->nextBlockSize;
                this->dataBuffer.remove(0, idx);
                isFoundStartPoint = true;
                break;
            }
        }
        if(isFoundStartPoint == false) {
            QLogger::QLog_Info(moduleLogFile, QString("readFromClient cannot found start point 0xFF55 - "
                                                   "do clear buffer, leave two final bytes to continue check new flag - "
                                                   "this->dataBuffer.size() = %1").arg(this->dataBuffer.size()));
            this->dataBuffer.remove(0, this->dataBuffer.size() - sizeof(startFlagValue));
        }
    }
    if (this->dataBuffer.size() < this->nextBlockSize + headerSize)
        return;

    quint8 appID;
    QDataStream inStartPackage(this->dataBuffer);
    inStartPackage.setVersion(QDataStream::Qt_5_9);
    inStartPackage.skipRawData(headerSize);
    inStartPackage >> appID;
    QLogger::QLog_Info(moduleLogFile, QString("readFromClient mlAppKey = '%1'; this->nextBlockSize = %2").arg(appID).arg(this->nextBlockSize));
    quint16 checksumByClient;
    int dataLength = this->nextBlockSize - sizeof(appID) - sizeof(checksumByClient);
    char rawData[dataLength];
    int byteRead = inStartPackage.readRawData(rawData, dataLength);
    if(byteRead != dataLength) {
        QLogger::QLog_Info(moduleLogFile, QString("readFromClient not read enough data: byteRead = '%1'; dataLength = '%2'")
                           .arg(byteRead).arg(dataLength));
    } else {
        quint16 checksumOnData = qChecksum(rawData, dataLength);
        inStartPackage >> checksumByClient;
        if(checksumOnData != checksumByClient) {
            QLogger::QLog_Info(moduleLogFile, QString("readFromClient invalid checksum! checksumOnData = %1; checksumByClient = %2")
                               .arg(checksumOnData).arg(checksumByClient));
        } else {
            QLogger::QLog_Info(moduleLogFile, QString("readFromClient dataLength = '%1'").arg(dataLength));
            QLogger::QLog_Info(moduleLogFile, QString("readFromClient rawData = '%1'").arg(rawData));

            QByteArray clientData = QByteArray::fromRawData(rawData, dataLength);
            QLogger::QLog_Info(moduleLogFile, QString("readFromClient clientData = '%1'").arg(clientData.data()));
            if(this->appName == ML_APP_NAME) {
                this->newDataFromMLApp(clientData);
            } else {
                this->newDataFromClient(clientData);
            }
        }
    }
    this->dataBuffer.remove(0, nextBlockSize + headerSize);
    if(this->dataBuffer.size() > 0) {
        QLogger::QLog_Info(moduleLogFile, QString("readFromClient size of bytes received = %1; this->dataBuffer remaining = %2")
                           .arg(nextBlockSize).arg(this->dataBuffer.size()));
    }
    nextBlockSize = 0;
    if(this->dataBuffer.isEmpty() == false) {
        this->processClientData();
    }
}

void ClientSocket::writeToClient(const QByteArray &data)
{
    quint16 startFlagValue = 0xFF55;
    quint8 headerSize = sizeof(startFlagValue) + sizeof(this->nextBlockSize);
    QByteArray block;
    QDataStream out(&block, QIODevice::WriteOnly);
    out.setVersion(QDataStream::Qt_5_9);
    out << startFlagValue;
    out << quint16(0);
    out << (quint8)this->appID;
    out.writeRawData(data.data(), data.size());
    quint16 checksumOnData = qChecksum(data.data(), data.size());
    out << checksumOnData;
    out.device()->seek(sizeof(quint16));
    out << quint16(block.size() - headerSize);
    QLogger::QLog_Info(moduleLogFile, QString("writeToClient str block = %1").arg(QString(block.data())));
    qint64 byteWritten = write(block);
    QLogger::QLog_Info(moduleLogFile, QString("writeToClient byteWritten = %1; data = %2").arg(byteWritten).arg(QString(data.toHex().toUpper())));
}

void ClientSocket::newDataFromClient(const QByteArray &data)
{
    QLogger::QLog_Info(moduleLogFile, QString("ClientSocket::readFromClient clientData = '%1'").arg(data.data()));
    JsonData jsonData;
    QVariantMap mapData = jsonData.decodeJsonString(data);
    QString cmdKey = mapData.value("cmdKey").toString();
    QVariantMap mapCmdData = mapData.value("mapCmdData").toMap();
    QString appName = mapCmdData.value("appName").toString();
    if(cmdKey == ML_APP_CMD_KEY_HELLO) {
        emit this->signalReceiveHelloCommandFromSubApp(appName);
        QLogger::QLog_Info(moduleLogFile, QString("ClientSocket::readFromClient receive hello cmd from subapp; appName = '%1'").arg(appName));
        this->appName = appName;
    } else {
        QLogger::QLog_Info(moduleLogFile, QString("ClientSocket::readFromClient unknown cmdKey = '%1'").arg(cmdKey));
    }
}

void ClientSocket::slotSendDataToMLApp(const ML_APP_COMMAND &cmd)
{
    QVariantMap mapData;
    mapData["cmdKey"] = cmd.cmdKey;
    mapData["uniqueID"] = cmd.uniqueID;
    mapData["mapCmdData"] = cmd.mapCmdData;
    mapData["timeoutRuntime"] = cmd.timeoutRuntime;
    mapData["timeoutInQueue"] = cmd.timeoutInQueue;
    mapData["isPriority"] = cmd.isPriority;
    JsonData jsonData;
    QByteArray data = jsonData.encodeJsonString(mapData);
    this->writeToClient(data);
}

void ClientSocket::newDataFromMLApp(const QByteArray &data)
{
    QLogger::QLog_Info(moduleLogFile, QString("MLAppClient::readFromClient clientData = '%1'").arg(data.data()));
    JsonData jsonData;
    QVariantMap mapData = jsonData.decodeJsonString(data);
    ML_APP_COMMAND cmd;
    cmd.cmdKey = mapData.value("cmdKey").toString();
    cmd.uniqueID = mapData.value("uniqueID").toString();
    cmd.exitCode = mapData.value("exitCode", -1).toInt();
    cmd.isTimeoutRuntime = mapData.value("isTimeoutRuntime", false).toBool();
    cmd.isFromClient = true;
    cmd.mapCmdData = mapData.value("mapCmdData").toMap();
    emit this->signalReceiveDataFromMLApp(cmd);
}
