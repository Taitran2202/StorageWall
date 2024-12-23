#include "servermanager.h"
#include <QTcpSocket>
#include <QDebug>
#include "logger/qlogger.h"
#include "globalvariable.h"
#include "clientsocket.h"
#include "mlappmanager.h"

using namespace QLogger;

ServerManager::ServerManager(QObject *parent)
    : QTcpServer(parent)
{
    moduleLogFile = g_app_name;
}

int ServerManager::initialize()
{
    if(!this->listen(QHostAddress::LocalHost, EXPOSE_TCP_SERVER_PORT)) {
        QLogger::QLog_Info(moduleLogFile, QString("Server could not start at port = '%1' - errorString = '%2'")
                           .arg(EXPOSE_TCP_SERVER_PORT).arg(this->errorString()));
        return -1;
    } else {
        QLogger::QLog_Info(moduleLogFile, QString("Server started at port = '%1'").arg(EXPOSE_TCP_SERVER_PORT));
    }
    return 0;
}

void ServerManager::incomingConnection(qintptr handle)
{
    ClientSocket *socket = new ClientSocket(this);
    socket->setSocketOption(QAbstractSocket::KeepAliveOption, true);
    socket->setSocketDescriptor(handle);
    connect(socket, SIGNAL(disconnected()), this, SLOT(slotSocketDisconnected()));
    connect(socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(slotSockerError(QAbstractSocket::SocketError)));
    connect(socket, SIGNAL(signalReceiveHelloCommandFromSubApp(QString)), this, SLOT(slotReceiveHelloCommandFromSubApp(QString)));
    QLogger::QLog_Info(moduleLogFile, QString("incomingConnection unknown socketId = %1").arg(handle));
    socket->sendHelloCommmand();

}

void ServerManager::slotSocketDisconnected()
{
    ClientSocket *socket = (ClientSocket*)this->sender();
    if(socket == NULL) {
        QLogger::QLog_Info(moduleLogFile, QString("slotSocketDisconnected invalid socket pointer"));
        return;
    }
    QLogger::QLog_Info(moduleLogFile, QString("slotSocketDisconnected appName = %1").arg(socket->getAppName()));
    if(socket->getAppName() == ML_APP_NAME) {
        MLAppManager *mlAppManager = MLAppManager::getInstance();
        mlAppManager->slotUpdateRunningStatus(false);
    }
    socket->deleteLater();
}

void ServerManager::slotSockerError(QAbstractSocket::SocketError socketError)
{
    ClientSocket *socket = (ClientSocket*)this->sender();
    if(socket == NULL) {
        QLogger::QLog_Info(moduleLogFile, QString("slotSockerError invalid socket pointer"));
        return;
    }
    QLogger::QLog_Info(moduleLogFile, QString("slotSocketDisconnected error id = %1").arg(socketError));
    QLogger::QLog_Info(moduleLogFile, QString("slotSocketDisconnected errorString = %1").arg(socket->errorString()));
}

void ServerManager::slotReceiveHelloCommandFromSubApp(const QString &appName)
{
    ClientSocket *socket = (ClientSocket*)this->sender();
    if(socket == NULL) {
        QLogger::QLog_Info(moduleLogFile, QString("slotReceiveHelloCommandFromSubApp invalid socket pointer"));
        return;
    }
    if(appName == "ML_APP") {
        MLAppManager *mlAppManager = MLAppManager::getInstance();
        disconnect(socket, SIGNAL(signalReceiveHelloCommandFromSubApp(QString)), this, SLOT(slotReceiveHelloCommandFromSubApp(QString)));
        connect(mlAppManager, SIGNAL(signalSendDataToMLApp(ML_APP_COMMAND)), socket, SLOT(slotSendDataToMLApp(ML_APP_COMMAND)));
        connect(socket, SIGNAL(signalReceiveDataFromMLApp(ML_APP_COMMAND)), mlAppManager, SLOT(slotReceiveDataFromMLApp(ML_APP_COMMAND)));
        mlAppManager->slotUpdateRunningStatus(true);
    } else {
        QLogger::QLog_Info(moduleLogFile, QString("slotReceiveHelloCommandFromSubApp from unknown appName = %1").arg(appName));
    }
}

