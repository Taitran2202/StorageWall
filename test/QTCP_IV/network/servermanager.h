#ifndef SERVERMANAGER_H
#define SERVERMANAGER_H

#include <QObject>
#include <QTcpServer>
#include "clientsocket.h"

class ServerManager : public QTcpServer
{
    Q_OBJECT
public:
    ServerManager(QObject *parent = nullptr);

    int initialize();

public slots:


private slots:
    void slotSocketDisconnected();
    void slotSockerError(QAbstractSocket::SocketError socketError);
    void slotReceiveHelloCommandFromSubApp(const QString &appName);

private:
    void incomingConnection(qintptr handle);
    QList<ClientSocket*> listSocket;
    QString moduleLogFile;

};

#endif // SERVERMANAGER_H
