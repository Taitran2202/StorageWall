#ifndef CLIENTSOCKET_H
#define CLIENTSOCKET_H

#include <QObject>
#include <QTcpSocket>
#include "mlappmanager.h"

#define EXPOSE_TCP_SERVER_PORT 1236 //1236

class ClientSocket: public QTcpSocket
{
    Q_OBJECT
public:
    ClientSocket(QObject *parent = nullptr);
    void setAppID(quint8 appID) {this->appID = appID;}
    void sendHelloCommmand();
    QString getAppName() {return this->appName;}

signals:
    void signalReceiveHelloCommandFromSubApp(const QString &appName);
    void signalReceiveDataFromMLApp(const ML_APP_COMMAND &cmd);

public slots:
    void slotSendDataToMLApp(const ML_APP_COMMAND &cmd);

protected slots:
    void readFromClient();

protected:
    void writeToClient(const QByteArray &data);
    void newDataFromClient(const QByteArray &data);
    void newDataFromMLApp(const QByteArray &data);

private:
    QByteArray dataBuffer;
    int chunkSize;
    quint16 nextBlockSize;
    quint8 appID;
    QString appName;
    QString moduleLogFile;
    void processClientData();

};

#endif // CLIENTSOCKET_H
