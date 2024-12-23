#ifndef MLAPPMANAGER_H
#define MLAPPMANAGER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QVariantMap>

#define ML_APP_CMD_KEY_HELLO                        "hello"
#define ML_APP_CMD_KEY_HANDSHAKE                    "handshake"
#define ML_APP_CMD_KEY_ANALYZE                      "analyze"
#define ML_APP_NAME                                 "ML_APP"

typedef struct ML_APP_COMMAND {
    QString cmdKey;
    QString uniqueID;
    bool isTimeoutInQueue;
    bool isFinished;
    bool isTimeoutRuntime;
    bool isRunning;
    bool isPriority;
    bool isCancelled;
    bool isFromClient;
    bool isAddedInQueue;
    bool isEnabled;
    int timeoutInQueue;
    int timeoutRuntime;
    int exitCode;
    QVariantMap mapCmdData;
    ML_APP_COMMAND() {
        isTimeoutInQueue = false;
        isFinished = false;
        isTimeoutRuntime = false;
        isRunning = false;
        isPriority = false;
        isCancelled = false;
        isFromClient = false;
        isAddedInQueue = false;
        isEnabled = false;
        timeoutInQueue = 3*60*1000;
        timeoutRuntime = 30*1000;
        exitCode = 0;
    }
}ML_APP_COMMAND;

typedef struct ML_CMD_HANDSHAKE_REQUEST {
    int masterPid;
    QString pathDebug;
    QString pathRoot;
    QString pathLog;
    ML_CMD_HANDSHAKE_REQUEST() {
        masterPid = -1;
    }
}ML_CMD_HANDSHAKE_REQUEST;

typedef struct ML_CMD_HANDSHAKE_RESPONSE {
    int clientPid;
    int result;
    ML_CMD_HANDSHAKE_RESPONSE() {
        clientPid = -1;
        result = -1;
    }
}ML_CMD_HANDSHAKE_RESPONSE;

typedef struct ML_CMD_ANALYZE_REQUEST {
    QString pathDebug;
    QList<QString> listImgPath;
    int fixtureType;
    int moduleIndex;
    ML_CMD_ANALYZE_REQUEST() {
        fixtureType = 0;
        moduleIndex = 0;
    }
}ML_CMD_ANALYZE_REQUEST;

typedef struct ML_CMD_ANALYZE_RESPONSE {
    QString serialNumber;
    QString modelNumber;
    int result;
    ML_CMD_ANALYZE_RESPONSE() {
        result = 0;
    }
}ML_CMD_ANALYZE_RESPONSE;

Q_DECLARE_METATYPE(ML_APP_COMMAND)

class MLAppManager : public QThread
{
    Q_OBJECT
public:
    static MLAppManager * getInstance();
    ~MLAppManager();
    int initialize();
    void setExitFlag(bool exitFlag) {this->exitFlag = exitFlag;}
    void setCancelFlag(bool cancelFlag) {this->cancelFlag = cancelFlag;}
    bool getRunningStatus() {return this->isAppRunning;}

    int handshakeWithMLApp(ML_CMD_HANDSHAKE_REQUEST request, ML_CMD_HANDSHAKE_RESPONSE &response);
    int analyzeSerialAndModel(ML_CMD_ANALYZE_REQUEST request, ML_CMD_ANALYZE_RESPONSE &response);

    int analyzeDentAndModel(QVariantMap mapRequest);
signals:
    void signalSendDataToMLApp(const ML_APP_COMMAND &cmd);

public slots:
    void slotReceiveDataFromMLApp(const ML_APP_COMMAND &cmd);
    void slotUpdateRunningStatus(bool isAppRunning) {this->isAppRunning = isAppRunning;}

private:
    static MLAppManager *INSTANCE;
    bool exitFlag;
    bool cancelFlag;
    qulonglong countCmd;
    QMutex lockListCmd;
    QList<ML_APP_COMMAND> listCmd;
    bool isAppRunning;
    QString moduleLogFile;

    MLAppManager();
    void run();
    QString createUniqueID();
    void appendCommand(ML_APP_COMMAND &subAppCmd);
    int waitCommandResponse(ML_APP_COMMAND &subAppCmd);
    int executeCommand(ML_APP_COMMAND &subAppCmd);
    void clearCommandInQueue();
};

#endif // MLAPPMANAGER_H
