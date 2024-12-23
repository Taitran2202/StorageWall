#ifndef MAINCONTROLLER_H
#define MAINCONTROLLER_H

#include <QObject>
#include <QThread>
#include "network/servermanager.h"


enum ID_STATION
{
    ID_STATION_ONE = 0,
    ID_STATION_TWO,
    ID_STATION_THREE
};


class MainController : public QThread
{
    Q_OBJECT
public:
    explicit MainController(QObject *parent = nullptr);

    int initialize();
signals:

public slots:


private:
    ServerManager *serverManager;
    QString moduleLogFile;

    void run();
};

#endif // MAINCONTROLLER_H
