#include <QCoreApplication>
#include "modules/maincontroller.h"
#include "globalvariable.h"
#include "logger/qlogger.h"
using namespace QLogger;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QLoggerManager *manager = QLoggerManager::getInstance();
    manager->addDestination("information.log", QStringList()<<g_app_name, QLogger::InfoLevel);

    MainController *mainController = new MainController();
    mainController->initialize();
    mainController->start();

    return a.exec();
}
