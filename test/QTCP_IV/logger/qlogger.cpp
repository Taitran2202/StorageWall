#include <QDebug>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
#include "qlogger.h"
#include "defines.h"
#include "globalvariable.h"

QString serviceCode;

/****************************************************************************************
** QLogger is a library to register and print logs into a file.
** Copyright (C) 2013 Francesc Martinez <es.linkedin.com/in/cescmm/en>
**
** This library is free software; you can redistribute it and/or
** modify it under the terms of the GNU Lesser General Public
** License as published by the Free Software Foundation; either
** version 2.1 of the License, or (at your option) any later version.
**
** This library is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
** Lesser General Public License for more details.
**
** You should have received a copy of the GNU Lesser General Public
** License along with this library; if not, write to the Free Software
** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
***************************************************************************************/

namespace QLogger
{
void QLog_Trace(const QString &module, const QString &message)
{
    QLog_(module, TraceLevel, message);
}

void QLog_Debug(const QString &module, const QString &message)
{
    QLog_(module, DebugLevel, message);
}

void QLog_Info(const QString &module, const QString &message)
{
    QLog_(module, InfoLevel, message);
}

void QLog_Warning(const QString &module, const QString &message)
{
    QLog_(module, WarnLevel, message);
}

void QLog_Error(const QString &module, const QString &message)
{
    QLog_(module, ErrorLevel, message);
}

void QLog_Fatal(const QString &module, const QString &message)
{
    QLog_(module, FatalLevel, message);
}

quint64 QLog_GetSizeOfDir(QString path)
{
    QDir dir(path);

    //Get size of directory
    QFileInfoList fileInfoList;
    quint64 sizeOfLogDir = 0;
    fileInfoList = dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot, QDir::Time);
    int totalOfFiles = fileInfoList.count();
    qDebug() <<"QLog_Cleanup > Total of files in directory: "<<path<<" is "<<totalOfFiles;
    for(int i = 0; i < totalOfFiles; i++)
    {
        QFileInfo fileInfo = fileInfoList.at(i);
        //qDebug() <<"QLog_Cleanup > File->"<<fileInfo.baseName();
        if(fileInfo.size())
        {
            sizeOfLogDir += fileInfo.size();
        }
    }

    if(sizeOfLogDir >= 1024 * 1024 * 1024)
    {
        fprintf(stdout,"QLog_Cleanup > Size of %s directory is %f (BBs)\n", dir.path().toLatin1().data(), (double)sizeOfLogDir / (1024 * 1024 * 1024));
    }
    else if(sizeOfLogDir >= 1024 * 1024)
    {
        fprintf(stdout,"QLog_Cleanup > Size of %s directory is %f (MBs)\n", dir.path().toLatin1().data(), (double)sizeOfLogDir / (1024 * 1024));
    }
    else if(sizeOfLogDir >= 1024)
    {
        fprintf(stdout,"QLog_Cleanup > Size of %s directory is %f (KBs)\n", dir.path().toLatin1().data(), (double)sizeOfLogDir / (1024));
    }
    else
    {
        fprintf(stdout,"QLog_Cleanup > Size of %s directory = %f (Bytes)\n", dir.path().toLatin1().data(), (double)sizeOfLogDir);
    }
    return sizeOfLogDir;
}

void QLog_RemoveTransactionLog(QString dirPath)
{
    QDir dir(dirPath);
    QFileInfoList fileInfoList = dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot, QDir::Time);
    for(int i = 0; i < fileInfoList.count(); i++)
    {
        QFileInfo fileInfo = fileInfoList.at(i);
        if(fileInfo.fileName() != "sw_manager.log")
        {
            qDebug() <<"QLog_RemoveTransactionLog > File->"<<fileInfo.baseName();
            dir.remove(fileInfo.fileName());
        }
    }
}

void QLog_Backup(QString inputDirPath, QString outputDirPath)
{
    QLog_Info(MODULE_QLOGGER, "Backup Logfile            [BEGIN]");
    QString command = "";

    QDir dir(inputDirPath);
    QString tarFilePath;
    QLog_Info(MODULE_QLOGGER, QString("Total of files for backup: %1").arg(dir.entryList().count()));
    if(serviceCode.right(1) == "_")
    {
        QDateTime currentTime = QDateTime::currentDateTime();
        tarFilePath = outputDirPath + "/" + dir.dirName() + QString("_") + serviceCode;
        tarFilePath += QString("%1__%2.tar.gz").arg(currentTime.date().toString("dd_MM_yy")).arg(currentTime.time().toString("hh_mm_ss"));
    }
    else
    {
        tarFilePath = outputDirPath + "/" + dir.dirName() + QString("_") + serviceCode + QString(".tar.gz");
    }

    command.clear();
    command = "tar -czf " + tarFilePath + " -C /home logs";
    QLog_Info(MODULE_QLOGGER, "Execute comand: " + command);
    system(command.toUtf8().data());

    command = "sync";
    QLog_Info(MODULE_QLOGGER, "Execute comand: " + command);
    system(command.toUtf8().data());

    QLog_RemoveTransactionLog(inputDirPath);

    QLog_Info(MODULE_QLOGGER, "Backup Logfile            [OK]");
}

/**
 * @brief QLog_Cleanup
 */
void QLog_Cleanup(QString dirPath)
{
    QDir dir(dirPath);

    if(dir.entryList().count() > QLOGGER_MAX_LOG_DIR_NUM_OF_FILE)
    {
        QStringList fileList = dir.entryList(QDir::AllEntries | QDir::NoDotAndDotDot, QDir::Time);
        for(int i = QLOGGER_MAX_LOG_DIR_NUM_OF_FILE; i < fileList.count(); i++)
        {
            if((fileList.at(i).compare(".") == 0) ||
                    (fileList.at(i).compare("..") == 0) ||
                    (fileList.at(i).compare("error") == 0) ||
                    (fileList.at(i) == "information.log"))
            {
                continue;
            }

            qDebug() <<"QLog_Cleanup > Remove File->"<<fileList.at(i);
            dir.remove(fileList.at(i));
        }
    }
    //Kiem tra kich thuoc thu muc
    while(1)
    {
        if(QLog_GetSizeOfDir(dirPath) < QLOGGER_MAX_LOG_DIR_SIZE)
        {
            qDebug() <<"QLog_Cleanup > nothing on directory: "<<dirPath;
            break;
        }

        QFileInfoList fileInfoList = dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot, QDir::Time);
        qint64 maxFileZise = 0;
        QString delFile;
        for(int i = 0; i < fileInfoList.count(); i++)
        {
            QFileInfo fileInfo = fileInfoList.at(i);
            //qDebug() <<"QLog_Cleanup > File->"<<fileInfo.baseName();
            if((fileInfo.size() > maxFileZise) && (fileInfo.fileName() != "information.log"))
            {
                maxFileZise = fileInfo.size();
                delFile = fileInfo.fileName();
            }
        }

        qDebug() <<"QLog_Cleanup > Remove file "<<delFile<<" in directory: "<<dirPath;
        dir.remove(delFile);
        usleep(1000);
    }
}

/**
 * @brief QLog_
 * @param module
 * @param level
 * @param message
 */
void QLog_(const QString &module, LogLevel level, const QString &message)
{
    if(level < g_log_debug_level) {
        return;
    }
    QLoggerManager *manager = QLoggerManager::getInstance();

    QMutexLocker(&manager->mutexDestination);

    QLoggerWriter *logWriter = manager->getLogWriter(module);
    if(logWriter == nullptr){
        fprintf(stdout, "[QLoggerWriter] %s is NULL \n", module.toLatin1().data());
        return;
    }
    logWriter->setLevel(level);
    if (logWriter and logWriter->getLevel() >= level)
    {
        logWriter->write(module,message);
    }
}

//QLoggerManager
QLoggerManager * QLoggerManager::INSTANCE = NULL;

QLoggerManager::QLoggerManager() : QThread(), mutex(QMutex::Recursive)
{
    start();
}

bool QLoggerManager::removeDestination(QString module)
{
    qDebug()<<"Remove Destination: "<<module;
    mutexDestination.lock();
    bool result = false;
    if(moduleDest.count(module) != 0)
    {
        qDebug()<<"Remove Destination: "<<module<<" Exist";
        QLoggerWriter *log = moduleDest.take(module);

        if(log != NULL){
            qDebug()<<"Remove Destination: "<<module<<" Exist => Remove Passed";
            delete log;
            log = NULL;
            result = true;
        }else{
            qDebug()<<"Remove Destination: "<<module<<" Exist => NULL";
        }

//        qDebug()<<"Get Destination Passed";

//        if(moduleDest.remove(module) != 0)
//        {
//            qDebug()<<"Remove Destination: "<<module<<" Exist => Remove Passed";

//        }
//        else
//        {
//            qDebug()<<"Remove Destination: "<<module<<" Exist => Remove Failed";
//        }
    }
    mutexDestination.unlock();
    return result;
}

bool QLoggerManager::checkDestination(QString module)
{
    mutexDestination.lock();
    bool result = false;
    if(moduleDest.count(module) != 0)
    {
        result = true;
    }
    mutexDestination.unlock();
    return result;
}

void QLoggerManager::getListDestination(QList<QString> *listDestination)
{
    mutexDestination.lock();
    QMap<QString,QLoggerWriter*>::iterator it;
    for (it = moduleDest.begin(); it != moduleDest.end(); ++it) {
        listDestination->append(it.key());
    }
    mutexDestination.unlock();
}

int QLoggerManager::numListDestination()
{
    int num = 0;
    mutexDestination.lock();
    QMap<QString,QLoggerWriter*>::iterator it;
    for (it = moduleDest.begin(); it != moduleDest.end(); ++it) {
        num++;
    }
    mutexDestination.unlock();
    return num;
}

void QLoggerManager::clearDestination()
{
    mutexDestination.lock();
    QList<QString> listDestination;
    listDestination.clear();
    QMap<QString,QLoggerWriter*>::iterator it;
    for (it = moduleDest.begin(); it != moduleDest.end(); ++it) {
        listDestination.append(it.key());
    }

    for(int i = 0; i < listDestination.count(); i++)
    {
        if(listDestination.contains("phone_id") == true)
        {
//            moduleDest.remove(listDestination.at(i));
            delete moduleDest.take(listDestination.at(i));
        }
    }
    mutexDestination.unlock();
}

/**
 * @brief QLoggerManager::getInstance
 * @return
 */
QLoggerManager * QLoggerManager::getInstance()
{
    if (!INSTANCE)
        INSTANCE = new QLoggerManager();

    return INSTANCE;
}

/**
 * @brief QLoggerManager::levelToText
 * @param level
 * @return
 */
QString QLoggerManager::levelToText(const LogLevel &level)
{
    switch (level)
    {
    case TraceLevel: return "Trace";
    case DebugLevel: return "Debug";
    case InfoLevel: return "Info";
    case WarnLevel: return "Warning";
    case ErrorLevel: return "Error";
    case FatalLevel: return "Fatal";
    default: return QString();
    }
}

/**
 * @brief QLoggerManager::addDestination
 * @param fileDest
 * @param modules
 * @param level
 * @return
 */
bool QLoggerManager::addDestination(const QString &fileDest, const QStringList &modules, LogLevel level)
{
    bool res = false;
    qDebug()<<"Add log module: "<<fileDest<<" Name: ";
    mutexDestination.lock();
    QLoggerWriter *log;
    foreach (QString module, modules)
    {
        log = new QLoggerWriter(fileDest,level);
        if(moduleDest.count(module) == 0)
        {
            moduleDest.insert(module, log);
            qDebug()<<"Add log module: "<<fileDest<<" =>> Passed";
            res = true;
        }
        else
        {
            qDebug()<<"Add log module: "<<fileDest<<" =>> Failed";
            if(log != NULL)
            {
                delete log;
                log = NULL;
            }
        }
    }
    mutexDestination.unlock();
    return res;
}

/**
 * @brief QLoggerManager::closeLogger
 */
void QLoggerManager::closeLogger()
{
    exit(0);
    deleteLater();
}

/**
 * @brief QLoggerWriter::QLoggerWriter
 * @param fileDestination
 * @param level
 */
QLoggerWriter::QLoggerWriter(const QString &fileDestination, LogLevel level)
{
    m_fileDestination = fileDestination;
    m_level = level;
}

/**
 * @brief QLoggerWriter::write
 * @param module
 * @param message
 */
void QLoggerWriter::write(const QString &module, const QString &message)
{
    QString _fileName = m_fileDestination;

    int MAX_SIZE = 1024 * 1024*20;
    fprintf(stdout, "{%s} %s\n", module.toLatin1().data(), message.toLatin1().data());

    QFile file(_fileName);
    QString toRemove = _fileName.section('.',-1);
    QString fileNameAux = _fileName.left(_fileName.size() - toRemove.size()-1);
    bool renamed = false;
    QString newName = fileNameAux + "_%1__%2.log";

    //Renomenem l'arxiu si està ple
    if (file.size() >= MAX_SIZE)
    {
        //Creem un fixer nou
        QDateTime currentTime = QDateTime::currentDateTime();
        newName = newName.arg(currentTime.date().toString("dd_MM_yy")).arg(currentTime.time().toString("hh_mm_ss"));
        renamed = file.rename(_fileName, newName);

    }

    file.setFileName(_fileName);
    if (file.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append))
    {
        QTextStream out(&file);
        QString dtFormat = QDateTime::currentDateTime().toString("dd-MM-yyyy hh:mm:ss.zzz");

        if (renamed)
            out << QString("%1 - Previuous log %2\n").arg(dtFormat).arg(newName);

        QString logLevel = QLoggerManager::levelToText(m_level);
        QString text = QString("[%1] [%2] {%3} %4\n").arg(dtFormat).arg(logLevel).arg(module).arg(message);
        out << text;
        file.close();
    }
}
}
