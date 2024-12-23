#ifndef JSONDATA_H
#define JSONDATA_H

#include <QString>
#include <QByteArray>
#include <QVariantMap>

class JsonData
{
public:
    JsonData();
    ~JsonData();
    bool result;
    QByteArray encodeJsonString(QVariantMap encodeMap);
    QVariantMap decodeJsonString(QByteArray jsonString);
};

#endif // JSONDATA_H
