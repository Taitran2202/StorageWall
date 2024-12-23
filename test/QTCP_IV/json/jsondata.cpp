#include "jsondata.h"
#include "json/parser.h"
#include "json/serializer.h"


JsonData::JsonData()
{
    this->result = false;
}

JsonData::~JsonData()
{

}

QByteArray JsonData::encodeJsonString(QVariantMap encodeMap)
{
    QJson::Serializer serializer;
    QByteArray json = serializer.serialize(encodeMap, &this->result);
    return json;
}

QVariantMap JsonData::decodeJsonString(QByteArray jsonString)
{
    QJson::Parser parser;
    return parser.parse (jsonString, &this->result).toMap();
}


