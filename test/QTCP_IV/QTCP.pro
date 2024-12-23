QT -= gui
QT += network

CONFIG += c++11 console
CONFIG -= app_bundle
TARGET = detect_phone_slot

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    network/clientsocket.cpp \
    network/mlappmanager.cpp \
    network/servermanager.cpp \
    logger/qlogger.cpp \
    json/parser.cpp \
    json/json_parser.cc \
    json/qobjecthelper.cpp \
    json/json_scanner.cc \
    json/parserrunnable.cpp \
    json/json_scanner.cpp \
    json/serializer.cpp \
    json/jsondata.cpp \
    json/serializerrunnable.cpp \
    globalvariable.cpp \
    modules/maincontroller.cpp

HEADERS += \
    defines.h \
    network/clientsocket.h \
    network/mlappmanager.h \
    network/servermanager.h \
    logger/qlogger.h \
    json/qobjecthelper.h \
    json/parserrunnable.h \
    json/json_scanner.h \
    json/location.hh \
    json/stack.hh \
    json/position.hh \
    json/parser_p.h \
    json/serializer.h \
    json/jsondata.h \
    json/qjson_debug.h \
    json/serializerrunnable.h \
    json/FlexLexer.h \
    json/json_parser.hh \
    json/parser.h \
    json/qjson_export.h \
    globalvariable.h \
    modules/maincontroller.h

#unix {
#    target.path = /home/greystone/ivaluate/apps
#    INSTALLS += target
#}

#LIBS        +=  -L/usr/local/lib -lmcrypt

#DESTDIR = $$PWD/../../../build_install/apps
