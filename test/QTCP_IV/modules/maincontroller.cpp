#include "maincontroller.h"
#include "network/servermanager.h"
#include "network/mlappmanager.h"
#include "globalvariable.h"
#include <unistd.h>

MainController::MainController(QObject *parent) : QThread(parent)
{
    moduleLogFile = g_app_name;
    this->serverManager = new ServerManager(this);
    MLAppManager *mlAppManager = MLAppManager::getInstance();
    mlAppManager->initialize();
    mlAppManager->start();
}

int MainController::initialize()
{
    int res = this->serverManager->initialize();
    return res;
}

void MainController::run()
{
    QStringList phoneId_list = { "09102023104059", "11102023052503", "11102023052503"};
    int i = 0;

    while(true) {
        MLAppManager *mlAppManager = MLAppManager::getInstance();
        if(mlAppManager->getRunningStatus()) {
            qDebug()<<"ML app is running";

//            ML_CMD_HANDSHAKE_REQUEST request;
//            request.masterPid = getpid();
//            request.pathDebug = "debug";
//            request.pathRoot = "ggnonphone";
//            request.pathLog = "log";
//            ML_CMD_HANDSHAKE_RESPONSE response;
//            mlAppManager->handshakeWithMLApp(request, response);

            if(0){
                ML_CMD_ANALYZE_REQUEST request;
                request.fixtureType = 0;
                request.moduleIndex = 0;
                //            request.listImgPath.append("/mnt/Data/debug/samples/test1.jpg");
                request.listImgPath.append("/home/greystone/Tien_Truong/Projects/ML_Gitlab/detect-screen-id/modules/image/image_detect_device_5_side_2_1_square.jpg");
                //            request.listImgPath.append("/home/greystone/Tien_Truong/Projects/ML_Gitlab/detect-screen-id/modules/image/image_detect_device_6_side_2_1_square.jpg");
                //            request.listImgPath.append("/home/greystone/Tien_Truong/Projects/ML_Gitlab/detect-screen-id/modules/image/image_detect_device_8_side_2_1_square_v1.jpg");
                //            request.listImgPath.append("/home/greystone/Tien_Truong/Projects/ML_Gitlab/detect-screen-id/modules/image/image_detect_device_10_side_2_1_square.jpg");
                request.pathDebug = "path_root/debug";
                ML_CMD_ANALYZE_RESPONSE response;
                mlAppManager->analyzeSerialAndModel(request, response);
                qDebug()<<QString("serialNumber = %1; modelNumber = %2").arg(response.serialNumber).arg(response.modelNumber);

            }

            if(1){
                QVariantMap mapRequest;

                //Define function communicate with ml-subapp to calibrate
                // mapRequest["requestType"] = "calibrate";
                // mapRequest["positionToCalib"] = "buffer";
                // mapRequest["pathToImg1"] = "";
                // mapRequest["pathToImg2"] = "/home/greystone/LongVu/Projects/StorageWall/Sources/Gitlab/dataset/model_detect_phone_in_kangaroo/org/images/img_44.png";
                // mapRequest["pathToDebug"] = "/home/greystone/StorageWall";
                // mapRequest["pathToList"] = "";
                // mapRequest["pathToResult"] = "";
                // mapRequest["transaction_id"] = "250324";


                // Define function communicate with ml-subapp to detect phone slot
                mapRequest["positionToDetect"] = "left";
                mapRequest["requestType"] = "detect_phone_slot";
                mapRequest["pathToImg"] = "/home/greystone/StorageWall/image_debug/W01W01LSR35C24_01.png";
                mapRequest["pathToDebug"] = "/home/greystone/StorageWall";
                mapRequest["pathToList"] = "";
                mapRequest["pathToResult"] = "";
                mapRequest["transaction_id"] = "250324";


                // Define function communicate with ml-subapp to scan barcode
                // mapRequest["requestType"] = "scan_barcode";
                // mapRequest["pathToImg"] = "";
                // mapRequest["pathToDebug"] = "/home/greystone/LongVu/Projects/StorageWall/debug";
                // mapRequest["pathToList"] = "/home/greystone/LongVu/Projects/StorageWall/debug/list_scan_barcode.txt";
                // mapRequest["pathToResult"] = "/home/greystone/LongVu/Projects/StorageWall/debug/output.json";
                // mapRequest["transaction_id"] = "250324";

                mlAppManager->analyzeDentAndModel(mapRequest);

            }
            i = i+1;

        } else {
            qDebug()<<"ML app is not running";
        }
        sleep(5);
    }
}
