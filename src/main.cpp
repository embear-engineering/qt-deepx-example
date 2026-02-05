#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <iostream>
#include <memory>
#include "videostreamitem.h"
#include "videostreamer.h"
#include <QThread>

#ifdef USE_DXRT
#include "yolo.h"
// Extern declaration, definition is in yolo_cfg.cpp to ensure correct initialization order
extern std::vector<YoloParam> yoloParams;
#endif

int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
    QGuiApplication app(argc, argv);

    // Register the custom QML type
    qmlRegisterType<VideoStreamItem>("com.deepx.app", 1, 0, "VideoStreamItem");

    QQmlApplicationEngine engine;
    const QUrl url(QStringLiteral("qrc:/qml/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    // Argument Parsing
    QStringList args = app.arguments();
#ifdef USE_DXRT
    if (args.size() < 4) {
        std::cerr << "Usage: " << args[0].toStdString() << " <model_path> <profile_index> <pipeline1> [pipeline2]" << std::endl;
        std::cerr << "Profile Index Mapping:" << std::endl;
        std::cerr << "0: yolov5s_320\n1: yolov5s_512\n2: yolov5s_640\n3: yolov7_512\n4: yolov7_640\n5: yolov8_640\n6: yolox_s_512\n7: yolov5s_face_640\n8: yolov3_512\n9: yolov4_416\n10: yolov9_640" << std::endl;
        return -1;
    }
#else
    // If no DXRT, we might not need model/profile, but let's keep signature or allow dummy
    if (args.size() < 2) {
        // Allow running with just a dummy pipeline arg or defaults if testing UI
        // But to be consistent:
        // ./app dummy_model 0 pipeline1
    }
#endif

    std::string modelPath = (args.size() > 1) ? args[1].toStdString() : "dummy_model";
    int profileIndex = (args.size() > 2) ? args[2].toInt() : 0;
    std::vector<std::string> pipelines;
    if (args.size() > 3) pipelines.push_back(args[3].toStdString());
    if (args.size() > 4) pipelines.push_back(args[4].toStdString());

    // If testing UI without args, add a dummy pipeline
    if (pipelines.empty()) {
        pipelines.push_back("dummy_pipeline");
    }

    YoloParam param;
#ifdef USE_DXRT
    if (profileIndex >= 0 && profileIndex < yoloParams.size()) {
        param = yoloParams[profileIndex];
    } else {
        std::cerr << "Invalid profile index" << std::endl;
        // fallback to 0
        if(!yoloParams.empty()) param = yoloParams[0];
    }
#else
    param.width = 640;
    param.height = 480;
#endif

    // Connect to QML Items
    QObject* root = engine.rootObjects().first();

    // Manage threads so they don't go out of scope
    std::vector<QThread*> threads;
    std::vector<VideoStreamer*> streamers;

#ifdef USE_DXRT
    dxrt::InferenceOption op_od;
    op_od.devices.push_back(0);
    auto ie = std::make_shared<dxrt::InferenceEngine>(modelPath, op_od);

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> od_postProcCallBack =
                [](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
    {
        auto arguments = (OdEstimationArgs*)arg;
        if (!arguments || !arguments->yolo) return -1;

        {
            std::unique_lock<std::mutex> lk(arguments->lk);
            int bufferSize = (int)arguments->od_results.size();
            if (bufferSize == 0) return 0;

            int index = arguments->od_process_count;
            if(index >= bufferSize) {
                index = index % bufferSize;
            } else if (index < 0) {
                index = 0;
            }

            auto od_result = arguments->yolo->PostProc(outputs);
            arguments->od_results[index] = od_result;
            arguments->od_process_count = arguments->od_process_count + 1;
            arguments->frame_idx = arguments->frame_idx + 1;
        }
        return 0;
    };
    ie->RegisterCallback(od_postProcCallBack);
#endif

    for (size_t i = 0; i < pipelines.size(); ++i) {
        if (i > 1) break; // Only 2 streams supported in QML currently

        QString objectName = QString("stream%1").arg(i);
        QObject* item = root->findChild<QObject*>(objectName);

        if (item) {
            // Make visible
            item->setProperty("visible", true);
            VideoStreamItem* videoItem = qobject_cast<VideoStreamItem*>(item);

            if (videoItem) {
                QThread* thread = new QThread;
#ifdef USE_DXRT
                VideoStreamer* streamer = new VideoStreamer(i, ie, modelPath, param, pipelines[i]);
#else
                VideoStreamer* streamer = new VideoStreamer(i, modelPath, param, pipelines[i]);
#endif
                streamer->moveToThread(thread);

                QObject::connect(thread, &QThread::started, streamer, &VideoStreamer::process);
                // Direct connection might be unsafe across threads for complex types,
                // but QImage is implicitly shared and registered. QueuedConnection is default for cross-thread.
                QObject::connect(streamer, &VideoStreamer::imageReady, videoItem, &VideoStreamItem::updateImage);
                QObject::connect(streamer, &VideoStreamer::finished, thread, &QThread::quit);
                QObject::connect(streamer, &VideoStreamer::finished, streamer, &VideoStreamer::deleteLater);
                QObject::connect(thread, &QThread::finished, thread, &QThread::deleteLater);

                threads.push_back(thread);
                streamers.push_back(streamer);
                thread->start();
            }
        } else {
            std::cerr << "Could not find QML item: " << objectName.toStdString() << std::endl;
        }
    }

    int ret = app.exec();

    // Cleanup
    for(auto s : streamers) s->stop();
    for(auto t : threads) {
        t->quit();
        t->wait();
    }

    return ret;
}
