#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <iostream>
#include <memory>
#include "videostreamitem.h"
#include "videostreamer.h"
#include "systemstats.h"
#include "debug.h"
#include <QThread>

#ifdef USE_DXRT
#include "yolo.h"
#include <dxrt/exception/exception.h>
#endif

// Definition of the global verbose flag (declared extern in debug.h).
bool g_verbose = false;

class PeopleCounter : public QObject {
    Q_OBJECT
    Q_PROPERTY(int peopleSeen READ peopleSeen NOTIFY peopleSeenChanged)
public:
    explicit PeopleCounter(QObject* parent = nullptr) : QObject(parent) {}
    int peopleSeen() const { return m_count; }
public slots:
    void onPeopleSeen(int n) {
        if (n > m_count) {
            m_count = n;
            emit peopleSeenChanged();
        }
    }
signals:
    void peopleSeenChanged();
private:
    int m_count = 0;
};

#include "main.moc"

int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
    QGuiApplication app(argc, argv);

    // Register the custom QML type
    qmlRegisterType<VideoStreamItem>("com.deepx.app", 1, 0, "VideoStreamItem");

    SystemStats   systemStats;
    PeopleCounter peopleCounter;

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("systemStats",    &systemStats);
    engine.rootContext()->setContextProperty("peopleCounter",  &peopleCounter);

    const QUrl url(QStringLiteral("qrc:/qml/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    // Argument Parsing
    QStringList args = app.arguments();
    std::string modelPath;
    std::string labelsPath;
    std::string trackerModelPath;
    std::vector<std::string> pipelines;
    float scoreThreshold = 0.25f;
    float nmsThreshold = 0.45f;

    for (int i = 1; i < args.size(); ++i) {
        QString arg = args[i];
        if (arg == "-v") {
            g_verbose = true;
            std::cerr << "[DBG] Verbose mode enabled\n";
        } else if (arg == "--labels" && i + 1 < args.size()) {
            labelsPath = args[++i].toStdString();
            DLOG("Parsed --labels: " << labelsPath);
        } else if (arg == "--tracker-model" && i + 1 < args.size()) {
            trackerModelPath = args[++i].toStdString();
            DLOG("Parsed --tracker-model: " << trackerModelPath);
        } else if (arg == "--score-threshold" && i + 1 < args.size()) {
            scoreThreshold = args[++i].toFloat();
            DLOG("Parsed --score-threshold: " << scoreThreshold);
        } else if (arg == "--nms-threshold" && i + 1 < args.size()) {
            nmsThreshold = args[++i].toFloat();
            DLOG("Parsed --nms-threshold: " << nmsThreshold);
        } else if (modelPath.empty()) {
            modelPath = arg.toStdString();
            DLOG("Parsed model path: " << modelPath);
        } else {
            pipelines.push_back(arg.toStdString());
            DLOG("Parsed pipeline: " << pipelines.back());
        }
    }

    DLOG("Argument summary:"
         << " model=" << (modelPath.empty() ? "<none>" : modelPath)
         << " labels=" << (labelsPath.empty() ? "<none>" : labelsPath)
         << " tracker=" << (trackerModelPath.empty() ? "<none>" : trackerModelPath)
         << " pipelines=" << pipelines.size()
         << " score_threshold=" << scoreThreshold
         << " nms_threshold=" << nmsThreshold);

#ifdef USE_DXRT
    if (modelPath.empty() || labelsPath.empty() || pipelines.empty()) {
        std::cerr << "Usage: " << args[0].toStdString()
                  << " <model_path> --labels <labels.json> <pipeline1> [pipeline2]"
                  << " [--tracker-model <vitTracker.onnx>]"
                  << " [--score-threshold N] [--nms-threshold N] [-v]" << std::endl;
        return -1;
    }
#else
    if (modelPath.empty()) modelPath = "dummy_model";
#endif

    if (pipelines.empty()) {
        pipelines.push_back("dummy_pipeline");
    }

    YoloParam param;
#ifdef USE_DXRT
    // Load labels from JSON
    std::vector<std::string> labels;
    {
        DLOG("Loading labels from: " << labelsPath);
        QFile file(QString::fromStdString(labelsPath));
        if (!file.open(QIODevice::ReadOnly)) {
            std::cerr << "Failed to open labels file: " << labelsPath << std::endl;
            return -1;
        }
        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        if (doc.isNull() || !doc.isObject()) {
            std::cerr << "Invalid JSON in labels file: " << labelsPath << std::endl;
            return -1;
        }
        QJsonArray arr = doc.object()["labels"].toArray();
        if (arr.isEmpty()) {
            std::cerr << "No 'labels' array found in: " << labelsPath << std::endl;
            return -1;
        }
        for (const QJsonValue& v : arr) {
            labels.push_back(v.toString().toStdString());
        }
        DLOG("Loaded " << labels.size() << " labels");
    }
    param = createYolo26xParam(labels, scoreThreshold, nmsThreshold);
    DLOG("YoloParam created: width=" << param.width << " height=" << param.height
         << " numClasses=" << param.numClasses
         << " scoreThreshold=" << param.scoreThreshold
         << " iouThreshold=" << param.iouThreshold);
#else
    param.width = 640;
    param.height = 640;
    DLOG("Using dummy YoloParam: width=" << param.width << " height=" << param.height);
#endif

    // Connect to QML Items
    QObject* root = engine.rootObjects().first();
    DLOG("Root QML object: " << (root ? root->metaObject()->className() : "<null>"));

    // Manage threads so they don't go out of scope
    std::vector<QThread*> threads;
    std::vector<VideoStreamer*> streamers;

#ifdef USE_DXRT
    dxrt::InferenceOption op_od;
    op_od.devices.push_back(0);
    DLOG("Creating InferenceEngine: model=" << modelPath << " device=0");
    std::shared_ptr<dxrt::InferenceEngine> ie;
    try {
        ie = std::make_shared<dxrt::InferenceEngine>(modelPath, op_od);
        DLOG("InferenceEngine created successfully");
    } catch (const dxrt::Exception& e) {
        std::cerr << "Failed to initialize InferenceEngine: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize InferenceEngine: " << e.what() << std::endl;
        return -1;
    }

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

            DLOG("Callback: running PostProc on buffer index " << index
                 << " (process_count=" << arguments->od_process_count << ")");
            auto od_result = arguments->yolo->PostProc(outputs);
            DLOG("Callback: PostProc returned " << od_result.size() << " detections");
            arguments->od_results[index] = od_result;
            arguments->od_process_count = arguments->od_process_count + 1;
            arguments->frame_idx = arguments->frame_idx + 1;
        }
        return 0;
    };
    ie->RegisterCallback(od_postProcCallBack);
    DLOG("Post-processing callback registered");
#endif

    for (size_t i = 0; i < pipelines.size(); ++i) {
        if (i > 1) break; // Only 2 streams supported in QML currently

        QString objectName = QString("stream%1").arg(i);
        DLOG("Looking for QML item: " << objectName.toStdString());
        QObject* item = root->findChild<QObject*>(objectName);

        if (item) {
            DLOG("Found QML item: " << objectName.toStdString());
            // Make visible
            item->setProperty("visible", true);
            VideoStreamItem* videoItem = qobject_cast<VideoStreamItem*>(item);

            if (videoItem) {
                DLOG("Creating VideoStreamer for stream " << i << ": pipeline=" << pipelines[i]);
                QThread* thread = new QThread;
#ifdef USE_DXRT
                VideoStreamer* streamer = new VideoStreamer(i, ie, modelPath, param, pipelines[i], trackerModelPath);
#else
                VideoStreamer* streamer = new VideoStreamer(i, modelPath, param, pipelines[i], trackerModelPath);
#endif
                streamer->moveToThread(thread);

                QObject::connect(thread, &QThread::started, streamer, &VideoStreamer::process);
                QObject::connect(streamer, &VideoStreamer::imageReady, videoItem, &VideoStreamItem::updateImage);
                QObject::connect(streamer, &VideoStreamer::peopleSeen, &peopleCounter, &PeopleCounter::onPeopleSeen);
                QObject::connect(streamer, &VideoStreamer::finished, thread, &QThread::quit);
                QObject::connect(streamer, &VideoStreamer::finished, streamer, &VideoStreamer::deleteLater);
                QObject::connect(thread, &QThread::finished, thread, &QThread::deleteLater);

                threads.push_back(thread);
                streamers.push_back(streamer);
                thread->start();
                DLOG("Thread started for stream " << i);
            } else {
                DLOG("QML item " << objectName.toStdString() << " is not a VideoStreamItem");
            }
        } else {
            std::cerr << "Could not find QML item: " << objectName.toStdString() << std::endl;
        }
    }

    int ret = app.exec();

    // Cleanup
    DLOG("Stopping " << streamers.size() << " streamer(s)");
    for(auto s : streamers) s->stop();
    DLOG("Waiting for " << threads.size() << " thread(s) to finish");
    for(auto t : threads) {
        t->quit();
        t->wait();
    }
    DLOG("Cleanup complete, exiting with code " << ret);

    return ret;
}
