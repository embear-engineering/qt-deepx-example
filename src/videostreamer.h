#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QWaitCondition>
#include <mutex>
#include <vector>
#include <memory>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef USE_DXRT
#include <dxrt/dxrt_api.h>
#include "yolo.h"
#ifdef USE_OPENCV
#include "image.h"
#include "display.h"
#endif
#else
// Mock YoloParam if DXRT is missing
struct YoloParam {
    int height;
    int width;
    // ... add other fields if accessed directly
};
#endif

struct OdEstimationArgs {
#ifdef USE_DXRT
    std::vector<std::vector<BoundingBox>> od_results;
    Yolo* yolo = nullptr;
#endif
    std::vector<std::vector<int64_t>> od_output_shape;
    std::mutex lk;
    int od_process_count = 0;
    int frame_idx = 0;
};

class VideoStreamer : public QObject
{
    Q_OBJECT
public:
#ifdef USE_DXRT
    explicit VideoStreamer(int streamId, std::shared_ptr<dxrt::InferenceEngine> ie, const std::string& modelPath, const YoloParam& yoloParam, const std::string& pipeline, QObject *parent = nullptr);
#else
    explicit VideoStreamer(int streamId, const std::string& modelPath, const YoloParam& yoloParam, const std::string& pipeline, QObject *parent = nullptr);
#endif
    ~VideoStreamer();

    void stop();

public slots:
    void process();

signals:
    void imageReady(QImage image); // Removed streamId as signal, connected directly
    void finished();
    void error(QString message);

private:
    int m_streamId;
    std::string m_modelPath;
    YoloParam m_yoloParam;
    std::string m_pipeline;
    bool m_stop;
    int m_displayed_count = 0;

#ifdef USE_DXRT
    std::shared_ptr<dxrt::InferenceEngine> m_ie;
    Yolo* m_yolo;
#endif

    OdEstimationArgs m_odArgs;
    
    // Buffers
    static const int FRAME_BUFFERS = 10;
#ifdef USE_OPENCV
    cv::Mat m_frames[FRAME_BUFFERS];
    std::vector<cv::Mat> m_odInputs;
#endif
    std::vector<std::vector<uint8_t>> m_odOutputs;
};

#endif // VIDEOSTREAMER_H