#include "videostreamer.h"
#include "debug.h"
#include <QDebug>
#include <QPainter>
#include <QThread>
#include <thread>
#include <chrono>

#ifdef USE_DXRT
#ifdef USE_OPENCV
#include <utils/color_table.hpp>
#endif
#include <dxrt/exception/exception.h>
#endif

#ifdef USE_DXRT
VideoStreamer::VideoStreamer(int streamId, std::shared_ptr<dxrt::InferenceEngine> ie, const std::string& modelPath, const YoloParam& yoloParam, const std::string& pipeline, const std::string& trackerModelPath, QObject *parent)
    : QObject(parent), m_streamId(streamId), m_ie(ie), m_modelPath(modelPath), m_yoloParam(yoloParam), m_pipeline(pipeline), m_stop(false), m_trackerModelPath(trackerModelPath)
      , m_yolo(nullptr)
#else
VideoStreamer::VideoStreamer(int streamId, const std::string& modelPath, const YoloParam& yoloParam, const std::string& pipeline, const std::string& trackerModelPath, QObject *parent)
    : QObject(parent), m_streamId(streamId), m_modelPath(modelPath), m_yoloParam(yoloParam), m_pipeline(pipeline), m_stop(false), m_trackerModelPath(trackerModelPath)
#endif
{
    m_odOutputs.resize(FRAME_BUFFERS);
#ifdef USE_OPENCV
    m_odInputs.resize(FRAME_BUFFERS);
#endif
    DLOG("VideoStreamer[" << m_streamId << "] created: pipeline=" << m_pipeline
         << " model=" << m_modelPath);
}

VideoStreamer::~VideoStreamer()
{
    DLOG("VideoStreamer[" << m_streamId << "] destroyed");
    stop();
#ifdef USE_DXRT
    // m_ie is shared_ptr, no manual delete
    if (m_yolo) delete m_yolo;
#endif
}

void VideoStreamer::stop()
{
    DLOG("VideoStreamer[" << m_streamId << "] stop() called");
    m_stop = true;
}

void VideoStreamer::process()
{
    DLOG("VideoStreamer[" << m_streamId << "] process() started on thread "
         << QThread::currentThreadId());
    try {
#ifdef USE_DXRT
        if (!m_ie) {
             emit error("InferenceEngine not initialized");
             return;
        }
        DLOG("VideoStreamer[" << m_streamId << "] InferenceEngine is valid");

        m_yolo = new Yolo(m_yoloParam);
        DLOG("VideoStreamer[" << m_streamId << "] Yolo instance created");
        if(!m_yolo->LayerReorder(m_ie->GetOutputs())) {
            emit error("Layer reorder failed");
            return;
        }
        DLOG("VideoStreamer[" << m_streamId << "] LayerReorder succeeded");

        m_odArgs.yolo = m_yolo;

        // Setup buffers
        size_t outputSize = m_ie->GetOutputSize();
        DLOG("VideoStreamer[" << m_streamId << "] Allocating " << FRAME_BUFFERS
             << " frame buffers, inference output size=" << outputSize << " bytes");
        for(int i=0; i<FRAME_BUFFERS; i++) {
            m_odOutputs[i] = std::vector<uint8_t>(outputSize);
#ifdef USE_OPENCV
            m_odInputs[i] = cv::Mat(m_yoloParam.height, m_yoloParam.width, CV_8UC3);
#endif
        }

        // Setup Args
        std::vector<std::vector<int64_t>> output_shape;
        for(auto &o : m_ie->GetOutputs()) {
            output_shape.emplace_back(o.shape());
        }
        m_odArgs.od_output_shape = output_shape;
        m_odArgs.od_results = std::vector<std::vector<BoundingBox>>(FRAME_BUFFERS);
        DLOG("VideoStreamer[" << m_streamId << "] OD args initialised: "
             << output_shape.size() << " output tensor(s)");

#endif

#ifdef USE_OPENCV
        cv::VideoCapture cap;
        if (m_pipeline.find("!") != std::string::npos) {
             DLOG("VideoStreamer[" << m_streamId << "] Opening GStreamer pipeline: " << m_pipeline);
             cap.open(m_pipeline, cv::CAP_GSTREAMER);
        } else {
             DLOG("VideoStreamer[" << m_streamId << "] Opening video source: " << m_pipeline);
             cap.open(m_pipeline);
        }

        if(!cap.isOpened()) {
            emit error("Could not open pipeline: " + QString::fromStdString(m_pipeline));
            return;
        }
        DLOG("VideoStreamer[" << m_streamId << "] Pipeline opened successfully"
             << " (backend=" << cap.getBackendName().c_str() << ")");

#ifdef USE_DXRT
        if (!m_trackerModelPath.empty()) {
            m_personTracker = std::unique_ptr<PersonTracker>(
                new PersonTracker(m_trackerModelPath));
            DLOG("VideoStreamer[" << m_streamId << "] PersonTracker initialised: "
                 << m_trackerModelPath);
        } else {
            DLOG("VideoStreamer[" << m_streamId << "] No tracker model path – tracking disabled");
        }
#endif

        int index = 0;
#ifdef USE_DXRT
        auto objectColors = dxapp::common::color_table;
#endif

        while(!m_stop) {
            cv::Mat frame;
            cap >> frame;
            if(frame.empty()) {
                DLOG("VideoStreamer[" << m_streamId << "] Empty frame at index " << index << ", ending stream");
                break;
            }

            DLOG("VideoStreamer[" << m_streamId << "] Captured frame " << index
                 << " (" << frame.cols << "x" << frame.rows << ")");

            // Plain assignment is safe: OpenCV's GStreamer backend uses reference-counted
            // buffers, so m_frames[index] keeps the buffer alive without an extra copy.
            m_frames[index] = frame;

#ifdef USE_DXRT

#ifdef USE_OPENCV

            PreProc(frame, m_odInputs[index], true, true, 114);
            DLOG("VideoStreamer[" << m_streamId << "] PreProc done for buffer " << index);

#endif

            DLOG("VideoStreamer[" << m_streamId << "] RunAsync on buffer " << index
                 << " (process_count=" << m_odArgs.od_process_count << ")");
            std::ignore = m_ie->RunAsync(m_odInputs[index].data, &m_odArgs, (void*)m_odOutputs[index].data());

#endif

            // Display Logic
            {
#ifdef USE_DXRT
                 std::unique_lock<std::mutex> lk(m_odArgs.lk);
                 if (m_odArgs.od_process_count > m_displayed_count) {
                     int display_idx = m_displayed_count % FRAME_BUFFERS;
                     DLOG("VideoStreamer[" << m_streamId << "] Displaying inference result "
                          << m_displayed_count << " from buffer " << display_idx
                          << " (" << m_odArgs.od_results[display_idx].size() << " detections)");

                     if (!m_frames[display_idx].empty()) {
                         cv::Mat displayFrame = m_frames[display_idx].clone();
#ifdef USE_OPENCV
                         DisplayBoundingBox(displayFrame, m_odArgs.od_results[display_idx], m_yoloParam.height, m_yoloParam.width, objectColors, m_yoloParam.postproc_type, true);

                         if (m_personTracker) {
                             std::vector<BoundingBox> personDets;
                             for (const auto& bb : m_odArgs.od_results[display_idx])
                                 if (bb.label == 0) personDets.push_back(bb);
                             // Update tracker using the raw (unmodified) stored frame.
                             m_personTracker->update(personDets, m_frames[display_idx],
                                                     (float)m_yoloParam.width,
                                                     (float)m_yoloParam.height);
                             DisplayPersonTracks(displayFrame, m_personTracker->tracks());

                             int seen = m_personTracker->nextId();
                             if (seen != m_lastEmittedPeopleCount) {
                                 m_lastEmittedPeopleCount = seen;
                                 emit peopleSeen(seen);
                             }
                         }
#endif

                         // Convert BGR→RGB directly into the QImage buffer to avoid
                         // an intermediate cv::Mat allocation and a qimg.copy() call.
                         QImage qimg(displayFrame.cols, displayFrame.rows, QImage::Format_RGB888);
                         cv::Mat rgbWrapper(displayFrame.rows, displayFrame.cols, CV_8UC3,
                                            qimg.bits(), static_cast<size_t>(qimg.bytesPerLine()));
                         cv::cvtColor(displayFrame, rgbWrapper, cv::COLOR_BGR2RGB);
                         DLOG("VideoStreamer[" << m_streamId << "] Emitting imageReady ("
                              << qimg.width() << "x" << qimg.height() << ")");
                         emit imageReady(std::move(qimg));
                     } else {
                         DLOG("VideoStreamer[" << m_streamId << "] Display frame " << display_idx
                              << " is empty, skipping");
                     }
                     m_displayed_count++;
                 } else {
                     DLOG("VideoStreamer[" << m_streamId << "] No new inference result yet"
                          << " (process=" << m_odArgs.od_process_count
                          << " displayed=" << m_displayed_count << ")");
                 }
#else
                 // No DXRT, just display frame
                 cv::Mat displayFrame = frame.clone();
                 cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
                 QImage qimg((const unsigned char*)displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
                 DLOG("VideoStreamer[" << m_streamId << "] Emitting imageReady (no-DXRT) "
                      << qimg.width() << "x" << qimg.height());
                 emit imageReady(qimg.copy());

                 // Artificial delay to match typical framerate if no inference
                 std::this_thread::sleep_for(std::chrono::milliseconds(30));
#endif
            }

            index = (index + 1) % FRAME_BUFFERS;
        }
        DLOG("VideoStreamer[" << m_streamId << "] Stream loop ended (stop=" << m_stop << ")");
#else
        // NO OPENCV - Dummy Mode
        DLOG("VideoStreamer[" << m_streamId << "] Entering dummy mode (no OpenCV)");
        int frameNum = 0;
        while (!m_stop) {
            QImage dummy(640, 480, QImage::Format_RGB888);
            dummy.fill(Qt::blue);

            QPainter p(&dummy);
            p.setBrush(Qt::red);
            p.drawRect((frameNum * 5) % 640, 200, 50, 50);
            p.setPen(Qt::white);
            p.drawText(10, 20, QString("Frame: %1").arg(frameNum));
            p.end();

            DLOG("VideoStreamer[" << m_streamId << "] Emitting dummy frame " << frameNum);
            emit imageReady(dummy);

            frameNum++;
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
        }
        DLOG("VideoStreamer[" << m_streamId << "] Dummy loop ended after " << frameNum << " frames");
#endif

#ifdef USE_DXRT
    } catch (const dxrt::Exception& e) {
        DLOG("VideoStreamer[" << m_streamId << "] dxrt::Exception: " << e.what());
        emit error(QString::fromStdString(e.what()));
#endif
    } catch (const std::exception& e) {
        DLOG("VideoStreamer[" << m_streamId << "] std::exception: " << e.what());
        emit error(QString::fromStdString(e.what()));
    }
    DLOG("VideoStreamer[" << m_streamId << "] Emitting finished");
    emit finished();
}
