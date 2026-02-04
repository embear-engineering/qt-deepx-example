#include "videostreamer.h"
#include <QDebug>
#include <QPainter>
#include <thread>
#include <chrono>

#ifdef USE_DXRT
#ifdef USE_OPENCV
#include <utils/color_table.hpp>
#endif
#endif

VideoStreamer::VideoStreamer(int streamId, const std::string& modelPath, const YoloParam& yoloParam, const std::string& pipeline, QObject *parent)
    : QObject(parent), m_streamId(streamId), m_modelPath(modelPath), m_yoloParam(yoloParam), m_pipeline(pipeline), m_stop(false)
#ifdef USE_DXRT
      , m_ie(nullptr), m_yolo(nullptr)
#endif
{
    m_odOutputs.resize(FRAME_BUFFERS);
#ifdef USE_OPENCV
    m_odInputs.resize(FRAME_BUFFERS);
#endif
}

VideoStreamer::~VideoStreamer()
{
    stop();
#ifdef USE_DXRT
    if (m_ie) delete m_ie;
    if (m_yolo) delete m_yolo;
#endif
}

void VideoStreamer::stop()
{
    m_stop = true;
}

void VideoStreamer::process()
{
    try {
#ifdef USE_DXRT
        dxrt::InferenceOption op_od;
        op_od.devices.push_back(0); 

        m_ie = new dxrt::InferenceEngine(m_modelPath, op_od);
        
        m_yolo = new Yolo(m_yoloParam);
        if(!m_yolo->LayerReorder(m_ie->GetOutputs())) {
            emit error("Layer reorder failed");
            return;
        }

        // Setup buffers
        for(int i=0; i<FRAME_BUFFERS; i++) {
            m_odOutputs[i] = std::vector<uint8_t>(m_ie->GetOutputSize());
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

        // Callback
        std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> od_postProcCallBack = 
                    [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
        {
            auto arguments = (OdEstimationArgs*)arg;
            {
                std::unique_lock<std::mutex> lk(arguments->lk);
                int index = arguments->od_process_count;
                if(index >= FRAME_BUFFERS) {
                    index = index % FRAME_BUFFERS;
                } else if (index < 0) {
                    index = 0;
                }

                auto od_result = m_yolo->PostProc(outputs);
                arguments->od_results[index] = od_result;
                arguments->od_process_count = arguments->od_process_count + 1;
                arguments->frame_idx = arguments->frame_idx + 1;
            }
            return 0;
        };

        m_ie->RegisterCallback(od_postProcCallBack);
#endif

#ifdef USE_OPENCV
        cv::VideoCapture cap;
        if (m_pipeline.find("!") != std::string::npos) {
             cap.open(m_pipeline, cv::CAP_GSTREAMER);
        } else {
             cap.open(m_pipeline);
        }

        if(!cap.isOpened()) {
            emit error("Could not open pipeline: " + QString::fromStdString(m_pipeline));
            return;
        }

        int index = 0;
#ifdef USE_DXRT
        auto objectColors = dxapp::common::color_table;
#endif

        while(!m_stop) {
            cv::Mat frame;
            cap >> frame;
            if(frame.empty()) {
                break;
            }

            m_frames[index] = frame.clone(); 

#ifdef USE_DXRT

#ifdef USE_OPENCV

            PreProc(frame, m_odInputs[index], true, true, 114);

#endif

            std::ignore = m_ie->RunAsync(m_odInputs[index].data, &m_odArgs, (void*)m_odOutputs[index].data());

#endif

            // Display Logic
            {
#ifdef USE_DXRT
                 static int displayed_count = 0;
                 std::unique_lock<std::mutex> lk(m_odArgs.lk);
                 if (m_odArgs.od_process_count > displayed_count) {
                     int display_idx = displayed_count % FRAME_BUFFERS;
                     
                     if (!m_frames[display_idx].empty()) {
                         cv::Mat displayFrame = m_frames[display_idx].clone();
#ifdef USE_OPENCV
                         DisplayBoundingBox(displayFrame, m_odArgs.od_results[display_idx], m_yoloParam.height, m_yoloParam.width, objectColors, m_yoloParam.postproc_type, true);
#endif
                         
                         cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
                         QImage qimg((const unsigned char*)displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
                         emit imageReady(qimg.copy()); 
                     }
                     displayed_count++;
                 }
#else
                 // No DXRT, just display frame
                 cv::Mat displayFrame = frame.clone();
                 cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
                 QImage qimg((const unsigned char*)displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
                 emit imageReady(qimg.copy());
                 
                 // Artificial delay to match typical framerate if no inference
                 std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
#endif
            }

            index = (index + 1) % FRAME_BUFFERS;
        }
#else 
        // NO OPENCV - Dummy Mode
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

            emit imageReady(dummy);
            
            frameNum++;
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
        }
#endif

    } catch (const std::exception& e) {
        emit error(QString::fromStdString(e.what()));
    }
    emit finished();
}