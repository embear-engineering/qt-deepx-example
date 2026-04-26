#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <utils/common_util.hpp>
#include "yolo.h"
#include "nms.h"

// #define DUMP_DATA

void YoloLayerParam::Show()
{
    std::cout << "    - LayerParam: [ name : " << name << ", " << numGridX << " x " << numGridY << " x " << numBoxes << "boxes" << "], anchorWidth [";
    for(auto &w : anchorWidth) std::cout << w << ", ";
    std::cout << "], anchorHeight [";
    for(auto &h : anchorHeight) std::cout << h << ", ";
    std::cout << "], tensor index [";
    for(auto &t : tensorIdx) std::cout << t << ", ";
    std::cout << "]" << std::endl;
}
void YoloParam::Show()
{
    std::cout << "  YoloParam: " << std::endl << "    - conf_threshold: " << confThreshold << ", "
        << "score_threshold: " << scoreThreshold << ", "
        << "iou_threshold: " << iouThreshold << ", "
        << "num_classes: " << numClasses << ", "
        << "num_layers: " << layers.size() << std::endl;
    for(auto &layer:layers) layer.Show();
    std::cout << "    - classes: [";
    for(auto &c : classNames) std::cout << c << ", ";
    std::cout << "]" << std::endl;
}
Yolo::Yolo() { }
Yolo::~Yolo() { }
Yolo::Yolo(YoloParam &_cfg) :cfg(_cfg)
{
    if(cfg.layers.empty())
    {
        is_onnx_output = true;
    }
    else
    {
        anchorSize = cfg.layers[0].anchorWidth.size();
    }
    
    if(cfg.numBoxes==0)
    {   
        if(cfg.layers.size() > 0)
        {
            for(auto &layer:cfg.layers)
            {
                cfg.numBoxes += layer.numGridX*layer.numGridY*layer.numBoxes;
            }
        }
        else
        {
            std::cerr << "[DXAPP] [WARN] The numBoxes is not set. Please check the numBoxes." << std::endl; 
        }
    }

    if(cfg.numBoxes > 0)
    {
        //allocate memory
        Boxes = std::vector<float>(cfg.numBoxes*4);
        Keypoints = std::vector<float>(cfg.numBoxes*51);
    }
    
    for(size_t i=0; i<cfg.numClasses; i++)
    {
        std::vector<std::pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
    cfg.postproc_type = cfg.postproc_type;
}

bool Yolo::LayerReorder(dxrt::Tensors output_info)
{
    for(size_t i=0;i<output_info.size();i++)
    {
        if(cfg.onnxOutputName == output_info[i].name())
        {
            cfg.numBoxes = output_info.front().shape()[1];
            std::cout << "cfg.numBoxes: " << cfg.numBoxes << std::endl; 
            onnxOutputIdx.emplace_back(i);
            Boxes.clear();
            Keypoints.clear();
            Boxes = std::vector<float>(cfg.numBoxes*4);
            Keypoints = std::vector<float>(cfg.numBoxes*51);
        }
    }
    if(onnxOutputIdx.size() > 0)
    {
        cfg.Show();
        std::cout << "YOLO created : " << cfg.numBoxes << " boxes, " << cfg.numClasses << " classes, "<< std::endl;
        cfg.layers.clear();
        return true;
    }
    
    // Debug print if ONNX name not found
    std::cout << "[DXAPP] [DBG] ONNX output name '" << cfg.onnxOutputName << "' not found in model outputs. Trying layer matching..." << std::endl;
    
    std::vector<YoloLayerParam> temp;
    for(size_t i=0;i<output_info.size();i++)
    {   
        for(size_t j=0;j<cfg.layers.size();j++)
        {
            if(output_info[i].name() == cfg.layers[j].name)
            {
                cfg.layers[j].tensorIdx.clear();
                cfg.layers[j].tensorIdx.push_back(static_cast<int32_t>(i));
                temp.emplace_back(cfg.layers[j]);
                break;
            }
        }
    }
    if(temp.size() != output_info.size())
    {
        std::cerr << "[DXAPP] [ER] Yolo::LayerReorder : Output tensor size mismatch. Please check the model output configuration." << std::endl;
        std::cerr << "[DXAPP] [INFO] Model Outputs found: " << std::endl;
        for(auto& out : output_info) {
             std::cerr << " - " << out.name() << " (shape: [";
             for(auto s : out.shape()) std::cerr << s << ",";
             std::cerr << "])" << std::endl;
        }
        return false;
    }
    if(temp.empty())
    {
        std::cerr << "[DXAPP] [ER] Yolo::LayerReorder : Layer information is missing. This is only supported when USE_ORT=ON. Please modify and rebuild." << std::endl;
        return false;
    }
    cfg.layers.clear();
    cfg.layers = temp;
    cfg.Show();
    return true;
}

static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
{
    if(a.first > b.first)
        return true;
    else
        return false;
};

std::vector<BoundingBox> Yolo::PostProc(dxrt::TensorPtrs& dataSrc)
{
    for(uint32_t label=0; label<cfg.numClasses; label++) {
        ScoreIndices[label].clear();
    }
    Result.clear();

    if(cfg.layers.empty())
    {
        for(auto &data:dataSrc)
        {
            if(cfg.onnxOutputName == data->name())
            {
                auto num_elements = data->shape()[1];
                onnx_post_processing(dataSrc, num_elements);
                break;
            }
        }
    }
    else
    {
        raw_post_processing(dataSrc);
    }

    for(uint32_t label=0 ; label<cfg.numClasses ; label++)
    {
        sort(ScoreIndices[label].begin(), ScoreIndices[label].end(), scoreComapre);
    }
    
    Nms(
        cfg.numClasses,
        0,
        cfg.classNames, 
        ScoreIndices, Boxes.data(), Keypoints.data(), cfg.iouThreshold,
        Result,
        0
    );

    return Result;
}

void Yolo::onnx_post_processing(dxrt::TensorPtrs &outputs, int64_t num_elements) {
    int x = 0, y = 1, w = 2, h = 3;
    float scoreThreshold = cfg.scoreThreshold;
    float conf_threshold = cfg.confThreshold;
    auto *dataSrc = static_cast<void*>(outputs[onnxOutputIdx[0]]->data());
    auto data_pitch_size = outputs[onnxOutputIdx[0]]->shape()[2];
    cv::Mat raw_data;
    int class_index = 5;
    if(cfg.postproc_type == PostProcType::YOLOV8) 
    {
        data_pitch_size = outputs[onnxOutputIdx[0]]->shape()[1];
        num_elements = outputs[onnxOutputIdx[0]]->shape()[2];
        raw_data = cv::Mat(data_pitch_size, num_elements, CV_32F, (float*)dataSrc);
        raw_data = raw_data.t();
        dataSrc = static_cast<void*>(raw_data.data);
        class_index = 4;
    }

    for(int boxIdx=0;boxIdx<num_elements;boxIdx++)
    {
        auto *data = static_cast<float*>(dataSrc) + (data_pitch_size * boxIdx);
        auto obj_conf = data[4];
        if(cfg.postproc_type == PostProcType::YOLOV8)
        {
            obj_conf = 1.f;
        }
        if(obj_conf>conf_threshold)
        {
            int max_cls = -1;
            float max_score = scoreThreshold;
            for(int cls=0; cls<(int)cfg.numClasses; cls++)
            {
                auto cls_data = data[class_index + cls];
                auto cls_conf = obj_conf * cls_data;
                if(cls_conf > max_score)
                {
                    max_cls = cls;
                    max_score = cls_conf;
                }
                else continue;
            }
            if(max_cls > -1)
            {
                ScoreIndices[max_cls].emplace_back(max_score, boxIdx);
                Boxes[boxIdx*4+0] = data[x] - data[w] / 2.; /*x1*/
                Boxes[boxIdx*4+1] = data[y] - data[h] / 2.; /*y1*/
                Boxes[boxIdx*4+2] = data[x] + data[w] / 2.; /*x2*/
                Boxes[boxIdx*4+3] = data[y] + data[h] / 2.; /*y2*/

                switch(cfg.postproc_type)
                {
                    case PostProcType::POSE: // POSE
                        for(int k = 0; k < 17; k++)
                        {
                            int kptIdx = (k * 3) + 6;
                            Keypoints[boxIdx*51+k*3+0] = data[kptIdx + 0];
                            Keypoints[boxIdx*51+k*3+1] = data[kptIdx + 1];
                            Keypoints[boxIdx*51+k*3+2] = data[kptIdx + 2];
                        }
                        break;
                    case PostProcType::FACE: // FACE
                        for(int k = 0; k < 5; k++)
                        {
                            int kptIdx = (k * 2) + 5;
                            Keypoints[boxIdx*51+k*3+0] = data[kptIdx + 0];
                            Keypoints[boxIdx*51+k*3+1] = data[kptIdx + 1];
                            Keypoints[boxIdx*51+k*3+2] = 0.5;
                        }
                        break;
                    default:
                        break;
                }
            }
            else continue;
        }
    }
}

void Yolo::raw_post_processing(dxrt::TensorPtrs &outputs) {
    int boxIdx = 0;
    int x = 0, y = 1, w = 2, h = 3;
    std::vector<float> box_temp(4);
    if(cfg.postproc_type == PostProcType::YOLOV8)
    {
        // 6-output per-scale decoupled head: 3 DFL box regression tensors (cv2, 64ch) and
        // 3 class score tensors (cv3, numClasses ch) at scales 80x80, 40x40, 20x20.
        // LayerReorder re-orders layers to match model output order (not necessarily [reg,cls] pairs),
        // so classify by channel count and sort by spatial size to get correct [reg,cls] pairs.
        std::vector<std::pair<int, int>> reg_tensors;  // (spatial_size, tensor_idx)
        std::vector<std::pair<int, int>> cls_tensors;
        for (const auto& layer : cfg.layers) {
            int tidx = layer.tensorIdx[0];
            if (outputs[tidx]->shape().size() < 4) continue;
            int ch = outputs[tidx]->shape()[1];
            int sp = outputs[tidx]->shape()[2] * outputs[tidx]->shape()[3];
            if (ch == 64) {
                reg_tensors.emplace_back(sp, tidx);
            } else if (ch == static_cast<int>(cfg.numClasses)) {
                cls_tensors.emplace_back(sp, tidx);
            }
        }
        auto sort_desc = [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
            return a.first > b.first;
        };
        std::sort(reg_tensors.begin(), reg_tensors.end(), sort_desc);
        std::sort(cls_tensors.begin(), cls_tensors.end(), sort_desc);

        for (size_t i = 0; i < 3 && i < reg_tensors.size() && i < cls_tensors.size(); ++i) {
            int reg_idx = reg_tensors[i].second;
            int cls_idx = cls_tensors[i].second;
            const float* reg_data = static_cast<const float*>(outputs[reg_idx]->data());
            const float* cls_data = static_cast<const float*>(outputs[cls_idx]->data());

            int H = outputs[cls_idx]->shape()[2];
            int W = outputs[cls_idx]->shape()[3];
            int stride = cfg.width / W;
            int num_grid = H * W;

            for (int gh = 0; gh < H; ++gh) {
                for (int gw = 0; gw < W; ++gw) {
                    int sp = gh * W + gw;

                    int max_cls = -1;
                    float max_cls_conf = cfg.scoreThreshold;
                    for (int c = 0; c < static_cast<int>(cfg.numClasses); ++c) {
                        float conf = cls_data[c * num_grid + sp];
                        if (conf > max_cls_conf) {
                            max_cls_conf = conf;
                            max_cls = c;
                        }
                    }

                    if (max_cls != -1) {
                        // DFL decoding: softmax over 16 bins per coordinate, then weighted sum
                        float dist[4];
                        for (int k = 0; k < 4; ++k) {
                            float max_val = -std::numeric_limits<float>::infinity();
                            for (int d = 0; d < 16; ++d) {
                                float v = reg_data[(k * 16 + d) * num_grid + sp];
                                if (v > max_val) max_val = v;
                            }
                            float exp_sum = 0.0f, weighted_sum = 0.0f;
                            for (int d = 0; d < 16; ++d) {
                                float e = std::exp(reg_data[(k * 16 + d) * num_grid + sp] - max_val);
                                exp_sum += e;
                                weighted_sum += e * static_cast<float>(d);
                            }
                            dist[k] = weighted_sum / exp_sum;
                        }

                        float ax = gw + 0.5f, ay = gh + 0.5f;
                        Boxes[boxIdx * 4 + 0] = (ax - dist[0]) * stride;  // x1
                        Boxes[boxIdx * 4 + 1] = (ay - dist[1]) * stride;  // y1
                        Boxes[boxIdx * 4 + 2] = (ax + dist[2]) * stride;  // x2
                        Boxes[boxIdx * 4 + 3] = (ay + dist[3]) * stride;  // y2
                        ScoreIndices[max_cls].emplace_back(max_cls_conf, boxIdx);
                        boxIdx++;
                    }
                }
            }
        }
    }
    else if(anchorSize > 0)
    {
        for(auto &layer:cfg.layers)        
        {
            int strideX = cfg.width / layer.numGridX;
            int strideY = cfg.height / layer.numGridY;
            int numGridX = layer.numGridX;
            int numGridY = layer.numGridY;
            int tensorIdx = layer.tensorIdx[0];
            float scale_x_y = layer.scaleX;
            auto *output_per_layer = static_cast<float*>(outputs[tensorIdx]->data());
            for(int gY=0; gY<numGridY; gY++)
            {
                for(int gX=0; gX<numGridX; gX++)
                {
                    for(size_t box=0; box<layer.anchorWidth.size(); box++)
                    { 
                        bool boxDecoded = false;
                        int objectness_idx = ((box * (cfg.numClasses + 5) + 4) * numGridY * numGridX)
                                                + (gY * numGridX) 
                                                + gX;
                        if(cfg.postproc_type == PostProcType::FACE)
                        {
                            objectness_idx = ((box * (cfg.numClasses + 15) + 15) * numGridY * numGridX)
                                                + (gY * numGridX) 
                                                + gX;
                        }
                        auto objectness_score = sigmoid(output_per_layer[objectness_idx]);
                        
                        if(objectness_score > cfg.confThreshold)
                        {
                            int max_cls = -1;
                            float max_score = cfg.scoreThreshold;
                            for(int cls=0; cls<(int)cfg.numClasses;cls++)
                            {   
                                int cls_conf_idx = ((box * (cfg.numClasses + 5) + 5 + cls) * numGridY * numGridX) + (gY * numGridX) + gX;
                                if(cfg.postproc_type == PostProcType::FACE)
                                {
                                    cls_conf_idx = ((box * (cfg.numClasses + 15) + 4 + cls) * numGridY * numGridX) + (gY * numGridX) + gX;
                                }
                                float cls_conf = objectness_score * sigmoid(output_per_layer[cls_conf_idx]);
                                if(cls_conf > max_score)
                                {
                                    max_cls = cls;
                                    max_score = cls_conf;
                                }
                                else continue;
                            }
                            if(max_cls > -1)
                            {
                                ScoreIndices[max_cls].emplace_back(max_score, boxIdx);
                                if(!boxDecoded)
                                {
                                    for(int i = 0; i < 4; i++)
                                    {
                                        int box_idx = ((box * (cfg.numClasses + 5) + i) * numGridY * numGridX) + (gY * numGridX) + gX;
                                        if(cfg.postproc_type == PostProcType::FACE)
                                        {
                                            box_idx = ((box * (cfg.numClasses + 15) + i) * numGridY * numGridX) + (gY * numGridX) + gX;
                                        }
                                        box_temp[i] = output_per_layer[box_idx];
                                    }
                                    if(scale_x_y==0)
                                    {
                                        box_temp[x] = ( sigmoid(box_temp[x]) * 2. - 0.5 + gX ) * strideX;
                                        box_temp[y] = ( sigmoid(box_temp[y]) * 2. - 0.5 + gY ) * strideY;
                                    }
                                    else
                                    {
                                        box_temp[x] = (sigmoid(box_temp[x] * scale_x_y  - 0.5 * (scale_x_y - 1)) + gX) * strideX;
                                        box_temp[y] = (sigmoid(box_temp[y] * scale_x_y  - 0.5 * (scale_x_y - 1)) + gY) * strideY;
                                    }
                                    box_temp[w] = pow((sigmoid(box_temp[w]) * 2.), 2) * layer.anchorWidth[box];
                                    box_temp[h] = pow((sigmoid(box_temp[h]) * 2.), 2) * layer.anchorHeight[box];
                                    Boxes[boxIdx*4+0] = box_temp[x] - box_temp[w] / 2.; /*x1*/
                                    Boxes[boxIdx*4+1] = box_temp[y] - box_temp[h] / 2.; /*y1*/
                                    Boxes[boxIdx*4+2] = box_temp[x] + box_temp[w] / 2.; /*x2*/
                                    Boxes[boxIdx*4+3] = box_temp[y] + box_temp[h] / 2.; /*y2*/
                                    if(cfg.postproc_type == PostProcType::FACE)
                                    {
                                        for(int k = 0; k < 5; k++)
                                        {
                                            Keypoints[boxIdx*51+k*3+0] = 
                                                output_per_layer[((box * (cfg.numClasses + 15) + 5 + (k * 2)) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX] * layer.anchorWidth[box] + (gX * strideX);
                                            Keypoints[boxIdx*51+k*3+1] = 
                                                output_per_layer[((box * (cfg.numClasses + 15) + 6 + (k * 2)) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX] * layer.anchorHeight[box] + (gY * strideY);
                                            Keypoints[boxIdx*51+k*3+2] = 0.5;
                                        }
                                    }
                                    boxDecoded = true;

                                }
                            }
                        }
                        boxIdx++;
                    }
                }
            }
        }
    }

}

