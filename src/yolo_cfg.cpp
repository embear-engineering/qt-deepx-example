#include "yolo.h"

YoloParam createYolo26xParam(const std::vector<std::string>& classNames,
                             float scoreThreshold,
                             float nmsThreshold)
{
    YoloParam param;
    param.height = 640;
    param.width = 640;
    param.confThreshold = scoreThreshold;
    param.scoreThreshold = scoreThreshold;
    param.iouThreshold = nmsThreshold;
    param.numBoxes = 0;  // determined at runtime from model output shape
    param.numClasses = static_cast<uint32_t>(classNames.size());
    param.onnxOutputName = "output0";
    param.layers = {};   // empty → onnx_post_processing path
    param.classNames = classNames;
    param.postproc_type = PostProcType::YOLOV26;
    return param;
}
