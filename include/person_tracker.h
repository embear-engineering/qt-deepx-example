#pragma once

#include <opencv2/tracking.hpp>
#include <vector>
#include <string>
#include "bbox.h"

struct PersonTrack {
    int id;
    cv::Ptr<cv::TrackerVit> tracker;
    cv::Rect2d roi;       // last known bounding box in frame pixel space
    int missedFrames;     // consecutive frames without a matching detection
};

class PersonTracker {
public:
    // modelPath: path to vitTracker.onnx
    // maxMissed: frames a track survives without a matching detection before being dropped
    // iouThreshold: minimum IoU to associate a detection with an existing track
    explicit PersonTracker(const std::string& modelPath,
                           int maxMissed = 5,
                           float iouThreshold = 0.3f);

    // Update tracker state for one frame.
    // detections: BoundingBoxes from YOLO whose label == 0 ("person").
    // frame:      raw captured frame at its native resolution (BGR, unmodified).
    // modelW/H:   YOLO model input dimensions used for coordinate back-projection.
    // Returns the currently active tracks.
    const std::vector<PersonTrack>& update(const std::vector<BoundingBox>& detections,
                                           cv::Mat& frame,
                                           float modelW, float modelH);

    const std::vector<PersonTrack>& tracks() const { return m_tracks; }

private:
    std::string m_modelPath;
    int m_maxMissed;
    float m_iouThreshold;
    int m_nextId = 0;
    std::vector<PersonTrack> m_tracks;

    // Convert a BoundingBox (model space, letterboxed) to a pixel-space Rect2d.
    cv::Rect2d toFrameRect(const BoundingBox& b,
                           float frameW, float frameH,
                           float modelW, float modelH) const;

    float iou(const cv::Rect2d& a, const cv::Rect2d& b) const;
};
