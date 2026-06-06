#include "person_tracker.h"
#include "debug.h"
#include <algorithm>
#include <iostream>
#include <limits>

PersonTracker::PersonTracker(const std::string& modelPath, int maxMissed, float iouThreshold)
    : m_modelPath(modelPath), m_maxMissed(maxMissed), m_iouThreshold(iouThreshold)
{
    DLOG("PersonTracker: model=" << modelPath
         << " maxMissed=" << maxMissed
         << " iouThreshold=" << iouThreshold);
}

cv::Rect2d PersonTracker::toFrameRect(const BoundingBox& b,
                                       float frameW, float frameH,
                                       float modelW, float modelH) const
{
    // Inverse of the letterbox transform used by DisplayBoundingBox:
    //   model_coord = pixel_coord * r + padding
    //   r = min(modelW / frameW, modelH / frameH)  (uniform scale keeping aspect ratio)
    float r = std::min(modelW / frameW, modelH / frameH);
    float wPad = (modelW - frameW * r) / 2.0f;
    float hPad = (modelH - frameH * r) / 2.0f;

    float x1 = (b.box[0] - wPad) / r;
    float y1 = (b.box[1] - hPad) / r;
    float x2 = (b.box[2] - wPad) / r;
    float y2 = (b.box[3] - hPad) / r;

    x1 = std::max(0.0f, std::min(frameW - 1, x1));
    y1 = std::max(0.0f, std::min(frameH - 1, y1));
    x2 = std::max(0.0f, std::min(frameW - 1, x2));
    y2 = std::max(0.0f, std::min(frameH - 1, y2));

    return cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
}

float PersonTracker::iou(const cv::Rect2d& a, const cv::Rect2d& b) const
{
    double xInter1 = std::max(a.x, b.x);
    double yInter1 = std::max(a.y, b.y);
    double xInter2 = std::min(a.x + a.width,  b.x + b.width);
    double yInter2 = std::min(a.y + a.height, b.y + b.height);

    double interW = std::max(0.0, xInter2 - xInter1);
    double interH = std::max(0.0, yInter2 - yInter1);
    double inter  = interW * interH;
    if (inter == 0.0) return 0.0f;

    double aArea = a.width * a.height;
    double bArea = b.width * b.height;
    return static_cast<float>(inter / (aArea + bArea - inter));
}

const std::vector<PersonTrack>& PersonTracker::update(
        const std::vector<BoundingBox>& detections,
        cv::Mat& frame,
        float modelW, float modelH)
{
    float frameW = static_cast<float>(frame.cols);
    float frameH = static_cast<float>(frame.rows);

    // --- Step 1: Convert detections to pixel-space rects ---
    std::vector<cv::Rect2d> detRects;
    detRects.reserve(detections.size());
    for (const auto& bb : detections)
        detRects.push_back(toFrameRect(bb, frameW, frameH, modelW, modelH));

    // --- Step 2: Update all existing trackers ---
    for (auto& track : m_tracks) {
        cv::Rect trackRect(static_cast<int>(track.roi.x),
                           static_cast<int>(track.roi.y),
                           static_cast<int>(track.roi.width),
                           static_cast<int>(track.roi.height));
        bool ok = track.tracker->update(frame, trackRect);
        if (ok) {
            track.roi = cv::Rect2d(trackRect.x, trackRect.y,
                                   trackRect.width, trackRect.height);
        } else {
            track.missedFrames++;
        }
    }

    // --- Step 3 & 4: Greedy IoU matching – update existing tracks with best detection ---
    std::vector<bool> detMatched(detRects.size(), false);

    for (auto& track : m_tracks) {
        float bestIou  = m_iouThreshold;
        int   bestDet  = -1;

        for (int di = 0; di < (int)detRects.size(); ++di) {
            if (detMatched[di]) continue;
            float score = iou(track.roi, detRects[di]);
            if (score > bestIou) {
                bestIou = score;
                bestDet = di;
            }
        }

        if (bestDet >= 0) {
            detMatched[bestDet] = true;
            track.missedFrames  = 0;
            // Re-initialise tracker to the confirmed detection position.
            const cv::Rect2d& r = detRects[bestDet];
            cv::Rect initRect(static_cast<int>(r.x), static_cast<int>(r.y),
                              static_cast<int>(r.width), static_cast<int>(r.height));
            track.tracker->init(frame, initRect);
            track.roi = detRects[bestDet];
            DLOG("PersonTracker: re-init track " << track.id
                 << " iou=" << bestIou);
        } else {
            track.missedFrames++;
            DLOG("PersonTracker: missed track " << track.id
                 << " missed=" << track.missedFrames);
        }
    }

    // --- Step 5: Create new tracks for unmatched detections ---
    for (int di = 0; di < (int)detRects.size(); ++di) {
        if (detMatched[di]) continue;
        const cv::Rect2d& r = detRects[di];
        if (r.width <= 0 || r.height <= 0) continue;

        cv::TrackerVit::Params params;
        params.net = m_modelPath;

        cv::Ptr<cv::TrackerVit> tracker = cv::TrackerVit::create(params);
        cv::Rect initRect(static_cast<int>(r.x), static_cast<int>(r.y),
                          static_cast<int>(r.width), static_cast<int>(r.height));
        tracker->init(frame, initRect);

        PersonTrack newTrack;
        newTrack.id           = m_nextId++;
        newTrack.tracker      = tracker;
        newTrack.roi          = r;
        newTrack.missedFrames = 0;
        m_tracks.push_back(std::move(newTrack));

        DLOG("PersonTracker: new track id=" << newTrack.id
             << " at (" << r.x << "," << r.y << " " << r.width << "x" << r.height << ")");
    }

    // --- Step 6: Remove stale tracks ---
    m_tracks.erase(
        std::remove_if(m_tracks.begin(), m_tracks.end(),
                       [this](const PersonTrack& t) {
                           return t.missedFrames > m_maxMissed;
                       }),
        m_tracks.end());

    DLOG("PersonTracker: " << m_tracks.size() << " active track(s)");
    return m_tracks;
}
