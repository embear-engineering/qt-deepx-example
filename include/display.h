#pragma once

#include <list>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <map>
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "bbox.h"
#include "yolo.h"
#include "person_tracker.h"

void DisplayBoundingBox(cv::Mat& frame, 
                       std::vector<BoundingBox>& result, 
                       float OriginHeight, 
                       float OriginWidth, 
                       std::vector<cv::Scalar> ObjectColors, 
                       PostProcType type, 
                       bool ImageCenterAligned=false, 
                       float InputWidth=0.f, 
                       float InputHeight=0.f);

// Overlay person tracks on frame. Each active track is drawn with a distinct
// green rectangle and a persistent "ID: N" label.
void DisplayPersonTracks(cv::Mat& frame, const std::vector<PersonTrack>& tracks);