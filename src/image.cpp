#include "image.h"
#include "debug.h"
#include <opencv2/opencv.hpp>

void PreProc(cv::Mat& src, cv::Mat& dest, bool keepRatio, bool bgr2rgb, uint8_t padValue)
{
    DLOG("PreProc: src=" << src.cols << "x" << src.rows
         << " dest=" << dest.cols << "x" << dest.rows
         << " keepRatio=" << keepRatio << " bgr2rgb=" << bgr2rgb
         << " padValue=" << (int)padValue);
    cv::Mat Resized;
    if(keepRatio)
    {
        float dw, dh;
        uint16_t top, bottom, left, right;
        float ratioDest = (float)dest.cols/dest.rows;
        float ratioSrc = (float)src.cols/src.rows;
        int newWidth, newHeight;
        if(ratioSrc < ratioDest)
        {
            newHeight = dest.rows;
            newWidth = newHeight * ratioSrc;
        }
        else
        {
            newWidth = dest.cols;
            newHeight = newWidth / ratioSrc;
        }
        cv::Mat src2;
        cv::resize(src, src2, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        dw = (dest.cols - src2.cols)/2.;
        dh = (dest.rows - src2.rows)/2.;
        top    = (uint16_t)round(dh - 0.1);
        bottom = (uint16_t)round(dh + 0.1);
        left   = (uint16_t)round(dw - 0.1);
        right  = (uint16_t)round(dw + 0.1);
        DLOG("PreProc: keepRatio resize -> " << newWidth << "x" << newHeight
             << " pad top=" << top << " bottom=" << bottom
             << " left=" << left << " right=" << right);
        cv::copyMakeBorder(src2, dest, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(padValue,padValue,padValue));
    }
    else
    {
        DLOG("PreProc: simple resize to " << dest.cols << "x" << dest.rows);
        cv::resize(src, dest, cv::Size(dest.cols, dest.rows), 0, 0, cv::INTER_LINEAR);
    }
    if(bgr2rgb)
    {
        DLOG("PreProc: applying BGR->RGB conversion");
        cv::cvtColor(dest, dest, cv::COLOR_BGR2RGB);
    }
}