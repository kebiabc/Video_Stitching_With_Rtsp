#ifndef IMAGE_STITCHING_APP_H
#define IMAGE_STITCHING_APP_H

#include "opencv2/opencv.hpp"
#include "sensor_data_interface.h"
#include "image_stitcher.h"

class App {
public:
    App();  // 构造函数

    [[noreturn]] void run_stitching();  // 主运行函数

private:
    // 私有成员变量
    SensorDataInterface sensor_data_interface_;
    ImageStitcher image_stitcher_;
    cv::UMat image_concat_umat_;
    int total_cols_;

    // 私有成员函数
    void PushFrame(const cv::UMat& frame);  // 推流函数
};

#endif // IMAGE_STITCHING_APP_H