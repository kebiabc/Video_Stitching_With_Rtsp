#ifndef IMAGE_STITCHING_APP_H
#define IMAGE_STITCHING_APP_H

#include "opencv2/opencv.hpp"
#include "sensor_data_interface.h"
#include "image_stitcher.h"
#include <thread>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>

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
    
    std::queue<cv::UMat> frame_buffer_; // 缓冲区队列，存储待推送的帧
    const size_t max_buffer_size = 3;  // 缓冲区最大帧数
    std::mutex buffer_mutex_;  // 保护缓冲区的互斥锁
    std::condition_variable buffer_cond_; // 条件变量，用于同步推流和拼接

    std::atomic<bool> is_running_;  // 控制运行状态，确保线程停止
    std::thread stitching_thread_;  // 拼接线程
    std::thread streaming_thread_;  // 推流线程

    // 私有成员函数
    void PushFrame(const cv::UMat& frame);  // 推流函数
    void StitchFrames();  // 拼接图像帧的函数
    void StreamFrames();  // 推流图像帧的函数
};

#endif // IMAGE_STITCHING_APP_H
