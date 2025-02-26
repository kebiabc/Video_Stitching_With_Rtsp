#ifndef APP_H
#define APP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "sensor_data_interface.h"
#include "image_stitcher.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

class App {
public:
    App();
    void run_stitching();

private:
    void InitializeStreaming();
    void PushFrame(const cv::UMat& frame);
    void StitchFrames();
    void StreamFrames();

    SensorDataInterface sensor_data_interface_;
    ImageStitcher image_stitcher_;
    cv::UMat image_concat_umat_;
    int total_cols_;

    std::thread stitching_thread_;
    std::thread streaming_thread_;

    std::queue<cv::UMat> frame_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cond_;
    bool is_running_;

    // 添加推流相关的成员变量
    AVFormatContext* fmt_ctx_;       // FFmpeg 格式上下文
    AVStream* video_stream_;         // 视频流
    AVCodecContext* codec_ctx_;      // 编码器上下文
    SwsContext* sws_ctx_;            // 图像缩放上下文
};

#endif // APP_H
