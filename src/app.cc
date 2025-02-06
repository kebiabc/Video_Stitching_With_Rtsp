#include "app.h"
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <chrono>

#include "image_stitcher.h"
#include "stitching_param_generater.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/timestamp.h>
#include <libavutil/error.h>
}

// 构造函数
App::App() {
    sensor_data_interface_.InitVideoCapture();

    std::vector<cv::UMat> first_image_vector = std::vector<cv::UMat>(sensor_data_interface_.num_img_);
    std::vector<cv::Mat> first_mat_vector = std::vector<cv::Mat>(sensor_data_interface_.num_img_);
    std::vector<cv::UMat> reproj_xmap_vector;
    std::vector<cv::UMat> reproj_ymap_vector;
    std::vector<cv::UMat> undist_xmap_vector;
    std::vector<cv::UMat> undist_ymap_vector;
    std::vector<cv::Rect> image_roi_vect;

    std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
    sensor_data_interface_.get_image_vector(first_image_vector, image_mutex_vector);

    for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
        first_image_vector[i].copyTo(first_mat_vector[i]);
    }

    StitchingParamGenerator stitching_param_generator(first_mat_vector);

    stitching_param_generator.GetReprojParams(
        undist_xmap_vector,
        undist_ymap_vector,
        reproj_xmap_vector,
        reproj_ymap_vector,
        image_roi_vect
    );

    image_stitcher_.SetParams(
        100,
        undist_xmap_vector,
        undist_ymap_vector,
        reproj_xmap_vector,
        reproj_ymap_vector,
        image_roi_vect
    );
    total_cols_ = 0;
    for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
        total_cols_ += image_roi_vect[i].width;
    }
    image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

// 推流函数
void App::PushFrame(const cv::UMat& frame) {
    static bool initialized = false;
    static AVFormatContext* fmt_ctx = nullptr;
    static AVStream* video_stream = nullptr;
    static AVCodecContext* codec_ctx = nullptr;
    static SwsContext* sws_ctx = nullptr;
    static int64_t frame_index = 0;

    if (!initialized) {
        avformat_network_init();
        avformat_alloc_output_context2(&fmt_ctx, nullptr, "rtsp", "rtsp://192.168.1.81:8554/live");

        AVCodec* codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) {
            std::cerr << "Codec H.264 not found!" << std::endl;
            return;
        }

        video_stream = avformat_new_stream(fmt_ctx, codec);
        codec_ctx = avcodec_alloc_context3(codec);
        codec_ctx->width = frame.cols;
        codec_ctx->height = frame.rows;
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->time_base = {1, 60};
        codec_ctx->framerate = {60, 1};
        codec_ctx->bit_rate = 8 * 1024 * 1024;
        codec_ctx->gop_size = 50;

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            std::cerr << "Failed to open codec!" << std::endl;
            return;
        }

        sws_ctx = sws_getContext(frame.cols, frame.rows, AV_PIX_FMT_BGR24,
                                frame.cols, frame.rows, AV_PIX_FMT_YUV420P,
                                SWS_BICUBIC, nullptr, nullptr, nullptr);

        avcodec_parameters_from_context(video_stream->codecpar, codec_ctx);
        video_stream->time_base = codec_ctx->time_base;

        if (avformat_write_header(fmt_ctx, nullptr) < 0) {
            std::cerr << "Failed to write header!" << std::endl;
            return;
        }

        initialized = true;
    }

    AVFrame* av_frame = av_frame_alloc();
    av_frame->width = codec_ctx->width;
    av_frame->height = codec_ctx->height;
    av_frame->format = codec_ctx->pix_fmt;
    av_frame_get_buffer(av_frame, 0);

    uint8_t* bgr_data[1] = {frame.getMat(cv::ACCESS_READ).data};
    int bgr_linesize[1] = {int(frame.step)};
    sws_scale(sws_ctx, bgr_data, bgr_linesize, 0, frame.rows, av_frame->data, av_frame->linesize);

    int64_t duration = av_rescale_q(1, {1, 30}, codec_ctx->time_base);
    av_frame->pts = frame_index * duration;
    frame_index++;

    AVPacket* pkt = av_packet_alloc();
    avcodec_send_frame(codec_ctx, av_frame);
    while (avcodec_receive_packet(codec_ctx, pkt) == 0) {
        av_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    av_frame_free(&av_frame);
}

// 主运行函数
[[noreturn]] void App::run_stitching() {
    std::vector<cv::UMat> image_vector(sensor_data_interface_.num_img_);
    std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
    std::vector<cv::UMat> images_warped_vector(sensor_data_interface_.num_img_);
    std::thread record_videos_thread(
        &SensorDataInterface::RecordVideos,
        &sensor_data_interface_
    );

    size_t frame_idx = 0;

    // 图像中一个固定点 (x, y)
    int x = 1000;  // 源图像中的 x 坐标
    int y = 500;   // 源图像中的 y 坐标

    // 用于计算拼接后图像中的位置
    int total_offset_x = 0; // 总的 x 偏移量
    const auto& roi_vector = image_stitcher_.getRoiVector(); // 获取 ROI 向量

    // 缓冲区队列
    std::queue<cv::UMat> frame_buffer;
    const size_t max_buffer_size = 3; // 缓冲区最大帧数

    while (true) {
        auto t_start = std::chrono::steady_clock::now();
        std::vector<std::thread> warp_thread_vect;
        sensor_data_interface_.get_image_vector(image_vector, image_mutex_vector);
        auto t_got_images = std::chrono::steady_clock::now();

        // 记录拼接后的图像的总宽度
        total_offset_x = 0;
        for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
            total_offset_x += roi_vector[i].width;
        }

        for (size_t img_idx = 0; img_idx < sensor_data_interface_.num_img_; ++img_idx) {
            warp_thread_vect.emplace_back(
                &ImageStitcher::WarpImages,
                &image_stitcher_,
                img_idx,
                20,
                image_vector,
                std::ref(image_mutex_vector),
                std::ref(images_warped_vector),
                std::ref(image_concat_umat_)
            );
        }

        for (auto& warp_thread : warp_thread_vect) {
            warp_thread.join();
        }
        auto t_stitched = std::chrono::steady_clock::now();

        int offset_x = 0; // 当前图像的 x 偏移量（拼接时图像的左上角位置）
        for (size_t img_idx = 0; img_idx < sensor_data_interface_.num_img_; ++img_idx) {
            cv::Mat xmap = image_stitcher_.getFinalXMap(img_idx).getMat(cv::ACCESS_READ);
            cv::Mat ymap = image_stitcher_.getFinalYMap(img_idx).getMat(cv::ACCESS_READ);

            float new_x = xmap.at<float>(y, x);
            float new_y = ymap.at<float>(y, x);

            // 计算该点在拼接图像中的位置
            int final_x = new_x + offset_x;
            int final_y = new_y;

            // 在拼接后的图像上标记该点并添加文本
            cv::circle(image_concat_umat_, cv::Point(final_x, final_y), 5, cv::Scalar(0, 255, 0), -1); // 用绿色圆圈标记点
            cv::putText(image_concat_umat_, "Position of steel billet", cv::Point(final_x + 10, final_y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // 更新下一个图像的偏移量
            offset_x += roi_vector[img_idx].width;
        }

        // 将拼接后的图像存入缓冲区
        frame_buffer.push(image_concat_umat_);

        // 如果缓冲区已满，等待推送并清空缓冲区
        if (frame_buffer.size() > max_buffer_size) {
            while (!frame_buffer.empty()) {
                PushFrame(frame_buffer.front());
                frame_buffer.pop();
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 控制帧率，确保60fps
            }
        }

        frame_idx++;

        auto t_pushed = std::chrono::steady_clock::now();
        std::cout << "Total: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_pushed - t_start).count()
                  << " ms" << std::endl;
    }
}

int main() {
    App app;
    app.run_stitching();
}
