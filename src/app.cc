#include "app.h"
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <zmq.hpp>

#include "image_stitcher.h"
#include "stitching_param_generater.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/timestamp.h>
}

// 全局变量，用于存储目标坐标
std::mutex target_mutex;
double target_x = -1, target_y = -1;
int target_camera_id = -1;
bool target_updated = false;

void zmq_listener() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);  // 使用 SUB 套接字
    socket.connect("tcp://localhost:5555");  // 连接至发送端地址
    socket.set(zmq::sockopt::subscribe, "");

    while (true) {
        zmq::message_t msg;
        if (socket.recv(msg, zmq::recv_flags::none)) { // 阻塞接收消息
            std::string message(static_cast<char*>(msg.data()), msg.size());
            int cam_id;
            double x, y;
            // 使用逗号分隔解析
            if (sscanf(message.c_str(), "%d,%lf,%lf", &cam_id, &x, &y) == 3) {
                std::lock_guard<std::mutex> lock(target_mutex);
                target_camera_id = cam_id;
                target_x = x;
                target_y = y;
                target_updated = true;
                std::cout << "接收目标坐标: " << cam_id << ", " << x << ", " << y << std::endl;
            } else {
                std::cerr << "解析消息失败: " << message << std::endl;
            }
        } else {
            std::cerr << "接收消息失败" << std::endl;
        }
    }
}
// 构造函数
App::App() : is_running_(false), total_cols_(0) {
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
        codec_ctx->time_base = {1, 30};
        codec_ctx->framerate = {30, 1};
        codec_ctx->bit_rate = 20 * 1024 * 1024;
        codec_ctx->gop_size = 5;


        // 设置快速编码参数
        AVDictionary* codec_options = nullptr;
        av_dict_set(&codec_options, "preset", "ultrafast", 0);
        av_dict_set(&codec_options, "tune", "zerolatency", 0);
        
        if (avcodec_open2(codec_ctx, codec, &codec_options) < 0) {
            std::cerr << "Failed to open codec" << std::endl;
            av_dict_free(&codec_options);
            return;
        }
        av_dict_free(&codec_options);

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

void App::StitchFrames() {
    std::vector<cv::UMat> image_vector(sensor_data_interface_.num_img_);
    std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
    std::vector<cv::UMat> images_warped_vector(sensor_data_interface_.num_img_);
    std::thread record_videos_thread(
        &SensorDataInterface::RecordVideos,
        &sensor_data_interface_
    );

    size_t frame_idx = 0;

    // 额外存储上一次的目标坐标
    int last_target_camera_id = -1;
    double last_target_x = -1, last_target_y = -1;
    bool last_target_valid = false;

    while (is_running_) {
        auto t_start = std::chrono::steady_clock::now();
        std::vector<std::thread> warp_thread_vect;
        sensor_data_interface_.get_image_vector(image_vector, image_mutex_vector);
        auto t_got_images = std::chrono::steady_clock::now();

        // 检查是否有新的目标坐标
        {
            std::lock_guard<std::mutex> lock(target_mutex);
            if (target_updated) {
                last_target_camera_id = target_camera_id;
                last_target_x = target_x;
                last_target_y = target_y;
                last_target_valid = true;  // 标记坐标有效
                target_updated = false;
            }
        }

        // 拼接图像
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

        // 仅在坐标有效时标记目标位置
        if (last_target_valid && last_target_camera_id >= 0 && last_target_camera_id < sensor_data_interface_.num_img_) {
            const auto& roi_vector = image_stitcher_.getRoiVector();
            int total_offset_x = 0;
            for (size_t img_idx = 0; img_idx < last_target_camera_id; ++img_idx) {
                total_offset_x += roi_vector[img_idx].width;
            }

            cv::Mat xmap = image_stitcher_.getFinalXMap(last_target_camera_id).getMat(cv::ACCESS_READ);
            cv::Mat ymap = image_stitcher_.getFinalYMap(last_target_camera_id).getMat(cv::ACCESS_READ);

            // 查找目标在拼接图中的位置
            float new_x = xmap.at<float>(last_target_y, last_target_x);
            float new_y = ymap.at<float>(last_target_y, last_target_x);

            int final_x = new_x + total_offset_x;
            int final_y = new_y;

            // 在拼接后的图像上标记目标位置
            cv::circle(image_concat_umat_, cv::Point(final_x, final_y), 15, cv::Scalar(0, 255, 0), -1); // 用绿色圆圈标记点
            std::cout << "Target:(" << final_x << ", " << final_y << ")" << std::endl;
        }

        // 将拼接后的图像存入缓冲区
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            frame_buffer_.push(image_concat_umat_);
            buffer_cond_.notify_all();  // 唤醒推流线程
        }

        frame_idx++;

        auto t_pushed = std::chrono::steady_clock::now();
        std::cout << "Image capture: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_got_images - t_start).count()
                  << " ms, Stitching: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_stitched - t_got_images).count()
                  << " ms, Push: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_pushed - t_stitched).count()
                  << " ms, Total: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_pushed - t_start).count()
                  << " ms" << std::endl;
    }
}

// 推流图像帧
void App::StreamFrames() {
    while (is_running_) {
        cv::UMat frame_to_push;
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_cond_.wait(lock, [this] { return !frame_buffer_.empty() || !is_running_; });

            if (!frame_buffer_.empty()) {
                frame_to_push = frame_buffer_.front();
                frame_buffer_.pop();
            }
        }

        if (!frame_to_push.empty()) {
            PushFrame(frame_to_push);
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 控制推流速率，确保30fps
    }
}

// 主运行函数
[[noreturn]] void App::run_stitching() {
    is_running_ = true;

    // 启动 ZeroMQ 监听线程
    std::thread zmq_thread(zmq_listener);

    // 启动拼接和推流线程
    stitching_thread_ = std::thread(&App::StitchFrames, this);
    streaming_thread_ = std::thread(&App::StreamFrames, this);

    // 等待两个线程结束
    stitching_thread_.join();
    streaming_thread_.join();

    // 等待 ZeroMQ 线程结束
    zmq_thread.join();
}

int main() {
    App app;
    app.run_stitching();
}
