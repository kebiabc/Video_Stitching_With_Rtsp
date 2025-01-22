#include "app.h"
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
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
    std::vector<cv::UMat> first_image_vector(sensor_data_interface_.num_img_);
    std::vector<cv::Mat> first_mat_vector(sensor_data_interface_.num_img_);
    std::vector<cv::UMat> reproj_xmap_vector, undist_xmap_vector;
    std::vector<cv::UMat> undist_ymap_vector, reproj_ymap_vector;
    std::vector<cv::Rect> image_roi_vect;
    std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);

    sensor_data_interface_.get_image_vector(first_image_vector, image_mutex_vector);
    for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
        first_image_vector[i].copyTo(first_mat_vector[i]);
    }

    StitchingParamGenerator stitching_param_generator(first_mat_vector);
    stitching_param_generator.GetReprojParams(
        undist_xmap_vector, undist_ymap_vector,
        reproj_xmap_vector, reproj_ymap_vector, image_roi_vect
    );

    image_stitcher_.SetParams(
        100, undist_xmap_vector, undist_ymap_vector,
        reproj_xmap_vector, reproj_ymap_vector, image_roi_vect
    );

    total_cols_ = 0;
    for (const auto& roi : image_roi_vect) {
        total_cols_ += roi.width;
    }
    image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

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

    while (true) {
        auto t_start = std::chrono::steady_clock::now();

        sensor_data_interface_.get_image_vector(image_vector, image_mutex_vector);
        auto t_got_images = std::chrono::steady_clock::now();

        std::vector<std::thread> warp_thread_vect;
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

        PushFrame(image_concat_umat_);
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

int main() {
    App app;
    app.run_stitching();
}
