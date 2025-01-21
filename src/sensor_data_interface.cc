// Created by s1nh.org..
// https://zhuanlan.zhihu.com/p/38136322

#include "sensor_data_interface.h"

#include <string>
#include <thread>

SensorDataInterface::SensorDataInterface()
    : max_queue_length_(2) {
  num_img_ = 0;
}

//void SensorDataInterface::InitExampleImages() {
//  std::string img_dir = "../datasets/cam01/pic_raw/";
//  std::vector<std::string> img_file_name = {"0.jpg", "1.jpg", "2.jpg", "3.jpg"};
//
//  num_img_ = img_file_name.size();
//  image_queue_vector_ = std::vector<std::queue<cv::UMat>>(num_img_);
//
//  for (int i = 0; i < img_file_name.size(); ++i) {
//    std::string file_name = img_dir + img_file_name[i];
//    cv::UMat _;
//    cv::imread(file_name, 1).copyTo(_);
//    image_queue_vector_[i].push(_);
//  }
//}

void SensorDataInterface::InitVideoCapture() {
  std::cout << "Initializing video capture..." << std::endl;

  std::string video_dir = "../datasets/air-4cam-mp4/";
  std::vector<std::string> video_file_name = {"00.mp4", "01.mp4", "02.mp4", "03.mp4"};
  // std::string video_dir = "../datasets/dahua/";
  // std::vector<std::string> video_file_name = { "1.mp4", "2.mp4","3.mp4" };
    // std::cout << "Initializing RTSP streams..." << std::endl;

    // // �滻Ϊ RTSP ��ַ
    // std::vector<std::string> rtsp_urls = {
    //     "rtsp://admin:Aa123456@192.168.1.101/cam/realmonitor?channel=1&subtype=0",
    //     "rtsp://admin:Aa123456@192.168.1.102/cam/realmonitor?channel=1&subtype=0",
    //     "rtsp://admin:Aa123456@192.168.1.103/cam/realmonitor?channel=1&subtype=0",
    //     "rtsp://admin:Aa123456@192.168.1.104/cam/realmonitor?channel=1&subtype=0"
    // };


  // num_img_ = rtsp_urls.size();
  num_img_ = video_file_name.size();
  image_queue_vector_ = std::vector<std::queue<cv::UMat>>(num_img_);
  image_queue_mutex_vector_ = std::vector<std::mutex>(num_img_);

  // Init video capture.
 for (int i = 0; i < num_img_; ++i) {
   std::string file_name = video_dir + video_file_name[i];

   cv::VideoCapture capture(file_name);
   if (!capture.isOpened())
     std::cout << "Failed to open capture " << i << std::endl;
   video_capture_vector_.push_back(capture);

   cv::UMat frame;
   capture.read(frame);
   image_queue_vector_[i].push(frame);
 }
 std::cout << "Done. " << num_img_ << " captures initialized." << std::endl;
}

//     // ��ʼ�� RTSP ��
//   for (int i = 0; i < num_img_; ++i) {
//       std::string rtsp_url = rtsp_urls[i];

//       cv::VideoCapture capture(rtsp_url);
//       if (!capture.isOpened()) {
//           std::cout << "Failed to open RTSP stream " << i << std::endl;
//           continue;
//       }
//       video_capture_vector_.push_back(capture);

//       // ��ȡ��һ֡���������
//       cv::UMat frame;
//       if (capture.read(frame)) {
//           image_queue_vector_[i].push(frame);
//       }
//   }

//   std::cout << "Done. " << video_capture_vector_.size() << " RTSP streams initialized." << std::endl;
// }

[[noreturn]] void SensorDataInterface::RecordVideos() {
  size_t frame_idx = 0;
  while (true) {
    for (int i = 0; i < num_img_; ++i) {
      cv::UMat frame;
      video_capture_vector_[i].read(frame);
      if (frame.rows > 0) {
        image_queue_mutex_vector_[i].lock();
        image_queue_vector_[i].push(frame);
        if (image_queue_vector_[i].size() > max_queue_length_) {
          image_queue_vector_[i].pop();
            // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        image_queue_mutex_vector_[i].unlock();
      // } else {
      //     break;
      }
    }
    //std::cout << "[RecordVideos] recorded frame " << frame_idx << "." << std::endl;
    frame_idx++;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

void SensorDataInterface::get_image_vector(
    std::vector<cv::UMat>& image_vector,
    std::vector<std::mutex>& image_mutex_vector) {

  std::cout << "[SensorDataInterface] Getting new images...";
  for (size_t i = 0; i < num_img_; ++i) {
    cv::Mat img_undistort;
    cv::Mat img_cylindrical;


    // while (image_queue_vector_[i].empty()) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // }

    image_queue_mutex_vector_[i].lock();
    image_mutex_vector[i].lock();
    image_vector[i] = image_queue_vector_[i].front();
    // image_queue_vector_[i].pop();
    image_mutex_vector[i].unlock();
    image_queue_mutex_vector_[i].unlock();
  }
  std::cout << " Done." << std::endl;

}
