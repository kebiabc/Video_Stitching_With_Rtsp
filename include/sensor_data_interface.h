#ifndef IMAGE_STITCHING_SENSOR_DATA_INTERFACE_H
#define IMAGE_STITCHING_SENSOR_DATA_INTERFACE_H

#include <mutex>
#include <queue>
#include <vector>

#include <opencv2/opencv.hpp>

class SensorDataInterface {
public:

  SensorDataInterface();

  //void InitExampleImages();

  void InitVideoCapture();

  void get_image_vector(std::vector<cv::UMat>& image_vector,
                        std::vector<std::mutex>& image_mutex_vector);

  [[noreturn]] void RecordVideos();

  size_t num_img_;

private:
  const size_t max_queue_length_;
  std::vector<std::queue<cv::UMat>> image_queue_vector_;
  std::vector<std::mutex> image_queue_mutex_vector_;
  std::vector<cv::VideoCapture> video_capture_vector_;

//    std::vector<RectifyMap> cylindrical_map_vector_;
};

#endif //IMAGE_STITCHING_SENSOR_DATA_INTERFACE_H
