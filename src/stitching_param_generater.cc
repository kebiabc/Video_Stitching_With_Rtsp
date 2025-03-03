#include "stitching_param_generater.h"
#include <iostream>
#include <fstream>
#include <string>

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

StitchingParamGenerator::StitchingParamGenerator(
    const std::vector<cv::Mat>& image_vector) {

  std::cout << "[StitchingParamGenerator] Initializing..." << std::endl;

  num_img_ = image_vector.size();

  image_vector_ = image_vector;
  mask_vector_ = std::vector<cv::UMat>(num_img_);
  mask_warped_vector_ = std::vector<cv::UMat>(num_img_);
  image_size_vector_ = std::vector<cv::Size>(num_img_);
  image_warped_size_vector_ = std::vector<cv::Size>(num_img_);
  reproj_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  reproj_ymap_vector_ = std::vector<cv::UMat>(num_img_);
  camera_params_vector_ =
      std::vector<cv::detail::CameraParams>(camera_params_vector_);

  projected_image_roi_refined_vect_ = std::vector<cv::Rect>(num_img_);

  for (size_t img_idx = 0; img_idx < num_img_; img_idx++) {
    image_size_vector_[img_idx] = image_vector_[img_idx].size();
  }

  std::vector<cv::UMat> undist_xmap_vector;
  std::vector<cv::UMat> undist_ymap_vector;

  InitUndistortMap();

  for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
    cv::remap(image_vector_[img_idx],
              image_vector_[img_idx],
              undist_xmap_vector_[img_idx],
              undist_ymap_vector_[img_idx],
              cv::INTER_LINEAR);
  }

  InitCameraParam();
  InitWarper();

  std::cout << "[StitchingParamGenerator] Initialized." << std::endl;
}

void StitchingParamGenerator::InitCameraParam() {
  std::vector<ImageFeatures> features(num_img_);
  std::vector<Point2f> manual_points;

  for (int i = 0; i < num_img_; ++i) {
    std::cout << "Please select keypoints for image #" << i + 1 << "\n";
    Mat display_img = image_vector_[i].clone();

    std::vector<Point2f> points;
    setMouseCallback("Select Points", [](int event, int x, int y, int, void* userdata) {
      if (event == EVENT_LBUTTONDOWN) {
        auto* pts = static_cast<std::vector<Point2f>*>(userdata);
        pts->emplace_back(x, y);
        std::cout << "Point selected: " << x << ", " << y << std::endl;
        Mat* img = static_cast<Mat*>(pts->data());
        circle(*img, Point(x, y), 5, Scalar(0, 0, 255), -1);
        imshow("Select Points", *img);
      }
    }, &points);

    imshow("Select Points", display_img);
    while (true) {
      char key = (char)waitKey(1);
      if (key == 13) { // Enter key to confirm
        break;
      }
    }
    destroyWindow("Select Points");

    manual_points.insert(manual_points.end(), points.begin(), points.end());
    features[i].keypoints.clear();
    for (auto& pt : points) {
      features[i].keypoints.emplace_back(KeyPoint(pt, 1.0f));
    }

    features[i].img_idx = i;
    LOGLN("Manual Features in image #" << i + 1 << ": " << features[i].keypoints.size());
  }

  LOGLN("Pairwise matching");
  std::vector<MatchesInfo> pairwise_matches;
  Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(false, 0.3f);
  (*matcher)(features, pairwise_matches);
  matcher->collectGarbage();

  Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();
  if (!(*estimator)(features, pairwise_matches, camera_params_vector_)) {
    std::cout << "Homography estimation failed.\n";
    assert(false);
  }

  for (auto& i : camera_params_vector_) {
    Mat R;
    i.R.convertTo(R, CV_32F);
    i.R = R;
  }

  Ptr<detail::BundleAdjusterBase> adjuster = makePtr<detail::BundleAdjusterReproj>();
  adjuster->setConfThresh(1.0);
  Mat_<uchar> refine_mask = Mat::eye(3, 3, CV_8U);
  adjuster->setRefinementMask(refine_mask);

  if (!(*adjuster)(features, pairwise_matches, camera_params_vector_)) {
    std::cout << "Camera parameters adjusting failed.\n";
    assert(false);
  }
}


void StitchingParamGenerator::InitWarper() {

  std::vector<double> focals;
  float median_focal_length;
  reproj_xmap_vector_ = std::vector<UMat>(num_img_);

  for (size_t i = 0; i < camera_params_vector_.size(); ++i) {
    LOGLN("Camera #" << i + 1 << ":\nK:\n" << camera_params_vector_[i].K()
                     << "\nR:\n" << camera_params_vector_[i].R);
    focals.push_back(camera_params_vector_[i].focal);
  }
  sort(focals.begin(), focals.end());
  if (focals.size() % 2 == 1)
    median_focal_length = static_cast<float>(focals[focals.size() / 2]);
  else
    median_focal_length =
        static_cast<float>(focals[focals.size() / 2 - 1] +
            focals[focals.size() / 2]) * 0.5f;

  Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
  if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0) {
    if (warp_type == "plane")
      warper_creator = makePtr<cv::PlaneWarperGpu>();
    else if (warp_type == "cylindrical")
      warper_creator = makePtr<cv::CylindricalWarperGpu>();
    else if (warp_type == "spherical")
      warper_creator = makePtr<cv::SphericalWarperGpu>();
  } else
#endif
  {
    if (warp_type == "plane")
      warper_creator = makePtr<cv::PlaneWarper>();
    else if (warp_type == "affine")
      warper_creator = makePtr<cv::AffineWarper>();
    else if (warp_type == "cylindrical")
      warper_creator = makePtr<cv::CylindricalWarper>();
    else if (warp_type == "spherical")
      warper_creator = makePtr<cv::SphericalWarper>();
    else if (warp_type == "fisheye")
      warper_creator = makePtr<cv::FisheyeWarper>();
    else if (warp_type == "stereographic")
      warper_creator = makePtr<cv::StereographicWarper>();
    else if (warp_type == "compressedPlaneA2B1")
      warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
    else if (warp_type == "compressedPlaneA1.5B1")
      warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
    else if (warp_type == "compressedPlanePortraitA2B1")
      warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
    else if (warp_type == "compressedPlanePortraitA1.5B1")
      warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
    else if (warp_type == "paniniA2B1")
      warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
    else if (warp_type == "paniniA1.5B1")
      warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
    else if (warp_type == "paniniPortraitA2B1")
      warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
    else if (warp_type == "paniniPortraitA1.5B1")
      warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
    else if (warp_type == "mercator")
      warper_creator = makePtr<cv::MercatorWarper>();
    else if (warp_type == "transverseMercator")
      warper_creator = makePtr<cv::TransverseMercatorWarper>();
  }
  if (!warper_creator) {
    std::cout << "Can't create the following warper '" << warp_type << "'\n";
    assert(false);
  }
  rotation_warper_ =
      warper_creator->create(static_cast<float>(median_focal_length));
  LOGLN("warped_image_scale: " << median_focal_length);

  std::vector<cv::Point> image_point_vect(num_img_);

  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    Mat_<float> K;
    camera_params_vector_[img_idx].K().convertTo(K, CV_32F);
    Rect rect = rotation_warper_->buildMaps(image_size_vector_[img_idx], K,
                                       camera_params_vector_[img_idx].R,
                                       reproj_xmap_vector_[img_idx],
                                       reproj_ymap_vector_[img_idx]);
    Point point(rect.x, rect.y);
    image_point_vect[img_idx] = point;
  }


  // Prepare images masks
  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    mask_vector_[img_idx].create(image_vector_[img_idx].size(), CV_8U);
    mask_vector_[img_idx].setTo(Scalar::all(255));
    remap(mask_vector_[img_idx],
          mask_warped_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          INTER_NEAREST);
    image_warped_size_vector_[img_idx] = mask_warped_vector_[img_idx].size();
  }

  timelapser_ = Timelapser::createDefault(timelapse_type);
  blender_ = Blender::createDefault(Blender::NO);
  timelapser_->initialize(image_point_vect, image_size_vector_);
  blender_->prepare(image_point_vect, image_size_vector_);

  std::vector<cv::Rect> projected_image_roi_vect = std::vector<cv::Rect>(num_img_);

  // Update corners and sizes
  // TODO(duchengyao): Figure out what bias means.
  Point roi_tl_bias(999999, 999999);
  for (int i = 0; i < num_img_; ++i) {
    // Update corner and size
    Size sz = image_vector_[i].size();
    Mat K;
    camera_params_vector_[i].K().convertTo(K, CV_32F);
    Rect roi = rotation_warper_->warpRoi(sz, K, camera_params_vector_[i].R);
    std::cout << "roi" << roi << std::endl;
    roi_tl_bias.x = min(roi.tl().x, roi_tl_bias.x);
    roi_tl_bias.y = min(roi.tl().y, roi_tl_bias.y);
    projected_image_roi_vect[i] = roi;
  }
  full_image_size_ = Point(0, 0);
  Point y_range = Point(-9999999, 999999);
  for (int i = 0; i < num_img_; ++i) {
    projected_image_roi_vect[i] -= roi_tl_bias;
    Point tl = projected_image_roi_vect[i].tl();
    Point br = projected_image_roi_vect[i].br();

    full_image_size_.x = max(br.x, full_image_size_.x);
    full_image_size_.y = max(br.y, full_image_size_.y);
    y_range.x = max(y_range.x, tl.y);
    y_range.y = min(y_range.y, br.y);
  }
  for (int i = 0; i < num_img_; ++i) {
    Rect rect = projected_image_roi_vect[i];
    rect.height =
        rect.height - (rect.br().y - y_range.y + y_range.x - rect.tl().y);
    rect.y = y_range.x - rect.y;
    projected_image_roi_vect[i] = rect;
    projected_image_roi_refined_vect_[i] = rect;
  }

  for (int i = 0; i < num_img_ - 1; ++i) {

    Rect rect_left = projected_image_roi_refined_vect_[i];
    int offset = (projected_image_roi_vect[i].br().x -
        projected_image_roi_vect[i + 1].tl().x) / 2;
    rect_left.width -= offset;
    Rect rect_right = projected_image_roi_vect[i + 1];
    rect_right.width -= offset;
    rect_right.x = offset;
    projected_image_roi_refined_vect_[i] = rect_left;
    projected_image_roi_refined_vect_[i + 1] = rect_right;
  }
}

void StitchingParamGenerator::InitUndistortMap() {
  std::vector<double> cam_focal_vector(num_img_);

  std::vector<cv::UMat> r_vector(num_img_);
  std::vector<cv::UMat> k_vector(num_img_);
  std::vector<std::vector<double>> d_vector(num_img_);
  cv::Size resolution;

  undist_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  undist_ymap_vector_ = std::vector<cv::UMat>(num_img_);

  for (size_t i = 0; i < num_img_; i++) {
    cv::FileStorage fs_read(
        "../params/old/camchain_" + std::to_string(i) + ".yaml",

        cv::FileStorage::READ);
    if (!fs_read.isOpened()) {
      fprintf(stderr, "%s:%d:loadParams falied. 'camera.yml' does not exist\n", __FILE__, __LINE__);
      return;
    }
    cv::Mat R, K;
    fs_read["KMat"] >> K;
    K.copyTo(k_vector[i]);
    fs_read["D"] >> d_vector[i];
    fs_read["RMat"] >> R;
    R.copyTo(r_vector[i]);
    fs_read["focal"] >> cam_focal_vector[i];
    fs_read["resolution"] >> resolution;
  }

  for (size_t i = 0; i < num_img_; i++) {
    cv::UMat K;
    cv::UMat R;
    cv::UMat NONE;
    k_vector[i].convertTo(K, CV_32F);
    cv::UMat::eye(3, 3, CV_32F).convertTo(R, CV_32F);

    cv::initUndistortRectifyMap(
        K, d_vector[i], R, NONE, resolution,
        CV_32FC1, undist_xmap_vector_[i], undist_ymap_vector_[i]);
  }
}

void StitchingParamGenerator::GetReprojParams(
    std::vector<cv::UMat>& undist_xmap_vector,
    std::vector<cv::UMat>& undist_ymap_vector,
    std::vector<cv::UMat>& reproj_xmap_vector,
    std::vector<cv::UMat>& reproj_ymap_vector,
    std::vector<cv::Rect>& projected_image_roi_refined_vect) {

  undist_xmap_vector = undist_xmap_vector_;
  undist_ymap_vector = undist_ymap_vector_;
  reproj_xmap_vector = reproj_xmap_vector_;
  reproj_ymap_vector = reproj_ymap_vector_;
  projected_image_roi_refined_vect = projected_image_roi_refined_vect_;
  std::cout << "[GetReprojParams] projected_image_roi_vect_refined: " << std::endl;

  size_t i = 0;
  for (auto& roi : projected_image_roi_refined_vect) {
    std::cout << "[GetReprojParams] roi [" << i << ": "
              << roi.width << "x"
              << roi.height << " from ("
              << roi.x << ", "
              << roi.y << ")]" << std::endl;
    i++;
    if (roi.width < 0 || roi.height < 0 || roi.x < 0 || roi.y < 0) {
      std::cout << "StitchingParamGenerator did not find a suitable feature point under the current parameters, "
                << "resulting in an incorrect ROI. "
                << "Please use \"opencv/stitching_detailed\" to find the correct parameters. "
                << "(see https://docs.opencv.org/4.8.0/d8/d19/tutorial_stitcher.html)" << std::endl;
      assert (false);
    }
  }
}
