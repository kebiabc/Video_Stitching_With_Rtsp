cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
project(image-stitching)
set(cmake_c_flags_debug "-g -Wall")

find_package(OpenCV 4 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(AVDEVICE REQUIRED libavdevice)
pkg_check_modules(SWSCALE REQUIRED libswscale)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${AVDEVICE_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${SWRESAMPLE_INCLUDE_DIRS}
    include
)

add_executable(${PROJECT_NAME}
    src/app.cc
    include/app.h
    src/sensor_data_interface.cc
    include/sensor_data_interface.h
    src/stitching_param_generater.cc
    include/stitching_param_generater.h
    src/image_stitcher.cc
    include/image_stitcher.h
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${AVDEVICE_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    -lpthread
    -lm
)
