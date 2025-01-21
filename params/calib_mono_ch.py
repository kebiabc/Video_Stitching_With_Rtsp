import cv2
import numpy as np
import yaml
import os

def save_to_yaml(file_path, calibration_results):
    # 将标定结果转化为字典格式
    data = {
        'cam_overlaps': [],
        'camera_model': 'pinhole',
        'focal': calibration_results['focal'],
        'D': calibration_results['distortion'].tolist(),  # 转换为列表
        'KMat': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': calibration_results['K'].tolist()  # 转换为列表
        },
        'distortion_model': 'radtan',
        'RMat': {
            'rows': 3,
            'cols': 3,
            'dt': 'f',
            'data': calibration_results['R'].tolist()  # 转换为列表
        },
        'EYEMat': {
            'rows': 3,
            'cols': 3,
            'dt': 'u',
            'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]  # 固定矩阵
        },
        'K': calibration_results['K'].tolist(),  # 转换为列表
        'R': calibration_results['R'].tolist(),  # 转换为列表
        'resolution': calibration_results['resolution']
    }

    # 保存为YAML文件，按照你的要求格式化括号
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def calibration(img_path, grid_size, square_size):
    # 创建标定板的3D点坐标
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 按实际方格大小缩放

    # 存储所有图片的3D点和2D点
    objpoints = []  # 3D点（世界坐标系下）
    imgpoints = []  # 2D点（像素坐标系下）

    # 自动生成图像路径
    img_num = len(os.listdir(img_path))
    images = [os.path.join(img_path, f"{i}.png") for i in range(1, img_num + 1)]

    # 逐一提取棋盘格角点
    for img_path in images:
        # 加载图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.equalizeHist(img)
        # 查找棋盘格角点
        found, corners = cv2.findChessboardCorners(img, grid_size)

        if found:
            # 对角点进行亚像素级精确化
            corners_subpix = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), 
                                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
        else:
            print(f"未检测到棋盘格角点: {img_path}")

    # 标定单目标定（先计算内参）
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None
    )

    # 计算焦距
    focal_length = mtx[0, 0]

    # 返回标定结果
    return {
        'focal': focal_length,
        'distortion': dist.flatten(),
        'K': mtx,
        'R': rvecs[0],  # 使用第一个旋转向量
        'resolution': img.shape[::-1]  # 图像分辨率
    }

if __name__ == '__main__':
    # 设置标定板的网格大小和每个方格的实际尺寸
    grid_size = (8, 11)  # 棋盘格的行列数
    square_size = 30  # 每个方格的实际尺寸（单位：mm）
    img_path = "dataset/101"  # 图像文件夹路径

    # 执行标定
    calib_results = calibration(img_path, grid_size, square_size)

    # 保存为YAML文件
    save_to_yaml('camera_calib_results.yaml', calib_results)
    print("标定结果已保存为camera_calib_results.yaml")
