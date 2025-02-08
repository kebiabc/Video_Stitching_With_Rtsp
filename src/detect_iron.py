import zmq
import cv2
import time
import random

# 初始化 ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)  # PUB-SUB 模式
socket.bind("tcp://localhost:5555")  # 监听端口 5555

print("目标检测启动...")
while True:
    # 模拟目标检测（实际可用 YOLO 之类的模型）
    target_camera_id = random.randint(0,3)
    target_x = random.randint(200, 1800)
    target_y = random.randint(100, 900)

    # 发送目标坐标
    message = f"{target_camera_id},{target_x},{target_y}"
    socket.send_string(message)
    print(f"发送目标坐标: {message}")

    time.sleep(1.5)  # 模拟检测间隔