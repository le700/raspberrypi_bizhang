#!/usr/bin/python3
# coding=utf8
import os
import random
import sys
import time
import math
import threading
import numpy as np
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.Sonar as Sonar
import hiwonder.ActionGroupControl as AGC
from collections import deque

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# YOLO相关配置 (使用OpenCV DNN + YOLOv3-tiny)
YOLO_AVAILABLE = False
YOLO_NET = None
YOLO_OUTPUT_LAYERS = []
try:
    # 只需要OpenCV，不需要PyTorch
    YOLO_AVAILABLE = True
except Exception:
    pass

debug=False

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

board = rrc.Board()
ctl = Controller(board)


def hand_up():
    ctl.set_bus_servo_pulse(8, 330, 1000)
    time.sleep(0.3)
    ctl.set_bus_servo_pulse(7, 860, 1000)
    ctl.set_bus_servo_pulse(6, 860, 1000)
    time.sleep(1)


def hand_down():
    ctl.set_bus_servo_pulse(7, 800, 1000)
    ctl.set_bus_servo_pulse(6, 575, 1000)
    time.sleep(0.3)
    ctl.set_bus_servo_pulse(8, 725, 1000)
    time.sleep(1)


# 状态控制变量
current_step = 1
obstacle_count = 0  
distance = 99999  # 单位：mm
goforward = 0
# 避障触发锁（防止重复触发）
obstacle_lock = threading.Lock()
# 强制停止巡线标志
stop_patrol = False  
# 避障进行中标志（防止避障时误触发）
avoiding_obstacle = False  

# 跌倒检测相关变量
fall_recovery_in_progress = False
screen_black = False
ULTRASONIC_FALL_THR = 50
BLACK_SCREEN_THR = 150

# 线程锁与视觉变量
distance_lock = threading.Lock()
img_lock = threading.Lock()
yolo_lock = threading.Lock()
IMG_CENTER_X = 200
img_centerx = IMG_CENTER_X
MIN_CONTOUR_AREA = 100

# YOLO相关变量
detected_objects = []
object_confidence = 0.0
object_position = 'none'  # 'left', 'center', 'right', 'none'
USE_YOLO = False  # 可配置是否启用YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_FRAME_SKIP = 3  # 跳帧检测以提升性能
yolo_frame_count = 0
IMG_WIDTH = 640  # 摄像头分辨率宽度

# YOLO检测函数 - 只检测障碍物方位，不估算距离
def yolo_detect(frame):
    global detected_objects, object_confidence, object_position, YOLO_NET, YOLO_OUTPUT_LAYERS
    if not YOLO_AVAILABLE or not USE_YOLO or YOLO_NET is None:
        return []
    
    try:
        height, width = frame.shape[:2]
        
        # 构建输入blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        YOLO_NET.setInput(blob)
        
        # 前向传播
        layer_outputs = YOLO_NET.forward(YOLO_OUTPUT_LAYERS)
        
        objects = []
        boxes = []
        confidences = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > YOLO_CONFIDENCE_THRESHOLD:
                    # 计算边界框坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    objects.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': confidence
                    })
        
        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONFIDENCE_THRESHOLD, 0.4)
        
        final_objects = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            final_objects.append(objects[i])
        
        # 更新全局变量 - 只记录位置
        with yolo_lock:
            detected_objects = final_objects.copy()
            if final_objects:
                # 取最靠近中心且置信度最高的对象
                best_obj = max(final_objects, key=lambda obj: obj['confidence'])
                object_confidence = best_obj['confidence']
                
                # 判断方位：左(0-213)、中(213-426)、右(426-640)
                if best_obj['center_x'] < IMG_WIDTH / 3:
                    object_position = 'left'
                elif best_obj['center_x'] > IMG_WIDTH * 2 / 3:
                    object_position = 'right'
                else:
                    object_position = 'center'
            else:
                object_confidence = 0.0
                object_position = 'none'
        
        return final_objects
    except Exception as e:
        return []

# 视觉线程（单线巡线逻辑 + YOLO检测）

def vision_loop():
    global img_centerx, screen_black, yolo_frame_count
    if not CV2_AVAILABLE:
        while True:
            with img_lock:
                img_centerx = IMG_CENTER_X
                screen_black = False
            time.sleep(0.05)
        return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("摄像头打开失败，使用默认值")
        while True:
            with img_lock:
                img_centerx = IMG_CENTER_X
                screen_black = False
            time.sleep(0.05)
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            roi = frame[int(h * 0.5):h, 0:w]
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            lower = np.array([180, 0, 0])
            upper = np.array([255, 255, 255])
            mask = cv2.inRange(lab, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            white_pixels = cv2.countNonZero(mask)
            with img_lock:
                screen_black = white_pixels < BLACK_SCREEN_THR
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA]
            with img_lock:
                if len(valid_contours) >= 1:
                    valid_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
                    largest_contour = valid_contours[0]
                    m = cv2.moments(largest_contour)
                    if m['m00'] > 0:
                        img_centerx = int(m['m10'] / m['m00'])
                    else:
                        img_centerx = -1
                else:
                    img_centerx = -1
            
            # YOLO检测（跳帧处理）
            if USE_YOLO and YOLO_AVAILABLE:
                yolo_frame_count += 1
                if yolo_frame_count >= YOLO_FRAME_SKIP:
                    yolo_detect(frame)
                    yolo_frame_count = 0
            
            time.sleep(0.02)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def move():
    global current_step, obstacle_count, distance, goforward, fall_recovery_in_progress, stop_patrol, avoiding_obstacle

    DIST_OBSTACLE_MM = 250
    LINE_OFFSET_THR = 80
    LARGE_OFFSET_THR = 150
    TURN_ADJUST_TIME = 0.12
    SEARCH_TIME = 0.2

    while True:
        with distance_lock:
            local_distance = distance
        with img_lock:
            local_imgx = img_centerx
            local_screen_black = screen_black
        with yolo_lock:
            local_yolo_position = object_position
            local_yolo_conf = object_confidence
        
        if local_screen_black and local_distance < ULTRASONIC_FALL_THR:
            if not fall_recovery_in_progress:
                print("检测到跌倒，开始自动起立...")
                fall_recovery_in_progress = True
                try:
                    AGC.runActionGroup('stand_up_front')
                    print("起立动作完成")
                except Exception as e:
                    print(f"起立失败: {e}")
                fall_recovery_in_progress = False
                current_step = 1
                stop_patrol = False
                goforward = 0
            time.sleep(0.1)
            continue

        with obstacle_lock:
            # 只有不在避障过程中才检测新障碍物
            if not avoiding_obstacle and local_distance > 0 and local_distance <= DIST_OBSTACLE_MM:
                avoiding_obstacle = True
                stop_patrol = True
                obstacle_count += 1
                print(f"🚨 触发避障！超声波={local_distance}mm，YOLO检测到障碍物在'{local_yolo_position}'方位")
                print("→ 进入避障流程，临时禁用障碍物检测...")
                
                # 根据YOLO检测到的方位选择避障方向
                if USE_YOLO and local_yolo_conf > 0.3 and local_yolo_position != 'none':
                    # 障碍物在左边 → 从右边绕（step 2）
                    # 障碍物在右边 → 从左边绕（step 3）
                    # 障碍物在中间 → 交替选择
                    if local_yolo_position == 'left':
                        current_step = 2  # 从右边绕
                        print("→ YOLO检测障碍物在左侧，选择从右侧绕行")
                    elif local_yolo_position == 'right':
                        current_step = 3  # 从左边绕
                        print("→ YOLO检测障碍物在右侧，选择从左侧绕行")
                    else:  # center
                        current_step = 2 if obstacle_count % 2 == 1 else 3
                        print(f"→ YOLO检测障碍物在中间，轮流选择避障方向")
                else:
                    # YOLO未启用或未检测到，使用轮流策略
                    current_step = 2 if obstacle_count % 2 == 1 else 3
                    print(f"→ 使用轮流避障策略")
                time.sleep(0.3)
            else:
                stop_patrol = False

        if stop_patrol and current_step == 1:
            time.sleep(0.1)
            continue

        if current_step == 1:
            if local_imgx == -1:
                print("未检测到线，缓慢搜索")
                AGC.runActionGroup('turn_left')
                time.sleep(SEARCH_TIME)
            else:
                offset = local_imgx - IMG_CENTER_X
                print(f"巡线: 线心={local_imgx}, 偏移={offset}, 距离={local_distance}mm")
                if abs(offset) <= LINE_OFFSET_THR:
                    AGC.runActionGroup('go_forward_fast')
                    print("直行（快速）")
                    goforward += 1
                elif offset > LARGE_OFFSET_THR:
                    AGC.runActionGroup('turn_right')
                    time.sleep(TURN_ADJUST_TIME)
                    AGC.runActionGroup('turn_right')
                    print("右急调")
                    goforward = 0
                elif offset > LINE_OFFSET_THR:
                    AGC.runActionGroup('turn_right')
                    print("右微调")
                    goforward = 0
                elif offset < -LARGE_OFFSET_THR:
                    AGC.runActionGroup('turn_left')
                    time.sleep(TURN_ADJUST_TIME)
                    AGC.runActionGroup('turn_left')
                    print("左急调")
                    goforward = 0
                else:
                    AGC.runActionGroup('turn_left')
                    print("左微调")
                    goforward = 0
            time.sleep(0.08)

        # 5. 避障流程1：右移→前进→回正（执行实际动作）
        elif current_step == 2:
            print("👉 执行避障流程2：右移避障 → 远离 → 左转回正")
            print("→ 避障过程中，YOLO持续监控周围环境...")
            # 强制停止当前动作
            time.sleep(0.2)
            # 步骤1：右移避开障碍物
            for _ in range(8):  
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            # 步骤2：向前远离障碍物（注意：可能遇到新障碍物）
            for i in range(10): 
                AGC.runActionGroup('zhixing4')
                # 避障过程中YOLO持续监控
                if USE_YOLO:
                    with yolo_lock:
                        if object_position == 'center' and object_confidence > 0.5:
                            print(f"⚠️ YOLO检测到正前方有新障碍物！增加观察时间...")
                            time.sleep(0.5)  # 增加观察时间
                time.sleep(0.18)
            # 步骤3：左转回正
            for _ in range(7):  
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            # 避障完成后短暂观察
            print("✅ 右移避障动作完成，短暂观察周围...")
            time.sleep(0.3)
            # 重置状态，回到巡线
            current_step = 1
            stop_patrol = False
            goforward = 0
            avoiding_obstacle = False  # 避障完成，重新启用障碍物检测
            print("✅ 确认安全，回到巡线状态")

        # 6. 避障流程2：左移→前进→回正（执行实际动作）
        elif current_step == 3:
            print("👉 执行避障流程3：左移避障 → 远离 → 右转回正")
            print("→ 避障过程中，YOLO持续监控周围环境...")
            # 强制停止当前动作
            time.sleep(0.2)
            # 步骤1：左移避开障碍物
            for _ in range(9):  
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            # 步骤2：向前远离障碍物（注意：可能遇到新障碍物）
            for i in range(10): 
                AGC.runActionGroup('go_forward_fast')
                # 避障过程中YOLO持续监控
                if USE_YOLO:
                    with yolo_lock:
                        if object_position == 'center' and object_confidence > 0.5:
                            print(f"⚠️ YOLO检测到正前方有新障碍物！增加观察时间...")
                            time.sleep(0.5)  # 增加观察时间
                time.sleep(0.18)
            # 步骤3：右转回正
            for _ in range(8):  
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            # 避障完成后短暂观察
            print("✅ 左移避障动作完成，短暂观察周围...")
            time.sleep(0.3)
            # 重置状态，回到巡线
            current_step = 1
            stop_patrol = False
            goforward = 0
            avoiding_obstacle = False  # 避障完成，重新启用障碍物检测
            print("✅ 确认安全，回到巡线状态")

        else:
            time.sleep(0.1)


# 启动线程
th_move = threading.Thread(target=move)
th_move.daemon = True
th_move.start()

th_vision = threading.Thread(target=vision_loop)
th_vision.daemon = True
th_vision.start()

if __name__ == "__main__":
    distance_list = []
    s = Sonar.Sonar()
    s.startSymphony()
    
    # YOLOv3-tiny模型初始化 (OpenCV DNN)
    if YOLO_AVAILABLE and USE_YOLO:
        try:
            print("正在加载YOLOv3-tiny模型...")
            # 加载YOLOv3-tiny网络
            YOLO_NET = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
            # 尝试使用OpenCL加速
            YOLO_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            YOLO_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 树莓派用CPU
            
            # 获取输出层
            layer_names = YOLO_NET.getLayerNames()
            YOLO_OUTPUT_LAYERS = [layer_names[i[0] - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
            print("YOLOv3-tiny模型加载完成！")
        except Exception as e:
            print(f"YOLOv3-tiny模型加载失败: {e}")
            print("请确保 yolov3-tiny.cfg 和 yolov3-tiny.weights 文件在同一目录下")
            print("可以从 https://pjreddie.com/darknet/yolo/ 下载")
            YOLO_AVAILABLE = False
            USE_YOLO = False

    # 初始化：先站立
    AGC.runActionGroup('stand_slow')
    time.sleep(1)
    print(f"程序启动完成，开始巡线+避障 (YOLO: {'启用' if USE_YOLO and YOLO_AVAILABLE else '禁用'})")

    try:
        while True:
            # 读取超声波数据（过滤无效值）
            dist = s.getDistance()
            if 0 < dist < 5000:  # 只保留有效测距值
                distance_list.append(dist)
            # 取6帧平均，降低噪声
            if len(distance_list) >= 6:
                with distance_lock:
                    distance = int(round(np.mean(np.array(distance_list))))
                    if YOLO_AVAILABLE and USE_YOLO:
                        with yolo_lock:
                            yolo_pos = object_position
                            yolo_conf = object_confidence
                        print(f"当前测距: {distance} mm, YOLO检测: {yolo_pos} (置信度: {yolo_conf:.2f})")
                    else:
                        print(f"当前测距: {distance} mm")
                    distance_list = []
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n程序中断，停止所有动作")
        stop_patrol = True
        if CV2_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
