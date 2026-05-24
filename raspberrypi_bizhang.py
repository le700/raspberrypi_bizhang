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

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

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
# 新增：避障触发锁（防止重复触发）
obstacle_lock = threading.Lock()
# 新增：强制停止巡线标志
stop_patrol = False  

# 跌倒检测相关变量
fall_recovery_in_progress = False
screen_black = False
ULTRASONIC_FALL_THR = 50
BLACK_SCREEN_THR = 150

# 线程锁与视觉变量
distance_lock = threading.Lock()
img_lock = threading.Lock()
IMG_CENTER_X = 200
img_centerx = IMG_CENTER_X
MIN_CONTOUR_AREA = 100

# 视觉线程（单线巡线逻辑）
def vision_loop():
    global img_centerx, screen_black
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
            time.sleep(0.02)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def move():
    global current_step, obstacle_count, distance, goforward, fall_recovery_in_progress, stop_patrol

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
            if local_distance > 0 and local_distance <= DIST_OBSTACLE_MM:
                stop_patrol = True
                obstacle_count += 1
                print(f"🚨 触发避障！距离={local_distance}mm，切换到避障流程")
                current_step = 2 if obstacle_count % 2 == 1 else 3
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
            # 强制停止当前动作
            time.sleep(0.2)
            # 步骤1：右移避开障碍物
            for _ in range(8):  
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            # 步骤2：向前远离障碍物
            for _ in range(10): 
                AGC.runActionGroup('zhixing4')
                time.sleep(0.18)
            # 步骤3：左转回正
            for _ in range(7):  
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            # 重置状态，回到巡线
            current_step = 1
            stop_patrol = False
            goforward = 0
            print("✅ 右移避障完成，回到巡线状态")

        # 6. 避障流程2：左移→前进→回正（执行实际动作）
        elif current_step == 3:
            print("👉 执行避障流程3：左移避障 → 远离 → 右转回正")
            # 强制停止当前动作
            time.sleep(0.2)
            # 步骤1：左移避开障碍物
            for _ in range(9):  
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            # 步骤2：向前远离障碍物
            for _ in range(10): 
                AGC.runActionGroup('go_forward_fast')
                time.sleep(0.18)
            # 步骤3：右转回正
            for _ in range(8):  
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            # 重置状态，回到巡线
            current_step = 1
            stop_patrol = False
            goforward = 0
            print("✅ 左移避障完成，回到巡线状态")

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

    # 初始化：先站立
    AGC.runActionGroup('stand_slow')
    time.sleep(1)
    print("程序启动完成，开始巡线+避障")

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
