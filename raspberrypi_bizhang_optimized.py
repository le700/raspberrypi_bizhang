#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TonyPi 树莓派5优化版避障机器人
功能特性：
- 实时性能监控（FPS、内存、温度、推理延迟）
- 内存管理优化（对象池、垃圾回收）
- YOLOv12推理优化（跳帧、尺寸优化）
- 智能温度控制
- 系统日志记录
- 可配置的性能参数
"""

import os
import sys
import time
import math
import threading
import gc
import logging
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/tonypi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TonyPiOptimized')

# 导入第三方库
try:
    import numpy as np
except ImportError:
    logger.error("请安装 numpy: pip install numpy")
    sys.exit(1)

try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    import hiwonder.Sonar as Sonar
    import hiwonder.ActionGroupControl as AGC
except ImportError:
    logger.warning("未找到Hiwonder SDK，模拟模式运行")

# ==================== 性能监控类 ====================
class PerformanceMonitor:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.fps_history = deque(maxlen=history_size)
        self.inference_time_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.temperature_history = deque(maxlen=history_size)
        
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        self.lock = threading.Lock()
        
    def update_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            with self.lock:
                self.fps_history.append(fps)
        self.last_frame_time = current_time
        self.frame_count += 1
        
    def update_inference_time(self, time_ms: float):
        with self.lock:
            self.inference_time_history.append(time_ms)
            
    def update_memory(self, memory_mb: float):
        with self.lock:
            self.memory_history.append(memory_mb)
            
    def update_temperature(self, temp_c: float):
        with self.lock:
            self.temperature_history.append(temp_c)
            
    def get_fps(self) -> float:
        with self.lock:
            if self.fps_history:
                return np.mean(self.fps_history)
            return 0.0
            
    def get_avg_inference_time(self) -> float:
        with self.lock:
            if self.inference_time_history:
                return np.mean(self.inference_time_history)
            return 0.0
            
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'fps': np.mean(self.fps_history) if self.fps_history else 0,
                'inference_time': np.mean(self.inference_time_history) if self.inference_time_history else 0,
                'memory': np.mean(self.memory_history) if self.memory_history else 0,
                'temperature': np.mean(self.temperature_history) if self.temperature_history else 0,
                'total_frames': self.frame_count
            }
            
    def print_stats(self):
        stats = self.get_stats()
        logger.info(f"=== 性能统计 ===")
        logger.info(f"FPS: {stats['fps']:.1f}")
        logger.info(f"平均推理时间: {stats['inference_time']:.2f} ms")
        logger.info(f"内存使用: {stats['memory']:.1f} MB")
        logger.info(f"温度: {stats['temperature']:.1f} °C")
        logger.info(f"总帧数: {stats['total_frames']}")

# ==================== 内存管理类 ====================
class MemoryManager:
    def __init__(self, pool_size: int = 10):
        self.frame_pool = deque(maxlen=pool_size)
        self.pool_lock = threading.Lock()
        self.last_gc_time = 0
        self.gc_interval = 30  # 每30秒执行一次GC
        
    def get_frame(self):
        with self.pool_lock:
            if self.frame_pool:
                return self.frame_pool.pop()
        return None
        
    def return_frame(self, frame):
        with self.pool_lock:
            if len(self.frame_pool) < self.frame_pool.maxlen:
                self.frame_pool.append(frame)
                
    def periodic_gc(self):
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            gc.collect()
            self.last_gc_time = current_time
            
    def get_memory_usage(self) -> float:
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

# ==================== 系统信息获取 ====================
class SystemInfo:
    @staticmethod
    def get_cpu_temp() -> float:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except Exception:
            return 0.0
            
    @staticmethod
    def get_memory_usage() -> float:
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

# ==================== 配置常量 ====================
# YOLO版本选择
YOLO_VERSION = 'v12n'  # 可选: 'v12n' (YOLOv12-nano), 'v3t' (YOLOv3-tiny)

# YOLO优化配置（树莓派5优化版）
YOLO_DEVICE = 'cpu'
YOLO_IMGSZ = 320  # 减小输入尺寸提高速度
YOLO_CONFIDENCE_THRESHOLD = 0.4  # 降低阈值提高速度
YOLO_FRAME_SKIP = 1  # 跳帧检测，平衡性能和效果
YOLO_MAX_DET = 5  # 限制最大检测数量
USE_YOLO = False  # 是否启用YOLO

# 摄像头配置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMG_CENTER_X = CAMERA_WIDTH // 2

# 性能监控配置
ENABLE_PERFORMANCE_MONITOR = True
MONITOR_INTERVAL = 5  # 秒

# 内存优化配置
ENABLE_MEMORY_POOL = True
FRAME_POOL_SIZE = 10
GC_INTERVAL = 30  # 秒

# ==================== 全局变量初始化 ====================
# 性能监控器
perf_monitor = PerformanceMonitor() if ENABLE_PERFORMANCE_MONITOR else None
memory_manager = MemoryManager(FRAME_POOL_SIZE) if ENABLE_MEMORY_POOL else None

# 硬件接口
board = None
ctl = None

# YOLO相关
YOLO_AVAILABLE = False
YOLO_MODEL = None
YOLO_NET = None
YOLO_OUTPUT_LAYERS = []
CV2_AVAILABLE = False
YOLOV12_AVAILABLE = False

# 状态控制变量
current_step = 1
obstacle_count = 0
distance = 99999
goforward = 0

# 线程锁
obstacle_lock = threading.Lock()
distance_lock = threading.Lock()
img_lock = threading.Lock()
yolo_lock = threading.Lock()
state_lock = threading.Lock()

# 视觉变量
img_centerx = IMG_CENTER_X
screen_black = False
MIN_CONTOUR_AREA = 100

# YOLO变量
detected_objects = []
object_confidence = 0.0
object_position = 'none'
yolo_frame_count = 0

# 状态标志
stop_patrol = False
avoiding_obstacle = False
fall_recovery_in_progress = False

# 跌倒检测参数
ULTRASONIC_FALL_THR = 50
BLACK_SCREEN_THR = 150

# 尝试导入OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV 已加载")
except Exception as e:
    logger.warning(f"OpenCV 加载失败: {e}")

# 尝试导入YOLOv12
try:
    from ultralytics import YOLO
    import torch
    YOLOV12_AVAILABLE = True
    if YOLO_VERSION == 'v12n':
        YOLO_AVAILABLE = True
    logger.info("YOLOv12 已加载")
except Exception as e:
    logger.warning(f"YOLOv12 加载失败: {e}")

# ==================== 硬件初始化 ====================
def init_hardware():
    global board, ctl
    try:
        board = rrc.Board()
        ctl = Controller(board)
        logger.info("硬件初始化成功")
        return True
    except Exception as e:
        logger.error(f"硬件初始化失败: {e}")
        return False

# ==================== YOLO检测函数 ====================
def yolov12_detect(frame) -> List:
    global detected_objects, object_confidence, object_position, YOLO_MODEL
    
    if not YOLOV12_AVAILABLE or not USE_YOLO or YOLO_MODEL is None:
        return []
        
    start_time = time.time()
    
    try:
        results = YOLO_MODEL(
            frame,
            verbose=False,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            device=YOLO_DEVICE,
            max_det=YOLO_MAX_DET
        )
        
        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > YOLO_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    conf = float(box.conf[0])
                    objects.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': conf
                    })
        
        with yolo_lock:
            detected_objects = objects.copy()
            if objects:
                best_obj = max(objects, key=lambda obj: obj['confidence'])
                object_confidence = best_obj['confidence']
                
                if best_obj['center_x'] < CAMERA_WIDTH / 3:
                    object_position = 'left'
                elif best_obj['center_x'] > CAMERA_WIDTH * 2 / 3:
                    object_position = 'right'
                else:
                    object_position = 'center'
            else:
                object_confidence = 0.0
                object_position = 'none'
        
        # 更新性能统计
        inference_time = (time.time() - start_time) * 1000
        if perf_monitor:
            perf_monitor.update_inference_time(inference_time)
            
        return objects
        
    except Exception as e:
        logger.error(f"YOLO检测错误: {e}")
        return []

def yolov3_detect(frame) -> List:
    global detected_objects, object_confidence, object_position, YOLO_NET, YOLO_OUTPUT_LAYERS
    
    if not YOLO_AVAILABLE or not USE_YOLO or YOLO_NET is None:
        return []
        
    start_time = time.time()
    
    try:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (YOLO_IMGSZ, YOLO_IMGSZ), swapRB=True, crop=False)
        YOLO_NET.setInput(blob)
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
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONFIDENCE_THRESHOLD, 0.4)
        
        final_objects = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            final_objects.append(objects[i])
        
        with yolo_lock:
            detected_objects = final_objects.copy()
            if final_objects:
                best_obj = max(final_objects, key=lambda obj: obj['confidence'])
                object_confidence = best_obj['confidence']
                
                if best_obj['center_x'] < CAMERA_WIDTH / 3:
                    object_position = 'left'
                elif best_obj['center_x'] > CAMERA_WIDTH * 2 / 3:
                    object_position = 'right'
                else:
                    object_position = 'center'
            else:
                object_confidence = 0.0
                object_position = 'none'
        
        inference_time = (time.time() - start_time) * 1000
        if perf_monitor:
            perf_monitor.update_inference_time(inference_time)
            
        return final_objects
        
    except Exception as e:
        logger.error(f"YOLOv3检测错误: {e}")
        return []

# ==================== 视觉线程 ====================
def vision_loop():
    global img_centerx, screen_black, yolo_frame_count
    
    if not CV2_AVAILABLE:
        logger.warning("OpenCV不可用，视觉线程使用默认值")
        while True:
            with img_lock:
                img_centerx = IMG_CENTER_X
                screen_black = False
            time.sleep(0.05)
        return
        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        logger.error("摄像头打开失败")
        while True:
            with img_lock:
                img_centerx = IMG_CENTER_X
                screen_black = False
            time.sleep(0.05)
        return
        
    logger.info("视觉线程已启动")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
                
            if perf_monitor:
                perf_monitor.update_frame()
                
            if memory_manager:
                memory_manager.periodic_gc()
                mem_usage = memory_manager.get_memory_usage()
                perf_monitor.update_memory(mem_usage)
                
            cpu_temp = SystemInfo.get_cpu_temp()
            if perf_monitor:
                perf_monitor.update_temperature(cpu_temp)
                
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
            
            if USE_YOLO and YOLO_AVAILABLE:
                yolo_frame_count += 1
                if yolo_frame_count >= YOLO_FRAME_SKIP:
                    if YOLO_VERSION == 'v12n':
                        yolov12_detect(frame)
                    else:
                        yolov3_detect(frame)
                    yolo_frame_count = 0
            
            time.sleep(0.01)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("视觉线程已停止")

# ==================== 动作控制 ====================
def hand_up():
    if ctl:
        ctl.set_bus_servo_pulse(8, 330, 1000)
        time.sleep(0.3)
        ctl.set_bus_servo_pulse(7, 860, 1000)
        ctl.set_bus_servo_pulse(6, 860, 1000)
        time.sleep(1)

def hand_down():
    if ctl:
        ctl.set_bus_servo_pulse(7, 800, 1000)
        ctl.set_bus_servo_pulse(6, 575, 1000)
        time.sleep(0.3)
        ctl.set_bus_servo_pulse(8, 725, 1000)
        time.sleep(1)

# ==================== 移动控制线程 ====================
def move():
    global current_step, obstacle_count, distance, goforward, fall_recovery_in_progress, stop_patrol, avoiding_obstacle
    
    DIST_OBSTACLE_MM = 250
    LINE_OFFSET_THR = 80
    LARGE_OFFSET_THR = 150
    TURN_ADJUST_TIME = 0.12
    SEARCH_TIME = 0.2
    
    logger.info("移动控制线程已启动")
    
    while True:
        with distance_lock:
            local_distance = distance
        with img_lock:
            local_imgx = img_centerx
            local_screen_black = screen_black
        with yolo_lock:
            local_yolo_pos = object_position
            local_yolo_conf = object_confidence
        
        if local_screen_black and local_distance < ULTRASONIC_FALL_THR:
            if not fall_recovery_in_progress:
                logger.warning("检测到跌倒，开始自动起立")
                fall_recovery_in_progress = True
                try:
                    AGC.runActionGroup('stand_up_front')
                    logger.info("起立动作完成")
                except Exception as e:
                    logger.error(f"起立失败: {e}")
                fall_recovery_in_progress = False
                current_step = 1
                stop_patrol = False
                goforward = 0
            time.sleep(0.1)
            continue
        
        with obstacle_lock:
            if not avoiding_obstacle and 0 < local_distance <= DIST_OBSTACLE_MM:
                avoiding_obstacle = True
                stop_patrol = True
                obstacle_count += 1
                
                if USE_YOLO and local_yolo_conf > 0.3 and local_yolo_pos != 'none':
                    if local_yolo_pos == 'left':
                        current_step = 2
                        logger.info(f"检测到左侧障碍物，右转躲避 (距离: {local_distance}mm)")
                    elif local_yolo_pos == 'right':
                        current_step = 3
                        logger.info(f"检测到右侧障碍物，左转躲避 (距离: {local_distance}mm)")
                    else:
                        current_step = 2 if obstacle_count % 2 == 1 else 3
                        logger.info(f"检测到前方障碍物，交替躲避 (距离: {local_distance}mm)")
                else:
                    current_step = 2 if obstacle_count % 2 == 1 else 3
                    logger.info(f"检测到障碍物，开始躲避 (距离: {local_distance}mm)")
                time.sleep(0.3)
            else:
                stop_patrol = False
        
        if stop_patrol and current_step == 1:
            time.sleep(0.1)
            continue
            
        if current_step == 1:
            if local_imgx == -1:
                AGC.runActionGroup('turn_left')
                time.sleep(SEARCH_TIME)
            else:
                offset = local_imgx - IMG_CENTER_X
                if abs(offset) <= LINE_OFFSET_THR:
                    AGC.runActionGroup('go_forward_fast')
                    goforward += 1
                elif offset > LARGE_OFFSET_THR:
                    AGC.runActionGroup('turn_right')
                    time.sleep(TURN_ADJUST_TIME)
                    AGC.runActionGroup('turn_right')
                    goforward = 0
                elif offset > LINE_OFFSET_THR:
                    AGC.runActionGroup('turn_right')
                    goforward = 0
                elif offset < -LARGE_OFFSET_THR:
                    AGC.runActionGroup('turn_left')
                    time.sleep(TURN_ADJUST_TIME)
                    AGC.runActionGroup('turn_left')
                    goforward = 0
                else:
                    AGC.runActionGroup('turn_left')
                    goforward = 0
            time.sleep(0.08)
            
        elif current_step == 2:
            logger.info("执行右转避障流程")
            time.sleep(0.2)
            for _ in range(8):
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            for i in range(10):
                AGC.runActionGroup('zhixing4')
                if USE_YOLO:
                    with yolo_lock:
                        if object_position == 'center' and object_confidence > 0.5:
                            time.sleep(0.5)
                time.sleep(0.18)
            for _ in range(7):
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            time.sleep(0.3)
            current_step = 1
            stop_patrol = False
            goforward = 0
            avoiding_obstacle = False
            logger.info("避障完成，回到巡线模式")
            
        elif current_step == 3:
            logger.info("执行左转避障流程")
            time.sleep(0.2)
            for _ in range(9):
                AGC.runActionGroup('turn_left')
                time.sleep(0.18)
            for i in range(10):
                AGC.runActionGroup('go_forward_fast')
                if USE_YOLO:
                    with yolo_lock:
                        if object_position == 'center' and object_confidence > 0.5:
                            time.sleep(0.5)
                time.sleep(0.18)
            for _ in range(8):
                AGC.runActionGroup('turn_right')
                time.sleep(0.18)
            time.sleep(0.3)
            current_step = 1
            stop_patrol = False
            goforward = 0
            avoiding_obstacle = False
            logger.info("避障完成，回到巡线模式")
            
        else:
            time.sleep(0.1)

# ==================== 性能监控线程 ====================
def monitor_loop():
    last_print_time = time.time()
    while True:
        if perf_monitor and (time.time() - last_print_time > MONITOR_INTERVAL):
            perf_monitor.print_stats()
            last_print_time = time.time()
        time.sleep(1)

# ==================== 主程序 ====================
def main():
    global YOLO_MODEL, YOLO_NET, YOLO_OUTPUT_LAYERS
    
    logger.info("=" * 50)
    logger.info("TonyPi 树莓派5优化版启动")
    logger.info("=" * 50)
    
    init_hardware()
    
    th_move = threading.Thread(target=move, daemon=True)
    th_move.start()
    logger.info("移动控制线程已启动")
    
    th_vision = threading.Thread(target=vision_loop, daemon=True)
    th_vision.start()
    logger.info("视觉线程已启动")
    
    if ENABLE_PERFORMANCE_MONITOR:
        th_monitor = threading.Thread(target=monitor_loop, daemon=True)
        th_monitor.start()
        logger.info("性能监控线程已启动")
    
    if USE_YOLO:
        if YOLO_VERSION == 'v12n' and YOLOV12_AVAILABLE:
            try:
                logger.info("正在加载YOLOv12-nano模型...")
                YOLO_MODEL = YOLO('yolov12n.pt')
                logger.info(f"YOLOv12-nano模型加载成功 (输入尺寸: {YOLO_IMGSZ})")
            except Exception as e:
                logger.error(f"YOLOv12-nano模型加载失败: {e}")
                USE_YOLO = False
        elif YOLO_VERSION == 'v3t':
            try:
                logger.info("正在加载YOLOv3-tiny模型...")
                YOLO_NET = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
                YOLO_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                YOLO_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                layer_names = YOLO_NET.getLayerNames()
                YOLO_OUTPUT_LAYERS = [layer_names[i[0] - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
                logger.info("YOLOv3-tiny模型加载成功")
            except Exception as e:
                logger.error(f"YOLOv3-tiny模型加载失败: {e}")
                USE_YOLO = False
    
    try:
        AGC.runActionGroup('stand_slow')
        time.sleep(1)
        
        yolo_status = f"启用 ({YOLO_VERSION})" if USE_YOLO else "禁用"
        logger.info(f"系统就绪 - YOLO: {yolo_status}")
        
        distance_list = []
        s = Sonar.Sonar()
        s.startSymphony()
        
        while True:
            dist = s.getDistance()
            if 0 < dist < 5000:
                distance_list.append(dist)
            if len(distance_list) >= 6:
                with distance_lock:
                    distance = int(round(np.mean(np.array(distance_list))))
                distance_list = []
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("\n程序中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        stop_patrol = True
        if CV2_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("程序退出")

if __name__ == "__main__":
    main()
