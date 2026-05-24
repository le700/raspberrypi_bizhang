#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TonyPi避障机器人优化版
主要改进：
1. 增强的超声波滤波（中值+3σ异常检测）
2. YOLO距离估算
3. 多目标跟踪
4. 贝叶斯传感器融合
5. 增强的有限状态机
6. 动态障碍物检测
7. 历史记忆与死胡同检测
"""

import os
import random
import sys
import time
import math
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

try:
    import hiwonder.ros_robot_controller_sdk as rrc
    from hiwonder.Controller import Controller
    import hiwonder.Sonar as Sonar
    import hiwonder.ActionGroupControl as AGC
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False
    print("⚠️  未检测到HiWonder硬件库，运行在模拟模式")

# YOLO版本选择
YOLO_VERSION = 'v3t'  # 'v12n' 或 'v3t'
USE_YOLO = True  # 默认启用YOLO

# OpenCV和YOLO导入
CV2_AVAILABLE = False
YOLOV12_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    pass

try:
    from ultralytics import YOLO
    import torch
    YOLOV12_AVAILABLE = True
except Exception:
    pass

# ==================== 配置参数 ====================
class Config:
    # 超声波参数
    ULTRASONIC_WINDOW = 10
    ULTRASONIC_MIN_DIST = 0
    ULTRASONIC_MAX_DIST = 5000
    
    # 避障参数
    DIST_OBSTACLE_TRIGGER = 250  # mm
    DIST_EMERGENCY = 100  # mm
    DIST_WARNING = 350  # mm
    
    # YOLO参数
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_FRAME_SKIP = 2
    YOLO_IMGSZ = 416
    YOLO_DEVICE = 'cpu'
    
    # 相机参数（用于距离估算）
    CAMERA_FOCAL_LENGTH = 600  # 像素
    REAL_OBJECT_WIDTH = 0.5  # 米（假设典型障碍物宽度）
    
    # 状态机参数
    STATE_HISTORY_SIZE = 10
    POSITION_HISTORY_SIZE = 50
    
    # 融合参数
    BAYESIAN_PRIOR = 0.5

# ==================== 数据类 ====================
@dataclass
class Detection:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    class_id: int = 0
    estimated_distance: float = float('inf')
    track_id: Optional[int] = None

@dataclass
class SensorData:
    ultrasonic_dist: float = float('inf')
    detections: List[Detection] = None
    line_center_x: Optional[float] = None
    screen_black: bool = False
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.detections is None:
            self.detections = []

@dataclass
class FusionResult:
    obstacle_prob: float = 0.0
    estimated_distance: float = float('inf')
    obstacle_direction: str = 'none'  # left, center, right
    is_dynamic: bool = False

# ==================== 超声波滤波器 ====================
class UltrasonicFilter:
    def __init__(self, window_size: int = 10):
        self.window = deque(maxlen=window_size)
        self.last_valid = 0.0
    
    def update(self, value: float) -> float:
        if Config.ULTRASONIC_MIN_DIST < value < Config.ULTRASONIC_MAX_DIST:
            # 异常值检测（3σ原则）
            if len(self.window) >= 3:
                mean = np.mean(self.window)
                std = np.std(self.window)
                if std > 0 and abs(value - mean) > 3 * std:
                    return self.last_valid
            
            self.window.append(value)
            self.last_valid = np.median(self.window)  # 中值滤波更鲁棒
        
        return self.last_valid

# ==================== 多目标跟踪器 ====================
class SimpleTracker:
    def __init__(self):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.max_age = 10  # 最大丢帧数
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        current_time = time.time()
        updated_tracks = {}
        
        for det in detections:
            best_match_id = None
            best_iou = 0.3  # IoU阈值
            
            # 匹配现有轨迹
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(det, track['last_det'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # 更新轨迹
                track = self.tracks[best_match_id]
                track['last_det'] = det
                track['history'].append((det.center_x, det.center_y, current_time))
                track['age'] = 0
                det.track_id = best_match_id
                updated_tracks[best_match_id] = track
            else:
                # 新建轨迹
                new_id = self.next_id
                self.next_id += 1
                det.track_id = new_id
                updated_tracks[new_id] = {
                    'last_det': det,
                    'history': [(det.center_x, det.center_y, current_time)],
                    'age': 0
                }
        
        # 处理未匹配的轨迹
        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                track['age'] += 1
                if track['age'] < self.max_age:
                    updated_tracks[track_id] = track
        
        self.tracks = updated_tracks
        return detections
    
    def _calculate_iou(self, det1: Detection, det2: Detection) -> float:
        # 简化的IoU计算（只用中心点距离）
        dist = np.sqrt((det1.center_x - det2.center_x)**2 + (det1.center_y - det2.center_y)**2)
        max_dist = max(det1.width, det1.height, det2.width, det2.height)
        return max(0, 1 - dist / max_dist)
    
    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        if track_id not in self.tracks:
            return None
        
        history = self.tracks[track_id]['history']
        if len(history) < 3:
            return None
        
        velocities = []
        for i in range(1, len(history)):
            dt = history[i][2] - history[i-1][2]
            if dt <= 0:
                continue
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            velocities.append((dx/dt, dy/dt))
        
        if velocities:
            return np.mean(velocities, axis=0)
        return None

# ==================== 贝叶斯融合 ====================
class BayesianFusion:
    def __init__(self):
        self.p_obstacle = Config.BAYESIAN_PRIOR
    
    def _us_likelihood(self, dist: float) -> float:
        """超声波似然：距离越小，障碍物概率越大"""
        if dist < Config.DIST_EMERGENCY:
            return 0.99
        elif dist < Config.DIST_OBSTACLE_TRIGGER:
            return 0.8
        elif dist < Config.DIST_WARNING:
            return 0.5
        else:
            return 0.1
    
    def _yolo_likelihood(self, detections: List[Detection]) -> float:
        """YOLO似然：检测到目标且置信度高则概率大"""
        if not detections:
            return 0.1
        
        max_conf = max(d.confidence for d in detections)
        has_close = any(d.estimated_distance < Config.DIST_WARNING for d in detections)
        
        likelihood = max_conf * 0.7
        if has_close:
            likelihood += 0.2
        
        return min(0.99, likelihood)
    
    def fuse(self, sensor_data: SensorData) -> FusionResult:
        us_dist = sensor_data.ultrasonic_dist
        detections = sensor_data.detections
        
        # 计算似然
        p_us_obs = self._us_likelihood(us_dist)
        p_us_no = 1 - p_us_obs
        
        p_yolo_obs = self._yolo_likelihood(detections)
        p_yolo_no = 1 - p_yolo_obs
        
        # 贝叶斯更新
        numerator = self.p_obstacle * p_us_obs * p_yolo_obs
        denominator = numerator + (1 - self.p_obstacle) * p_us_no * p_yolo_no
        
        obstacle_prob = numerator / denominator if denominator > 0 else 0
        
        # 更新先验
        self.p_obstacle = 0.7 * self.p_obstacle + 0.3 * obstacle_prob
        
        # 距离估计
        dist_estimate = self._estimate_distance(us_dist, detections, obstacle_prob)
        
        # 方向判断
        direction = self._determine_direction(detections, us_dist)
        
        # 动态检测
        is_dynamic = self._detect_dynamic(detections)
        
        return FusionResult(
            obstacle_prob=obstacle_prob,
            estimated_distance=dist_estimate,
            obstacle_direction=direction,
            is_dynamic=is_dynamic
        )
    
    def _estimate_distance(self, us_dist: float, detections: List[Detection], prob: float) -> float:
        if not detections:
            return us_dist
        
        # YOLO距离估算
        yolo_dists = [d.estimated_distance for d in detections if d.estimated_distance < float('inf')]
        if yolo_dists:
            yolo_dist = np.min(yolo_dists)  # 取最近的
            # 加权融合
            weight_yolo = prob
            weight_us = 1 - prob
            return weight_us * us_dist + weight_yolo * yolo_dist * 1000  # 转换为mm
        
        return us_dist
    
    def _determine_direction(self, detections: List[Detection], us_dist: float) -> str:
        if not detections:
            if us_dist < Config.DIST_OBSTACLE_TRIGGER:
                return 'center'
            return 'none'
        
        # 取最近的障碍物
        closest = min(detections, key=lambda d: d.estimated_distance)
        img_width = 640
        
        if closest.center_x < img_width / 3:
            return 'left'
        elif closest.center_x > img_width * 2 / 3:
            return 'right'
        else:
            return 'center'
    
    def _detect_dynamic(self, detections: List[Detection]) -> bool:
        for det in detections:
            if det.track_id is not None and tracker:
                vel = tracker.get_velocity(det.track_id)
                if vel is not None:
                    speed = np.sqrt(vel[0]**2 + vel[1]**2)
                    if speed > 10:  # 像素/帧
                        return True
        return False

# ==================== 增强状态机 ====================
class StateMachine:
    STATE_IDLE = 'idle'
    STATE_PATROL = 'patrol'
    STATE_EVALUATE = 'evaluate'
    STATE_AVOID_STATIC = 'avoid_static'
    STATE_WAIT_DYNAMIC = 'wait_dynamic'
    STATE_ESCAPE = 'escape'
    
    def __init__(self):
        self.state = self.STATE_IDLE
        self.state_history = deque(maxlen=Config.STATE_HISTORY_SIZE)
        self.state_start_time = time.time()
        self.obstacle_count = 0
    
    def transition(self, fusion_result: FusionResult, sensor_data: SensorData) -> str:
        self.state_history.append(self.state)
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        new_state = self.state
        
        if self.state == self.STATE_IDLE:
            new_state = self.STATE_PATROL
        
        elif self.state == self.STATE_PATROL:
            if fusion_result.obstacle_prob > 0.6:
                if fusion_result.is_dynamic:
                    new_state = self.STATE_WAIT_DYNAMIC
                else:
                    new_state = self.STATE_EVALUATE
        
        elif self.state == self.STATE_EVALUATE:
            self.obstacle_count += 1
            new_state = self.STATE_AVOID_STATIC
        
        elif self.state == self.STATE_AVOID_STATIC:
            if fusion_result.obstacle_prob < 0.3 or state_duration > 5.0:
                new_state = self.STATE_PATROL
        
        elif self.state == self.STATE_WAIT_DYNAMIC:
            if not fusion_result.is_dynamic or state_duration > 3.0:
                new_state = self.STATE_PATROL
        
        elif self.state == self.STATE_ESCAPE:
            if state_duration > 4.0:
                new_state = self.STATE_PATROL
        
        if new_state != self.state:
            self.state = new_state
            self.state_start_time = current_time
            print(f"🔄 状态转换: {self.state_history[-1]} → {new_state}")
        
        return new_state

# ==================== 历史分析器 ====================
class HistoryAnalyzer:
    def __init__(self):
        self.direction_history = deque(maxlen=20)
        self.action_history = deque(maxlen=30)
    
    def record_action(self, action: str):
        self.action_history.append(action)
        if 'turn_left' in action:
            self.direction_history.append(-1)
        elif 'turn_right' in action:
            self.direction_history.append(1)
        else:
            self.direction_history.append(0)
    
    def detect_loop(self) -> bool:
        """检测是否在打转"""
        if len(self.direction_history) < 15:
            return False
        
        recent = list(self.direction_history)[-15:]
        left_count = recent.count(-1)
        right_count = recent.count(1)
        
        # 如果大部分是转向
        if left_count > 10 or right_count > 10:
            return True
        return False
    
    def get_escape_sequence(self) -> List[str]:
        """逃脱序列"""
        return ['turn_left', 'turn_left', 'turn_left', 
                'go_forward_fast', 'go_forward_fast', 'go_forward_fast']

# ==================== 全局变量 ====================
us_filter = UltrasonicFilter()
tracker = SimpleTracker()
fusion = BayesianFusion()
state_machine = StateMachine()
history_analyzer = HistoryAnalyzer()

# 线程安全的全局状态
class GlobalState:
    def __init__(self):
        self._lock = threading.Lock()
        self._distance = float('inf')
        self._detections = []
        self._line_center_x = None
        self._screen_black = False
        self._current_step = 1
        self._stop_patrol = False
        self._avoiding_obstacle = False
        self._fall_recovery = False
    
    @property
    def distance(self):
        with self._lock:
            return self._distance
    
    @distance.setter
    def distance(self, value):
        with self._lock:
            self._distance = value
    
    @property
    def detections(self):
        with self._lock:
            return self._detections.copy()
    
    @detections.setter
    def detections(self, value):
        with self._lock:
            self._detections = value
    
    @property
    def line_center_x(self):
        with self._lock:
            return self._line_center_x
    
    @line_center_x.setter
    def line_center_x(self, value):
        with self._lock:
            self._line_center_x = value
    
    @property
    def screen_black(self):
        with self._lock:
            return self._screen_black
    
    @screen_black.setter
    def screen_black(self, value):
        with self._lock:
            self._screen_black = value
    
    @property
    def current_step(self):
        with self._lock:
            return self._current_step
    
    @current_step.setter
    def current_step(self, value):
        with self._lock:
            self._current_step = value
    
    @property
    def stop_patrol(self):
        with self._lock:
            return self._stop_patrol
    
    @stop_patrol.setter
    def stop_patrol(self, value):
        with self._lock:
            self._stop_patrol = value
    
    @property
    def avoiding_obstacle(self):
        with self._lock:
            return self._avoiding_obstacle
    
    @avoiding_obstacle.setter
    def avoiding_obstacle(self, value):
        with self._lock:
            self._avoiding_obstacle = value
    
    @property
    def fall_recovery(self):
        with self._lock:
            return self._fall_recovery
    
    @fall_recovery.setter
    def fall_recovery(self, value):
        with self._lock:
            self._fall_recovery = value

global_state = GlobalState()

# ==================== YOLO函数 ====================
YOLO_MODEL = None
YOLO_NET = None
YOLO_OUTPUT_LAYERS = []

def estimate_distance_from_bbox(width_pixels: float) -> float:
    """从边界框宽度估算距离（米）"""
    if width_pixels <= 0:
        return float('inf')
    return (Config.REAL_OBJECT_WIDTH * Config.CAMERA_FOCAL_LENGTH) / width_pixels

def yolov12_detect(frame):
    global YOLO_MODEL
    if not YOLOV12_AVAILABLE or not USE_YOLO or YOLO_MODEL is None:
        return []
    
    try:
        results = YOLO_MODEL(frame, verbose=False, imgsz=Config.YOLO_IMGSZ, 
                            conf=Config.YOLO_CONFIDENCE_THRESHOLD, device=Config.YOLO_DEVICE)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > Config.YOLO_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0]) if box.cls is not None else 0
                    
                    est_dist = estimate_distance_from_bbox(width)
                    
                    detections.append(Detection(
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        confidence=conf,
                        class_id=class_id,
                        estimated_distance=est_dist
                    ))
        
        # 跟踪
        detections = tracker.update(detections)
        return detections
    except Exception as e:
        print(f"YOLOv12检测错误: {e}")
        return []

def yolov3_detect(frame):
    global YOLO_NET, YOLO_OUTPUT_LAYERS
    if not CV2_AVAILABLE or not USE_YOLO or YOLO_NET is None:
        return []
    
    try:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        YOLO_NET.setInput(blob)
        layer_outputs = YOLO_NET.forward(YOLO_OUTPUT_LAYERS)
        
        boxes = []
        confidences = []
        detections_raw = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > Config.YOLO_CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    detections_raw.append((center_x, center_y, w, h, confidence, class_id))
        
        # NMS
        indices = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, Config.YOLO_CONFIDENCE_THRESHOLD, 0.4)
        
        detections = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            center_x, center_y, w, h, conf, class_id = detections_raw[i]
            est_dist = estimate_distance_from_bbox(w)
            
            detections.append(Detection(
                center_x=center_x,
                center_y=center_y,
                width=w,
                height=h,
                confidence=conf,
                class_id=class_id,
                estimated_distance=est_dist
            ))
        
        # 跟踪
        detections = tracker.update(detections)
        return detections
    except Exception as e:
        print(f"YOLOv3检测错误: {e}")
        return []

# ==================== 视觉线程 ====================
def vision_thread():
    print("👁️  视觉线程启动")
    
    if not CV2_AVAILABLE:
        print("⚠️  OpenCV不可用，视觉线程回退")
        while True:
            global_state.line_center_x = 320
            global_state.screen_black = False
            time.sleep(0.05)
        return
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("⚠️  摄像头打开失败")
            while True:
                global_state.line_center_x = 320
                global_state.screen_black = False
                time.sleep(0.05)
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            
            h, w = frame.shape[:2]
            roi = frame[int(h * 0.5):h, 0:w]
            
            # 巡线
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            lower = np.array([180, 0, 0])
            upper = np.array([255, 255, 255])
            mask = cv2.inRange(lab, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            white_pixels = cv2.countNonZero(mask)
            
            global_state.screen_black = white_pixels < 150
            
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 100]
            
            if valid_contours:
                valid_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
                largest = valid_contours[0]
                m = cv2.moments(largest)
                if m['m00'] > 0:
                    global_state.line_center_x = int(m['m10'] / m['m00'])
                else:
                    global_state.line_center_x = -1
            else:
                global_state.line_center_x = -1
            
            # YOLO检测（跳帧）
            if USE_YOLO:
                frame_count += 1
                if frame_count >= Config.YOLO_FRAME_SKIP:
                    frame_count = 0
                    if YOLO_VERSION == 'v12n':
                        dets = yolov12_detect(frame)
                    else:
                        dets = yolov3_detect(frame)
                    global_state.detections = dets
            
            time.sleep(0.02)
    except Exception as e:
        print(f"视觉线程错误: {e}")
    finally:
        if cap is not None:
            cap.release()

# ==================== 动作执行 ====================
def execute_action(action: str):
    """执行动作"""
    history_analyzer.record_action(action)
    
    if not HW_AVAILABLE:
        print(f"🎬 [模拟] 执行动作: {action}")
        time.sleep(0.2)
        return
    
    try:
        AGC.runActionGroup(action)
    except Exception as e:
        print(f"动作执行错误: {e}")

# ==================== 决策线程 ====================
def decision_thread():
    print("🧠 决策线程启动")
    
    while True:
        try:
            # 收集传感器数据
            sensor_data = SensorData(
                ultrasonic_dist=global_state.distance,
                detections=global_state.detections,
                line_center_x=global_state.line_center_x,
                screen_black=global_state.screen_black,
                timestamp=time.time()
            )
            
            # 跌倒检测
            if sensor_data.screen_black and sensor_data.ultrasonic_dist < 50:
                if not global_state.fall_recovery:
                    print("⚠️  检测到跌倒！")
                    global_state.fall_recovery = True
                    execute_action('stand_up_front')
                    global_state.fall_recovery = False
                    global_state.current_step = 1
                time.sleep(0.1)
                continue
            
            # 传感器融合
            fusion_result = fusion.fuse(sensor_data)
            
            # 状态机更新
            state = state_machine.transition(fusion_result, sensor_data)
            
            # 检查循环
            if history_analyzer.detect_loop():
                print("🔄 检测到循环，执行逃脱")
                for action in history_analyzer.get_escape_sequence():
                    execute_action(action)
                continue
            
            # 根据状态执行动作
            if state == StateMachine.STATE_PATROL:
                patrol_action(sensor_data)
            elif state == StateMachine.STATE_EVALUATE:
                print("🤔 评估场景...")
                time.sleep(0.3)
            elif state == StateMachine.STATE_AVOID_STATIC:
                avoid_static_obstacle(fusion_result)
            elif state == StateMachine.STATE_WAIT_DYNAMIC:
                print("⏳ 等待动态障碍物...")
                time.sleep(0.3)
            elif state == StateMachine.STATE_ESCAPE:
                for action in history_analyzer.get_escape_sequence():
                    execute_action(action)
            
            # 打印状态
            print_status(sensor_data, fusion_result, state)
            
            time.sleep(0.05)
        except Exception as e:
            print(f"决策线程错误: {e}")
            time.sleep(0.1)

def patrol_action(sensor_data: SensorData):
    """巡线动作"""
    line_center = sensor_data.line_center_x
    IMG_CENTER = 320
    LINE_OFFSET_THR = 80
    LARGE_OFFSET_THR = 150
    
    if line_center == -1:
        print("🔍 未检测到线，搜索中...")
        execute_action('turn_left')
        time.sleep(0.2)
    else:
        offset = line_center - IMG_CENTER
        if abs(offset) <= LINE_OFFSET_THR:
            execute_action('go_forward_fast')
        elif offset > LARGE_OFFSET_THR:
            execute_action('turn_right')
            time.sleep(0.12)
            execute_action('turn_right')
        elif offset > LINE_OFFSET_THR:
            execute_action('turn_right')
        elif offset < -LARGE_OFFSET_THR:
            execute_action('turn_left')
            time.sleep(0.12)
            execute_action('turn_left')
        else:
            execute_action('turn_left')

def avoid_static_obstacle(fusion_result: FusionResult):
    """避障动作（动态选择方向）"""
    direction = fusion_result.obstacle_direction
    dist = fusion_result.estimated_distance
    
    print(f"🚧 避障: 距离={dist:.0f}mm, 方向={direction}")
    
    # 选择相反方向绕行
    if direction == 'left':
        # 障碍物在左，从右边绕
        avoid_sequence = [
            'turn_right', 'turn_right', 'turn_right',
            'go_forward_fast', 'go_forward_fast', 'go_forward_fast',
            'turn_left', 'turn_left', 'turn_left'
        ]
    elif direction == 'right':
        # 障碍物在右，从左边绕
        avoid_sequence = [
            'turn_left', 'turn_left', 'turn_left',
            'go_forward_fast', 'go_forward_fast', 'go_forward_fast',
            'turn_right', 'turn_right', 'turn_right'
        ]
    else:
        # 中间，交替选择
        if state_machine.obstacle_count % 2 == 0:
            avoid_sequence = [
                'turn_right', 'turn_right', 'turn_right',
                'go_forward_fast', 'go_forward_fast',
                'turn_left', 'turn_left', 'turn_left'
            ]
        else:
            avoid_sequence = [
                'turn_left', 'turn_left', 'turn_left',
                'go_forward_fast', 'go_forward_fast',
                'turn_right', 'turn_right', 'turn_right'
            ]
    
    # 执行序列
    for action in avoid_sequence:
        execute_action(action)
        time.sleep(0.1)

def print_status(sensor_data: SensorData, fusion_result: FusionResult, state: str):
    """打印状态信息"""
    det_count = len(sensor_data.detections)
    status = (
        f"📊 状态: {state:12} | "
        f"距离: {sensor_data.ultrasonic_dist:4.0f}mm | "
        f"障碍概率: {fusion_result.obstacle_prob:.2f} | "
        f"检测: {det_count:2d} | "
        f"方向: {fusion_result.obstacle_direction:6}"
    )
    print(status)

# ==================== 主函数 ====================
def main():
    print("🚀 TonyPi避障机器人优化版启动")
    print("=" * 60)
    
    # 初始化YOLO
    if USE_YOLO:
        if YOLO_VERSION == 'v12n':
            if YOLOV12_AVAILABLE:
                try:
                    print("📦 加载YOLOv12-nano...")
                    YOLO_MODEL = YOLO('yolov12n.pt')
                    print("✅ YOLOv12加载成功")
                except Exception as e:
                    print(f"❌ YOLOv12加载失败: {e}")
                    USE_YOLO = False
            else:
                print("⚠️  YOLOv12需要ultralytics库")
                USE_YOLO = False
        else:
            if CV2_AVAILABLE:
                try:
                    print("📦 加载YOLOv3-tiny...")
                    YOLO_NET = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
                    YOLO_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    YOLO_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    layer_names = YOLO_NET.getLayerNames()
                    YOLO_OUTPUT_LAYERS = [layer_names[i[0] - 1] if isinstance(i, (list, np.ndarray)) else layer_names[i - 1] 
                                        for i in YOLO_NET.getUnconnectedOutLayers()]
                    print("✅ YOLOv3-tiny加载成功")
                except Exception as e:
                    print(f"❌ YOLOv3-tiny加载失败: {e}")
                    print("💡 请确保yolov3-tiny.cfg和yolov3-tiny.weights在同目录")
                    USE_YOLO = False
    
    print(f"🎯 YOLO: {'启用' if USE_YOLO else '禁用'}")
    print("=" * 60)
    
    # 初始化硬件
    if HW_AVAILABLE:
        try:
            board = rrc.Board()
            controller = Controller(board)
            print("🤖 硬件初始化完成")
            execute_action('stand_slow')
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  硬件初始化失败: {e}")
    
    # 启动线程
    th_vision = threading.Thread(target=vision_thread, daemon=True)
    th_vision.start()
    
    th_decision = threading.Thread(target=decision_thread, daemon=True)
    th_decision.start()
    
    # 主循环：超声波读取
    if HW_AVAILABLE:
        try:
            sonar = Sonar.Sonar()
            sonar.startSymphony()
            
            distance_list = []
            while True:
                dist = sonar.getDistance()
                filtered_dist = us_filter.update(dist)
                
                distance_list.append(filtered_dist)
                if len(distance_list) >= 5:
                    global_state.distance = np.mean(distance_list)
                    distance_list = []
                
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n👋 程序停止")
    else:
        # 模拟模式
        print("🎮 运行在模拟模式")
        try:
            sim_dist = 500.0
            while True:
                # 模拟一些变化
                sim_dist += random.uniform(-20, 20)
                sim_dist = max(100, min(600, sim_dist))
                
                filtered_dist = us_filter.update(sim_dist)
                global_state.distance = filtered_dist
                
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n👋 程序停止")

if __name__ == '__main__':
    main()
