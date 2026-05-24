#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np

from ..config import RobotConfig, Paths
from ..utils import get_logger, ThreadSafeValue, ThreadSafeFlag


CV2_AVAILABLE = False
YOLOV12_AVAILABLE = False
cv2 = None
YOLO = None
torch = None

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


@dataclass
class DetectionResult:
    objects: List[Dict[str, Any]]
    position: str
    confidence: float


@dataclass
class LineFollowerResult:
    center_x: int
    screen_black: bool


class VisionProcessor:
    def __init__(self, config: RobotConfig, paths: Paths):
        self.config = config
        self.paths = paths
        self.logger = get_logger()
        
        self.running = ThreadSafeFlag(False)
        self.camera_thread: Optional[threading.Thread] = None
        
        self.line_center = ThreadSafeValue[int](config.IMG_CENTER_X)
        self.screen_black = ThreadSafeFlag(False)
        self.detected_objects = ThreadSafeValue[List[Dict[str, Any]]]([])
        self.object_position = ThreadSafeValue[str]('none')
        self.object_confidence = ThreadSafeValue[float](0.0)
        
        self.yolo_model = None
        self.yolo_net = None
        self.yolo_output_layers = []
        self.yolo_available = False
        self.yolo_frame_count = 0

    def initialize_yolo(self) -> bool:
        if not self.config.USE_YOLO:
            self.logger.info("YOLO检测已禁用")
            return False

        if self.config.YOLO_VERSION == 'v12n':
            return self._initialize_yolov12()
        else:
            return self._initialize_yolov3()

    def _initialize_yolov12(self) -> bool:
        if not YOLOV12_AVAILABLE:
            self.logger.warning("YOLOv12需要PyTorch，请安装：pip install torch torchvision")
            return False
        try:
            self.logger.info("正在加载YOLOv12-nano模型...")
            self.yolo_model = YOLO(self.paths.YOLOV12_MODEL)
            self.yolo_available = True
            self.logger.info("YOLOv12-nano模型加载完成！")
            return True
        except Exception as e:
            self.logger.error(f"YOLOv12-nano模型加载失败: {e}", exc_info=True)
            return False

    def _initialize_yolov3(self) -> bool:
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV不可用，无法使用YOLOv3-tiny")
            return False
        try:
            self.logger.info("正在加载YOLOv3-tiny模型...")
            self.yolo_net = cv2.dnn.readNetFromDarknet(
                self.paths.YOLOV3_CFG,
                self.paths.YOLOV3_WEIGHTS
            )
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [
                layer_names[i[0] - 1] 
                for i in self.yolo_net.getUnconnectedOutLayers()
            ]
            self.yolo_available = True
            self.logger.info("YOLOv3-tiny模型加载完成！")
            return True
        except Exception as e:
            self.logger.error(f"YOLOv3-tiny模型加载失败: {e}", exc_info=True)
            return False

    def start(self):
        if self.running.get():
            self.logger.warning("视觉处理器已在运行")
            return
        self.running.set(True)
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self.logger.info("视觉处理器已启动")

    def _camera_loop(self):
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV不可用，使用默认值")
            while self.running.get():
                self.line_center.set(self.config.IMG_CENTER_X)
                self.screen_black.set(False)
                time.sleep(0.05)
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.IMG_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.IMG_HEIGHT)

        if not cap.isOpened():
            self.logger.error("摄像头打开失败，使用默认值")
            while self.running.get():
                self.line_center.set(self.config.IMG_CENTER_X)
                self.screen_black.set(False)
                time.sleep(0.05)
            return

        try:
            while self.running.get():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("摄像头读取失败")
                    time.sleep(0.02)
                    continue

                line_result = self._process_line_following(frame)
                self.line_center.set(line_result.center_x)
                self.screen_black.set(line_result.screen_black)

                if self.config.USE_YOLO and self.yolo_available:
                    self.yolo_frame_count += 1
                    if self.yolo_frame_count >= self.config.YOLO_FRAME_SKIP:
                        self._process_yolo(frame)
                        self.yolo_frame_count = 0

                time.sleep(0.02)
        finally:
            cap.release()
            if CV2_AVAILABLE:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _process_line_following(self, frame) -> LineFollowerResult:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.5):h, 0:w]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lower = np.array([180, 0, 0])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(lab, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        white_pixels = cv2.countNonZero(mask)
        screen_black = white_pixels < self.config.BLACK_SCREEN_THR

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.config.MIN_CONTOUR_AREA]

        if valid_contours:
            valid_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
            largest_contour = valid_contours[0]
            m = cv2.moments(largest_contour)
            if m['m00'] > 0:
                center_x = int(m['m10'] / m['m00'])
            else:
                center_x = -1
        else:
            center_x = -1

        return LineFollowerResult(center_x=center_x, screen_black=screen_black)

    def _process_yolo(self, frame):
        if self.config.YOLO_VERSION == 'v12n':
            self._detect_yolov12(frame)
        else:
            self._detect_yolov3(frame)

    def _detect_yolov12(self, frame):
        if not self.yolo_model:
            return
        try:
            results = self.yolo_model(
                frame, 
                verbose=False, 
                imgsz=self.config.YOLO_IMGSZ, 
                conf=self.config.YOLO_CONFIDENCE_THRESHOLD,
                device=self.config.YOLO_DEVICE
            )
            
            objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf[0] > self.config.YOLO_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        conf = float(box.conf[0])
                        objects.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'confidence': conf
                        })

            self._update_detection_result(objects)
        except Exception as e:
            self.logger.error(f"YOLOv12检测失败: {e}")

    def _detect_yolov3(self, frame):
        if not self.yolo_net:
            return
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
            layer_outputs = self.yolo_net.forward(self.yolo_output_layers)

            objects = []
            boxes = []
            confidences = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.config.YOLO_CONFIDENCE_THRESHOLD:
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

            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.config.YOLO_CONFIDENCE_THRESHOLD, 0.4)
            final_objects = []
            for i in indices:
                i = i[0] if isinstance(i, (list, np.ndarray)) else i
                final_objects.append(objects[i])

            self._update_detection_result(final_objects)
        except Exception as e:
            self.logger.error(f"YOLOv3检测失败: {e}")

    def _update_detection_result(self, objects: List[Dict[str, Any]]):
        self.detected_objects.set(objects.copy())
        if objects:
            best_obj = max(objects, key=lambda obj: obj['confidence'])
            self.object_confidence.set(best_obj['confidence'])
            if best_obj['center_x'] < self.config.IMG_WIDTH / 3:
                pos = 'left'
            elif best_obj['center_x'] > self.config.IMG_WIDTH * 2 / 3:
                pos = 'right'
            else:
                pos = 'center'
            self.object_position.set(pos)
        else:
            self.object_confidence.set(0.0)
            self.object_position.set('none')

    def get_line_center(self) -> int:
        return self.line_center.get()

    def is_screen_black(self) -> bool:
        return self.screen_black.get()

    def get_object_position(self) -> str:
        return self.object_position.get()

    def get_object_confidence(self) -> float:
        return self.object_confidence.get()

    def stop(self):
        self.running.set(False)
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        self.logger.info("视觉处理器已停止")
