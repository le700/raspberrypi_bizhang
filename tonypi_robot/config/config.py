#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Literal


@dataclass
class RobotConfig:
    YOLO_VERSION: Literal['v12n', 'v3t'] = 'v3t'
    USE_YOLO: bool = False
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_FRAME_SKIP: int = 3
    YOLO_IMGSZ: int = 640
    YOLO_DEVICE: str = 'cpu'

    IMG_WIDTH: int = 640
    IMG_HEIGHT: int = 480
    IMG_CENTER_X: int = 200
    MIN_CONTOUR_AREA: int = 100

    ULTRASONIC_FALL_THR: int = 50
    BLACK_SCREEN_THR: int = 150

    DIST_OBSTACLE_MM: int = 250
    LINE_OFFSET_THR: int = 80
    LARGE_OFFSET_THR: int = 150
    TURN_ADJUST_TIME: float = 0.12
    SEARCH_TIME: float = 0.2

    ULTRASONIC_FILTER_SIZE: int = 6
    ULTRASONIC_MIN_DIST: int = 0
    ULTRASONIC_MAX_DIST: int = 5000

    DEBUG: bool = False


@dataclass
class Paths:
    YOLOV12_MODEL: str = 'yolov12n.pt'
    YOLOV3_CFG: str = 'yolov3-tiny.cfg'
    YOLOV3_WEIGHTS: str = 'yolov3-tiny.weights'


@dataclass
class ActionGroups:
    STAND_SLOW: str = 'stand_slow'
    STAND_UP_FRONT: str = 'stand_up_front'
    GO_FORWARD_FAST: str = 'go_forward_fast'
    ZHIXING4: str = 'zhixing4'
    TURN_LEFT: str = 'turn_left'
    TURN_RIGHT: str = 'turn_right'
