#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import deque
from typing import Optional, List

import numpy as np

import hiwonder.Sonar as Sonar

from ..config import RobotConfig
from ..utils import get_logger, ThreadSafeValue


class UltrasonicSensor:
    def __init__(self, config: RobotConfig):
        self.config = config
        self.logger = get_logger()
        self.distance = ThreadSafeValue[int](99999)
        self.sensor: Optional[Sonar.Sonar] = None
        self.distance_queue: deque = deque(maxlen=config.ULTRASONIC_FILTER_SIZE)

    def initialize(self) -> bool:
        try:
            self.sensor = Sonar.Sonar()
            self.sensor.startSymphony()
            self.logger.info("超声波传感器初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"超声波传感器初始化失败: {e}", exc_info=True)
            return False

    def read(self) -> Optional[int]:
        if not self.sensor:
            return None
        try:
            dist = self.sensor.getDistance()
            if (self.config.ULTRASONIC_MIN_DIST < dist < 
                    self.config.ULTRASONIC_MAX_DIST):
                self.distance_queue.append(dist)
                if len(self.distance_queue) >= self.config.ULTRASONIC_FILTER_SIZE:
                    filtered_dist = int(round(np.mean(list(self.distance_queue))))
                    self.distance.set(filtered_dist)
                    self.distance_queue.clear()
                    return filtered_dist
            return None
        except Exception as e:
            self.logger.error(f"读取超声波数据失败: {e}")
            return None

    def get_distance(self) -> int:
        return self.distance.get()

    def shutdown(self):
        self.logger.info("超声波传感器关闭")
