#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from enum import Enum
from typing import Optional

from ..config import RobotConfig
from ..utils import get_logger, ThreadSafeValue, ThreadSafeFlag
from ..sensors import UltrasonicSensor
from ..vision import VisionProcessor
from ..motion import MotionController


class RobotState(Enum):
    PATROLLING = 1
    AVOIDING_RIGHT = 2
    AVOIDING_LEFT = 3
    RECOVERING = 4


class RobotController:
    def __init__(
        self,
        config: RobotConfig,
        ultrasonic: UltrasonicSensor,
        vision: VisionProcessor,
        motion: MotionController
    ):
        self.config = config
        self.ultrasonic = ultrasonic
        self.vision = vision
        self.motion = motion
        self.logger = get_logger()

        self.state = ThreadSafeValue[RobotState](RobotState.PATROLLING)
        self.obstacle_count = ThreadSafeValue[int](0)
        self.avoiding_obstacle = ThreadSafeFlag(False)
        self.stop_patrol = ThreadSafeFlag(False)
        self.fall_recovery_in_progress = ThreadSafeFlag(False)
        self.running = ThreadSafeFlag(False)
        self.main_thread: Optional[threading.Thread] = None
        self.go_forward_count = 0

    def start(self):
        if self.running.get():
            self.logger.warning("机器人控制器已在运行")
            return
        
        self.motion.stand_slow()
        self.running.set(True)
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        self.logger.info("机器人控制器已启动")

    def _main_loop(self):
        try:
            while self.running.get():
                self._update()
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
        except Exception as e:
            self.logger.error(f"主循环发生错误: {e}", exc_info=True)

    def _update(self):
        dist = self.ultrasonic.get_distance()
        line_center = self.vision.get_line_center()
        screen_black = self.vision.is_screen_black()
        object_pos = self.vision.get_object_position()
        object_conf = self.vision.get_object_confidence()

        if self.config.DEBUG:
            if self.config.USE_YOLO:
                self.logger.debug(f"测距: {dist}mm, 线心: {line_center}, 检测: {object_pos} ({object_conf:.2f})")
            else:
                self.logger.debug(f"测距: {dist}mm, 线心: {line_center}")

        if screen_black and dist < self.config.ULTRASONIC_FALL_THR:
            if not self.fall_recovery_in_progress.get():
                self.fall_recovery_in_progress.set(True)
                self.logger.warning("检测到跌倒，开始自动起立...")
                self.motion.stand_up_front()
                self.logger.info("起立动作完成")
                self.fall_recovery_in_progress.set(False)
                self.state.set(RobotState.PATROLLING)
                self.stop_patrol.set(False)
                self.go_forward_count = 0
            time.sleep(0.1)
            return

        current_state = self.state.get()

        if current_state == RobotState.PATROLLING:
            self._handle_patrolling(dist, line_center, object_pos, object_conf)
        elif current_state == RobotState.AVOIDING_RIGHT:
            self._handle_avoid_right()
        elif current_state == RobotState.AVOIDING_LEFT:
            self._handle_avoid_left()

    def _handle_patrolling(
        self, 
        dist: int, 
        line_center: int, 
        object_pos: str, 
        object_conf: float
    ):
        with self.avoiding_obstacle._lock:
            if not self.avoiding_obstacle._value and 0 < dist <= self.config.DIST_OBSTACLE_MM:
                self.avoiding_obstacle._value = True
                self.stop_patrol.set(True)
                count = self.obstacle_count.get() + 1
                self.obstacle_count.set(count)
                
                self.logger.warning(f"触发避障！超声波={dist}mm")
                
                if self.config.USE_YOLO and object_conf > 0.3 and object_pos != 'none':
                    if object_pos == 'left':
                        self.logger.info("YOLO检测障碍物在左侧，选择从右侧绕行")
                        self.state.set(RobotState.AVOIDING_RIGHT)
                    elif object_pos == 'right':
                        self.logger.info("YOLO检测障碍物在右侧，选择从左侧绕行")
                        self.state.set(RobotState.AVOIDING_LEFT)
                    else:
                        if count % 2 == 1:
                            self.state.set(RobotState.AVOIDING_RIGHT)
                        else:
                            self.state.set(RobotState.AVOIDING_LEFT)
                        self.logger.info(f"YOLO检测障碍物在中间，轮流选择避障方向")
                else:
                    if count % 2 == 1:
                        self.state.set(RobotState.AVOIDING_RIGHT)
                    else:
                        self.state.set(RobotState.AVOIDING_LEFT)
                    self.logger.info("使用轮流避障策略")
                
                time.sleep(0.3)
                return
            else:
                self.stop_patrol.set(False)

        if self.stop_patrol.get():
            time.sleep(0.1)
            return

        if line_center == -1:
            self.logger.info("未检测到线，缓慢搜索")
            self.motion.turn_left()
            time.sleep(self.config.SEARCH_TIME)
        else:
            offset = line_center - self.config.IMG_CENTER_X
            self.logger.debug(f"巡线: 线心={line_center}, 偏移={offset}")
            if abs(offset) <= self.config.LINE_OFFSET_THR:
                self.motion.go_forward_fast()
                self.logger.debug("直行（快速）")
                self.go_forward_count += 1
            elif offset > self.config.LARGE_OFFSET_THR:
                self.motion.turn_right()
                time.sleep(self.config.TURN_ADJUST_TIME)
                self.motion.turn_right()
                self.logger.debug("右急调")
                self.go_forward_count = 0
            elif offset > self.config.LINE_OFFSET_THR:
                self.motion.turn_right()
                self.logger.debug("右微调")
                self.go_forward_count = 0
            elif offset < -self.config.LARGE_OFFSET_THR:
                self.motion.turn_left()
                time.sleep(self.config.TURN_ADJUST_TIME)
                self.motion.turn_left()
                self.logger.debug("左急调")
                self.go_forward_count = 0
            else:
                self.motion.turn_left()
                self.logger.debug("左微调")
                self.go_forward_count = 0
        time.sleep(0.08)

    def _handle_avoid_right(self):
        self.logger.info("执行避障：右转绕行")
        time.sleep(0.2)
        
        self.motion.turn_right(times=8, delay=0.18)
        
        for _ in range(10):
            self.motion.go_forward()
            if self.config.USE_YOLO:
                obj_pos = self.vision.get_object_position()
                obj_conf = self.vision.get_object_confidence()
                if obj_pos == 'center' and obj_conf > 0.5:
                    self.logger.warning("YOLO检测到正前方有新障碍物，增加观察时间")
                    time.sleep(0.5)
            time.sleep(0.18)
        
        self.motion.turn_left(times=7, delay=0.18)
        
        self.logger.info("右转绕行完成，短暂观察周围")
        time.sleep(0.3)
        
        self.state.set(RobotState.PATROLLING)
        self.stop_patrol.set(False)
        self.go_forward_count = 0
        self.avoiding_obstacle.set(False)
        self.logger.info("确认安全，回到巡线状态")

    def _handle_avoid_left(self):
        self.logger.info("执行避障：左转绕行")
        time.sleep(0.2)
        
        self.motion.turn_left(times=9, delay=0.18)
        
        for _ in range(10):
            self.motion.go_forward_fast()
            if self.config.USE_YOLO:
                obj_pos = self.vision.get_object_position()
                obj_conf = self.vision.get_object_confidence()
                if obj_pos == 'center' and obj_conf > 0.5:
                    self.logger.warning("YOLO检测到正前方有新障碍物，增加观察时间")
                    time.sleep(0.5)
            time.sleep(0.18)
        
        self.motion.turn_right(times=8, delay=0.18)
        
        self.logger.info("左转绕行完成，短暂观察周围")
        time.sleep(0.3)
        
        self.state.set(RobotState.PATROLLING)
        self.stop_patrol.set(False)
        self.go_forward_count = 0
        self.avoiding_obstacle.set(False)
        self.logger.info("确认安全，回到巡线状态")

    def stop(self):
        self.running.set(False)
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
        self.logger.info("机器人控制器已停止")
