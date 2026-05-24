#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import hiwonder.ActionGroupControl as AGC

from ..config import ActionGroups
from ..utils import get_logger


class MotionController:
    def __init__(self, actions: ActionGroups):
        self.actions = actions
        self.logger = get_logger()

    def run_action(self, action_name: str, delay: float = 0.0):
        try:
            AGC.runActionGroup(action_name)
            if delay > 0:
                time.sleep(delay)
        except Exception as e:
            self.logger.error(f"执行动作失败 {action_name} 失败: {e}", exc_info=True)

    def stand_slow(self):
        self.run_action(self.actions.STAND_SLOW, 1.0)

    def stand_up_front(self):
        self.run_action(self.actions.STAND_UP_FRONT)

    def go_forward_fast(self):
        self.run_action(self.actions.GO_FORWARD_FAST)

    def go_forward(self):
        self.run_action(self.actions.ZHIXING4)

    def turn_left(self, times: int = 1, delay: float = 0.18):
        for _ in range(times):
            self.run_action(self.actions.TURN_LEFT, delay)

    def turn_right(self, times: int = 1, delay: float = 0.18):
        for _ in range(times):
            self.run_action(self.actions.TURN_RIGHT, delay)

    def wave_hand(self):
        try:
            import hiwonder.ros_robot_controller_sdk as rrc
            from hiwonder.Controller import Controller
            board = rrc.Board()
            ctl = Controller(board)
            ctl.set_bus_servo_pulse(8, 330, 1000)
            time.sleep(0.3)
            ctl.set_bus_servo_pulse(7, 860, 1000)
            ctl.set_bus_servo_pulse(6, 860, 1000)
            time.sleep(1)
            ctl.set_bus_servo_pulse(7, 800, 1000)
            ctl.set_bus_servo_pulse(6, 575, 1000)
            time.sleep(0.3)
            ctl.set_bus_servo_pulse(8, 725, 1000)
            time.sleep(1)
        except Exception as e:
            self.logger.error(f"挥手动作失败: {e}")
