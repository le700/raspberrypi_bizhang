#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import signal

from config import RobotConfig, Paths, ActionGroups
from utils import get_logger
from sensors import UltrasonicSensor
from vision import VisionProcessor
from motion import MotionController
from controller import RobotController


def main():
    if sys.version_info.major == 2:
        print("请使用 Python 3 运行此程序！")
        sys.exit(1)

    config = RobotConfig()
    paths = Paths()
    actions = ActionGroups()

    logger = get_logger()
    logger.set_debug_mode(config.DEBUG)

    ultrasonic = UltrasonicSensor(config)
    vision = VisionProcessor(config, paths)
    motion = MotionController(actions)
    controller = RobotController(config, ultrasonic, vision, motion)

    def signal_handler(signum, frame):
        logger.info("收到停止信号，正在关闭...")
        controller.stop()
        vision.stop()
        ultrasonic.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("=" * 50)
        logger.info("TonyPi 避障机器人 - 模块化版本")
        logger.info("=" * 50)

        if not ultrasonic.initialize():
            logger.error("超声波传感器初始化失败")
            return

        vision.initialize_yolo()

        yolo_status = "启用" if config.USE_YOLO else "禁用"
        yolo_version = config.YOLO_VERSION if config.USE_YOLO else "-"
        logger.info(f"YOLO: {yolo_status} ({yolo_version})")

        vision.start()
        time.sleep(0.5)

        controller.start()

        logger.info("机器人启动完成，开始运行...")

        while True:
            dist = ultrasonic.read()
            if dist is not None:
                if config.USE_YOLO:
                    obj_pos = vision.get_object_position()
                    obj_conf = vision.get_object_confidence()
                    logger.debug(f"当前测距: {dist}mm, YOLO检测: {obj_pos} ({obj_conf:.2f})")
                else:
                    logger.debug(f"当前测距: {dist}mm")
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序发生错误: {e}", exc_info=True)
    finally:
        controller.stop()
        vision.stop()
        ultrasonic.shutdown()
        logger.info("程序已退出")


if __name__ == "__main__":
    main()
