#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from typing import Optional


class RobotLogger:
    _instance: Optional['RobotLogger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self._logger = logging.getLogger('TonyPiRobot')
        self._logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        self._logger.addHandler(console_handler)

    def set_debug_mode(self, debug: bool):
        if self._logger:
            level = logging.DEBUG if debug else logging.INFO
            self._logger.setLevel(level)
            for handler in self._logger.handlers:
                handler.setLevel(level)

    def debug(self, msg: str):
        if self._logger:
            self._logger.debug(msg)

    def info(self, msg: str):
        if self._logger:
            self._logger.info(msg)

    def warning(self, msg: str):
        if self._logger:
            self._logger.warning(msg)

    def error(self, msg: str, exc_info: bool = False):
        if self._logger:
            self._logger.error(msg, exc_info=exc_info)


def get_logger() -> RobotLogger:
    return RobotLogger()
