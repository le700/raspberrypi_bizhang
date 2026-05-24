#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class ThreadSafeValue(Generic[T]):
    def __init__(self, initial_value: T):
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T):
        with self._lock:
            self._value = value


class ThreadSafeFlag:
    def __init__(self, initial_value: bool = False):
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> bool:
        with self._lock:
            return self._value

    def set(self, value: bool):
        with self._lock:
            self._value = value

    def toggle(self):
        with self._lock:
            self._value = not self._value
