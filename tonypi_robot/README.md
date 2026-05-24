# TonyPi 避障机器人 - 模块化重构版本

## 概述

这是 TonyPi 树莓派避障机器人的模块化重构版本，将原来的单文件应用重构为清晰的模块化架构，提高了代码的可维护性、可扩展性和可读性。

## 项目结构

```
tonypi_robot/
├── __init__.py          # 包初始化
├── main.py              # 主入口文件
├── README.md            # 项目说明
├── config/              # 配置模块
│   ├── __init__.py
│   └── config.py        # 配置类定义
├── utils/               # 工具模块
│   ├── __init__.py
│   ├── logger.py        # 日志系统
│   └── thread_safe.py   # 线程安全工具
├── sensors/             # 传感器模块
│   ├── __init__.py
│   └── ultrasonic.py    # 超声波传感器
├── vision/              # 视觉处理模块
│   ├── __init__.py
│   └── vision_processor.py  # 摄像头和YOLO检测
├── motion/              # 运动控制模块
│   ├── __init__.py
│   └── motion_controller.py # 动作执行
└── controller/          # 主控制器模块
    ├── __init__.py
    └── robot_controller.py  # 状态机和主逻辑
```

## 模块职责

### 1. config - 配置管理模块
- `RobotConfig`: 机器人运行配置（YOLO设置、摄像头参数、避障参数等）
- `Paths`: 文件路径配置
- `ActionGroups`: 动作组名称常量

### 2. utils - 工具模块
- `RobotLogger`: 单例日志系统
- `ThreadSafeValue`: 线程安全的值包装器
- `ThreadSafeFlag`: 线程安全的布尔标志

### 3. sensors - 传感器模块
- `UltrasonicSensor`: 超声波传感器封装，负责测距和数据滤波

### 4. vision - 视觉处理模块
- `VisionProcessor`: 视觉处理主类，封装摄像头采集、巡线检测、YOLO目标检测
- `DetectionResult`: 检测结果数据类
- `LineFollowerResult`: 巡线结果数据类

### 5. motion - 运动控制模块
- `MotionController`: 运动控制器，封装所有机器人动作

### 6. controller - 主控制器模块
- `RobotState`: 状态枚举（巡线、避障、恢复）
- `RobotController`: 主控制器，实现状态机和主逻辑

## 重构改进

### 1. 代码组织
- ✅ 单文件 → 模块化架构
- ✅ 职责清晰分离
- ✅ 可独立测试各模块

### 2. 配置管理
- ✅ 硬编码 → 配置类
- ✅ 类型注解支持
- ✅ 易于修改和调试

### 3. 日志系统
- ✅ print → 结构化日志
- ✅ 支持不同日志级别
- ✅ 统一的日志格式

### 4. 线程安全
- ✅ 全局变量 → 线程安全包装
- ✅ 避免竞态条件
- ✅ 更安全的多线程操作

### 5. 错误处理
- ✅ 完善的异常捕获
- ✅ 更好的容错机制
- ✅ 清晰的错误日志

## 使用方法

### 运行程序
```bash
cd tonypi_robot
python3 main.py
```

### 配置修改
编辑 `config/config.py` 中的 `RobotConfig` 类来调整参数：
- `YOLO_VERSION`: 选择 YOLO 版本 ('v12n' 或 'v3t')
- `USE_YOLO`: 是否启用 YOLO 检测
- `DEBUG`: 是否启用调试模式
- 其他避障、摄像头参数

## 状态机

```
[启动] → PATROLLING (巡线)
           ↓
    检测到障碍物
           ↓
    ┌────┴────┐
    ↓         ↓
AVOIDING   AVOIDING
  RIGHT      LEFT
    ↓         ↓
    └────┬────┘
         ↓
    回到巡线
```

## 依赖要求
- Python 3.6+
- hiwonder 相关库
- OpenCV (可选，用于视觉)
- PyTorch + Ultralytics (可选，用于 YOLOv12)
- NumPy

## 作者
TonyPi 避障机器人项目
