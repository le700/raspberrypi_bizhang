
# TonyPi 避障机器人优化方案总结

## 📋 项目概述

本项目对 TonyPi 双足人形机器人的避障系统进行了全面优化，结合超声波传感器和 YOLO 视觉检测，实现了更智能、更鲁棒的避障能力。

## 🗂️ 文件结构

```
/workspace/
├── raspberrypi_bizhang.py          # 原始版本
├── raspberrypi_bizhang_improved.py # 优化版本（新增）
├── TonyPi_Optimization_Plan.md     # 详细优化方案（新增）
├── README_优化方案总结.md          # 本文档（新增）
├── YOLO_SETUP.md
├── YOLO_TRAIN_GUIDE.md
├── YOLOv12_SETUP.md
├── YOLOv12_TRAIN_GUIDE.md
└── requirements.txt
```

## 🎯 核心改进点

### 1️⃣ 现有算法分析

**原有问题**：
- 超声波仅使用简单平均滤波，抗干扰能力弱
- YOLO仅检测目标方位，无距离估算和跟踪
- 传感器融合为简单条件判断，信息利用率低
- 避障路径为固定动作序列，无动态调整
- 无动态障碍物检测和历史记忆

### 2️⃣ 传感器融合策略改进

**实现的优化**：
- ✅ **超声波滤波增强**：中值滤波 + 3σ异常值检测
  - 类：`UltrasonicFilter`
  - 更鲁棒的距离测量，有效过滤突发噪声

- ✅ **YOLO信息增强**：
  - 边界框距离估算（`estimate_distance_from_bbox`）
  - 多目标跟踪（`SimpleTracker`）
  - 动态障碍物检测（通过轨迹速度）

- ✅ **贝叶斯融合**：
  - 类：`BayesianFusion`
  - 融合超声波和YOLO的似然概率
  - 输出障碍物概率、距离估计、方位、是否动态

### 3️⃣ 避障路径规划优化

**实现的优化**：
- ✅ 动态避障方向选择（基于YOLO检测的障碍物方位）
- ✅ 多级避障策略（紧急/警告/正常）
- ✅ 模块化的动作序列生成

### 4️⃣ 决策逻辑改进

**实现的优化**：
- ✅ **增强有限状态机**（`StateMachine`）：
  - `idle` - 待机
  - `patrol` - 巡线
  - `evaluate` - 场景评估
  - `avoid_static` - 静态避障
  - `wait_dynamic` - 等待动态障碍
  - `escape` - 逃脱模式

### 5️⃣ 边缘情况处理

**实现的优化**：
- ✅ **历史分析器**（`HistoryAnalyzer`）：
  - 检测机器人是否陷入循环/打转
  - 提供逃脱动作序列
- ✅ 跌倒检测（保留原有功能）
- ✅ 动态障碍物等待策略

## 🚀 快速开始

### 环境要求

```bash
# 基础依赖
pip install opencv-python numpy

# 如果使用YOLOv12
pip install torch torchvision ultralytics
```

### 运行优化版本

```bash
# 1. 确保有YOLO模型文件
# YOLOv3-tiny: 需要 yolov3-tiny.cfg 和 yolov3-tiny.weights
# YOLOv12-nano: 会自动下载 yolov12n.pt

# 2. 修改配置（可选）
# 编辑 raspberrypi_bizhang_improved.py 中的 Config 类

# 3. 运行
python raspberrypi_bizhang_improved.py
```

### 模拟模式

如果没有硬件，程序会自动进入模拟模式，可以测试算法逻辑。

## 🔧 关键模块说明

### 数据类

- `Detection`：封装单个检测结果
- `SensorData`：统一传感器输入
- `FusionResult`：融合后的结果

### 核心类

| 类名 | 功能 |
|------|------|
| `UltrasonicFilter` | 超声波滤波 |
| `SimpleTracker` | 多目标跟踪 |
| `BayesianFusion` | 传感器融合 |
| `StateMachine` | 状态管理 |
| `HistoryAnalyzer` | 历史分析与循环检测 |
| `GlobalState` | 线程安全的全局状态 |

### 配置参数

在 `Config` 类中可调整：
- 超声波窗口大小
- 避障触发距离
- YOLO置信度阈值
- 相机焦距（用于距离估算）

## 📊 算法流程图

```
超声波 → 滤波 ─┐
                ├→ 贝叶斯融合 → 状态机 → 动作执行
  YOLO → 跟踪 ──┘
                ↑
            历史分析
```

## 🎓 技术亮点

1. **贝叶斯概率融合**：不是简单的硬编码规则，而是基于概率的软决策
2. **多目标跟踪**：可以同时处理多个障碍物
3. **动态障碍物感知**：通过轨迹分析区分动静态障碍
4. **防死循环机制**：记忆历史动作，检测并逃脱局部最优陷阱
5. **线程安全设计**：使用锁保护共享状态
6. **模拟模式**：无需硬件即可测试算法

## 📈 性能建议

1. **树莓派性能优化**：
   - 调整 `YOLO_FRAME_SKIP`（跳帧数）
   - 降低 `YOLO_IMGSZ`（输入分辨率）
   - 使用YOLOv12-nano比YOLOv3-tiny更快

2. **距离估算校准**：
   - 测量实际物体宽度，修改 `REAL_OBJECT_WIDTH`
   - 标定相机，调整 `CAMERA_FOCAL_LENGTH`

## 🔮 未来扩展方向

1. **路径规划增强**：
   - 实现DWA（动态窗口法）
   - 人工势场法
   - 同时定位与地图构建（SLAM）

2. **传感器扩展**：
   - 添加左右超声波
   - 集成IMU
   - 激光雷达（预算充足时）

3. **学习能力**：
   - 强化学习避障策略
   - 模仿学习

4. **多机器人协作**：
   - 机器人间通信
   - 协同避障

## 📝 注意事项

1. 原始版本 `raspberrypi_bizhang.py` 完全保留，可对比使用
2. 优化版本向后兼容原有动作组
3. 需要根据实际环境调整 `Config` 类参数
4. YOLO距离估算基于假设的物体宽度，实际使用建议标定

## 🤝 贡献建议

欢迎在以下方面进行改进：
- 更精细的状态机逻辑
- 更好的距离估计算法
- 更完善的测试用例
- 更多的边缘情况处理

---

**作者**：AI Assistant  
**版本**：1.0  
**日期**：2025
