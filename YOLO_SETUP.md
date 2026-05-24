# 摄像头 + YOLOv3-tiny + 超声波 避障系统使用说明

## 系统架构

本系统在原有避障功能基础上，新增了 **YOLOv3-tiny** 目标检测功能，使用 **OpenCV DNN** 实现，**无需 PyTorch**，实现摄像头 + YOLO + 超声波 三模传感器融合的智能避障系统。

## 为什么用 YOLOv3-tiny？

1. **超轻量：** 只有 33MB，YOLOv8n 是 6MB 但需要 PyTorch（更重！
2. **无需 PyTorch：** 仅依赖 OpenCV，安装简单
3. **树莓派友好：** OpenCV DNN 对 ARM 优化好
4. **足够用：** 对于避障场景完全够用

## 功能特性

1. **向后兼容：原有功能完全保留
2. **可配置启用/禁用 YOLO
3. **传感器数据融合
4. **智能避障方向选择
5. **性能优化（跳帧检测）

## 安装步骤

### 1. 基础依赖安装

```bash
pip install -r requirements.txt
```

### 2. 下载 YOLOv3-tiny 模型文件

需要下载两个文件放到同一目录：

- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`

下载方式：
```bash
# 方法1：使用 wget 下载
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg

# 方法2：手动下载
# 访问 https://pjreddie.com/darknet/yolo/ 下载
```

## 使用方法

### 启用 YOLO 功能

在 `raspberrypi_bizhang.py` 中找到：
```python
USE_YOLO = False  # 修改为 True 启用 YOLO
```

### 配置参数

- `YOLO_CONFIDENCE_THRESHOLD = 0.5`  # 检测置信度阈值
- `YOLO_FRAME_SKIP = 3`  # 跳帧数（性能优化

## 性能优化建议

### 树莓派性能优化

1. **调整跳帧数：** 增加 `YOLO_FRAME_SKIP` 数值（如 5-10）
2. **降低输入分辨率：** 修改代码中的 (416, 416) 为更小尺寸（如 320, 320）
3. **降低置信度阈值：** 可以加快处理速度
4. **使用树莓派 64位系统：** 性能更好

### 避障优化

- 数据融合权重可调
- 距离估计算法可自定义
- 避障方向智能选择

## 工作流程

1. 摄像头采集图像 → YOLOv3-tiny 检测目标 → 估算目标距离
2. 超声波测距
3. 数据融合
4. 智能决策避障
5. 执行动作

## 变量说明（原有变量完全保留！

所有原有变量名和动作片名称保持不变：

### 核心变量
- `current_step`
- `obstacle_count`
- `distance`
- `goforward`
- `stop_patrol`

### 动作片（未添加新动作！
- `stand_up_front`
- `turn_left`
- `go_forward_fast`
- `turn_right`
- `zhixing4`
- `stand_slow`

## 新增变量

新增变量用于增强功能，不影响原有逻辑。

## 如果性能还是不够？

如果 YOLOv3-tiny 在树莓派上还是卡，可以：

1. **进一步增加跳帧数**
2. **只检测特定类别**（比如只检测人
3. **甚至可以关闭 YOLO，只用超声波 + 视觉巡线**

## 文件说明

- `raspberrypi_bizhang.py` - 主程序
- `requirements.txt` - 依赖列表
- `yolov3-tiny.cfg` - 模型配置（需下载）
- `yolov3-tiny.weights` - 模型权重（需下载）
