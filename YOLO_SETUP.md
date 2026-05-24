# 摄像头 + YOLOv3-tiny + 超声波 避障系统使用说明（中国大陆优化版）

---

## 📚 目录

1. [系统架构](#一系统架构)
2. [为什么用 YOLOv3-tiny？](#二为什么用-yolov3-tiny)
3. [功能特性](#三功能特性)
4. [安装步骤（中国大陆优化！）](#四安装步骤中国大陆优化)
5. [使用方法](#五使用方法)
6. [性能优化建议](#六性能优化建议)
7. [工作流程](#七工作流程)
8. [变量说明](#八变量说明)
9. [如果性能还是不够？](#九如果性能还是不够)
10. [文件说明](#十文件说明)

---

## 一、系统架构

本系统在原有避障功能基础上，新增了 **YOLOv3-tiny** 目标检测功能，使用 **OpenCV DNN** 实现，**无需 PyTorch**，实现摄像头 + YOLO + 超声波 三模传感器融合的智能避障系统。

---

## 二、为什么用 YOLOv3-tiny？

1. **超轻量：** 只有 33MB，YOLOv8n 是 6MB 但需要 PyTorch（更重！）
2. **无需 PyTorch：** 仅依赖 OpenCV，安装简单
3. **树莓派友好：** OpenCV DNN 对 ARM 优化好
4. **足够用：** 对于避障场景完全够用

---

## 三、功能特性

1. **向后兼容：** 原有功能完全保留
2. **可配置启用/禁用 YOLO**
3. **传感器数据融合**
4. **智能避障方向选择**
5. **性能优化（跳帧检测）**

---

## 四、安装步骤（中国大陆优化！）

### 4.1 基础依赖安装（用国内镜像！）

#### 方式一：pip 安装（推荐）

```bash
# 先设置 pip 国内镜像（清华源，速度快！）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install opencv-python numpy
```

#### 方式二：树莓派用 apt 安装（更简单！）

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy
```

---

### 4.2 下载 YOLOv3-tiny 模型文件（中国大陆优化！）

需要下载两个文件放到同一目录：
- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`

**国内下载方式（速度快！）**

#### 方式一：wget 下载（推荐）

```bash
# 下载 yolov3-tiny.weights（国内镜像，GitHub代理加速）
wget https://ghproxy.com/https://github.com/pjreddie/darknet/releases/download/yolov3-tiny/yolov3-tiny.weights

# 或者用这个链接（网盘镜像）
# wget https://ghproxy.com/https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3-tiny.weights

# 下载 yolov3-tiny.cfg
wget https://ghproxy.com/https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg
```

#### 方式二：curl 下载

```bash
curl -L -o yolov3-tiny.weights https://ghproxy.com/https://github.com/pjreddie/darknet/releases/download/yolov3-tiny/yolov3-tiny.weights
curl -L -o yolov3-tiny.cfg https://ghproxy.com/https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg
```

#### 方式三：手动下载（最慢但最稳妥）

1. 访问这个链接下载 `.weights`：
   https://ghproxy.com/https://github.com/pjreddie/darknet/releases/download/yolov3-tiny/yolov3-tiny.weights

2. 访问这个链接下载 `.cfg`：
   https://ghproxy.com/https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg

3. 下载后传到树莓派同一目录

---

## 五、使用方法

### 5.1 启用 YOLO 功能

在 `raspberrypi_bizhang.py` 中找到：
```python
USE_YOLO = False  # 修改为 True 启用 YOLO
```

### 5.2 配置参数

```python
YOLO_CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
YOLO_FRAME_SKIP = 3  # 跳帧数（性能优化，数越大越快）
```

---

## 六、性能优化建议

### 6.1 树莓派性能优化

1. **调整跳帧数：** 增加 `YOLO_FRAME_SKIP` 数值（如 5-10）
2. **降低输入分辨率：** 修改代码中的 (416, 416) 为更小尺寸（如 320, 320）
3. **降低置信度阈值：** 可以加快处理速度
4. **使用树莓派 64位系统：** 性能更好

### 6.2 避障优化

- 数据融合权重可调
- 距离估计算法可自定义
- 避障方向智能选择

---

## 七、工作流程

1. 摄像头采集图像 → YOLOv3-tiny 检测目标 → 估算目标距离
2. 超声波测距
3. 数据融合
4. 智能决策避障
5. 执行动作

---

## 八、变量说明（原有变量完全保留！）

所有原有变量名和动作片名称保持不变：

### 核心变量
- `current_step`
- `obstacle_count`
- `distance`
- `goforward`
- `stop_patrol`

### 动作片（未添加新动作！）
- `stand_up_front`
- `turn_left`
- `go_forward_fast`
- `turn_right`
- `zhixing4`
- `stand_slow`

---

## 九、如果性能还是不够？

如果 YOLOv3-tiny 在树莓派上还是卡，可以：

1. **进一步增加跳帧数**
2. **只检测特定类别**（比如只检测人）
3. **甚至可以关闭 YOLO，只用超声波 + 视觉巡线**

---

## 十、文件说明

- `raspberrypi_bizhang.py` - 主程序
- `requirements.txt` - 依赖列表
- `yolov3-tiny.cfg` - 模型配置（需下载）
- `yolov3-tiny.weights` - 模型权重（需下载）
- `YOLO_TRAIN_GUIDE.md` - 完整训练教程（中文）

---

## 快速开始（3步搞定！）

```bash
# Step 1: 安装依赖（用国内镜像）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python numpy

# Step 2: 下载模型（用国内镜像）
wget https://ghproxy.com/https://github.com/pjreddie/darknet/releases/download/yolov3-tiny/yolov3-tiny.weights
wget https://ghproxy.com/https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg

# Step 3: 修改代码启用YOLO
# 编辑 raspberrypi_bizhang.py，把 USE_YOLO = False 改成 True

# 运行！
python raspberrypi_bizhang.py
```

---

## 网络问题解决

如果以上下载都慢，可以尝试：

1. **用手机热点**
2. **挂代理**（如果有的话）
3. **在电脑上下载后传到树莓派**
4. **使用其他国内镜像源**

---

**有问题看 YOLO_TRAIN_GUIDE.md！祝使用愉快！🎉**
