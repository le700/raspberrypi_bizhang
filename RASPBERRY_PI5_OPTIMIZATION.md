# TonyPi 树莓派5 性能优化和部署方案

---

## 📚 目录

1. [树莓派5硬件特性分析](#1-树莓派5硬件特性分析)
2. [树莓派5系统配置优化](#2-树莓派5系统配置优化)
3. [YOLO推理速度优化](#3-yolo推理速度优化)
4. [内存管理优化](#4-内存管理优化)
5. [部署流程优化](#5-部署流程优化)
6. [调试和监控系统](#6-调试和监控系统)

---

## 1. 树莓派5硬件特性分析

### 1.1 核心规格对比

| 特性 | 树莓派4 | 树莓派5 | 提升 |
|------|---------|---------|------|
| CPU | 4×Cortex-A76 1.5GHz | 4×Cortex-A76 2.4GHz | +60% |
| 内存 | LPDDR4-3200 | LPDDR4X-4267 | +33% |
| GPU | VideoCore VI 500MHz | VideoCore VII 800MHz | +60% |
| PCIe | 2.0 ×1 | 2.0 ×2 | +100% |
| 接口 | USB 3.0 ×2 | USB 3.2 ×2 | - |
| 功耗 | 4W | 8W | - |
| 散热 | 被动散热 | 主动散热推荐 | - |

### 1.2 关键优化潜力

- **算力提升**：约 2-3 倍性能
- **内存带宽**：提升 33%
- **IO性能**：提升 100%
- **实际应用场景**：
  - YOLO推理从 5-10 FPS → 15-25 FPS
  - 多线程处理更流畅
  - 支持更高分辨率的图像处理

---

## 2. 树莓派5系统配置优化

### 2.1 BIOS/配置优化

#### 2.1.1 系统配置文件 `/boot/firmware/config.txt

```bash
# 编辑配置文件
sudo nano /boot/firmware/config.txt
```

**推荐配置：

```ini
# 超频配置
arm_boost=1
force_turbo=1
over_voltage=3

# 内存分配 (给GPU分配更多内存给CPU使用
gpu_mem=128

# PCIe Gen3.0启用
dtparam=pciex1_gen=3

# USB 3.0 完整启用
dtparam=usb_max_current_enable=1

# 禁用不必要的功能
dtparam=audio=off
dtparam=act_led_trigger=none
dtparam=pwr_led_trigger=default-on

# 启用I2C/SPI保持开启（机器人需要）
dtparam=i2c_arm=on
dtparam=spi=on
```

#### 2.1.2 系统服务优化

```bash
# 禁用不必要的服务
sudo systemctl disable bluetooth.service
sudo systemctl disable avahi-daemon.service
sudo systemctl disable triggerhappy.service
sudo systemctl disable cups.service
sudo systemctl disable ModemManager.service
sudo systemctl disable avahi-daemon.socket
```

#### 2.1.3 内存优化

```bash
# 编辑 /etc/sysctl.conf
sudo nano /etc/sysctl.conf
```

添加以下内容：

```ini
# 减少swappiness
vm.swappiness=10

# 增加文件系统缓存优化
vm.vfs_cache_pressure=50

# 减少dirty writeback
vm.dirty_background_ratio=5
vm.dirty_ratio=10
```

应用配置：
```bash
sudo sysctl -p
```

### 2.2 散热管理优化

#### 2.2.1 安装温度监控

```bash
# 查看当前温度
vcgencmd measure_temp
```

#### 2.2.2 安装散热监控脚本

```bash
# 安装温度阈值设置
echo "thermal-70" > /sys/class/thermal/thermal_zone0/trip_point_0_temp
```

#### 2.2.3 散热片和风扇控制

如果你有主动散热风扇：

```bash
# 风扇控制配置
sudo tee /etc/systemd/system/fan-control.service << 'EOF'
[Unit]
Description=Fan Control Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/fan-control.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

### 2.3 网络优化

```bash
# 禁用IPv6
sudo tee -a /etc/sysctl.conf << 'EOF'
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
EOF
```

---

## 3. YOLO推理速度优化

### 3.1 模型选择和量化

#### 3.1.1 使用NCNN推理引擎

```bash
# 安装NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j4
sudo make install
```

#### 3.1.2 模型量化

使用ONNX Runtime

```python
# 量化脚本
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov12n.pt')

# 导出为ONNX
model.export(format='onnx', imgsz=320, opset=12)
```

#### 3.1.3 TensorRT优化（如果有）

```python
model.export(format='engine', imgsz=320, workspace=4)
```

### 3.2 推理优化配置

在代码中的优化配置：

```python
# YOLOv12优化配置
YOLO_IMGSZ = 320  # 输入尺寸，更小更快
YOLO_DEVICE = 'cpu'
YOLO_CONFIDENCE_THRESHOLD = 0.4  # 降低阈值提高速度
YOLO_FRAME_SKIP = 1  # 跳帧
YOLO_MAX_DET = 5  # 最大检测数限制
```

### 3.3 OpenCV DNN 优化：

```python
# OpenCV DNN 后端选择
YOLO_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
YOLO_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

---

## 4. 内存管理优化

### 4.1 内存监控

```bash
# 查看内存使用
free -h
```

### 4.2 内存优化技术

#### 4.2.1 图像预处理优化

```python
# 使用 numpy 内存优化
frame = cv2.resize(frame, (320, 240))
```

#### 4.2.2 对象池

```python
from collections import deque
frame_pool = deque(maxlen=10)
```

#### 4.2.3 垃圾回收

```python
import gc
gc.collect()
```

---

## 5. 部署流程优化

### 5.1 自动化部署脚本

一键部署脚本将在 `deploy_optimized.sh` 中提供

### 5.2 systemd 服务配置

自动启动机器人程序：

```bash
sudo tee /etc/systemd/system/tonypi.service << 'EOF'
[Unit]
Description=TonyPi Robot Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/TonyPi
ExecStart=/usr/bin/python3 /home/pi/TonyPi/raspberrypi_bizhang_optimized.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tonypi.service
sudo systemctl start tonypi.service
```

---

## 6. 调试和监控系统

### 6.1 性能监控仪表板

- 实时FPS监控
- 内存使用监控
- 温度监控
- 推理延迟监控

### 6.2 日志系统

- 系统日志将存储在 `/var/log/tonypi/

### 6.3 Web监控界面（可选）

---

## 快速开始

### 一键优化安装：

```bash
chmod +x deploy_optimized.sh
./deploy_optimized.sh
```
