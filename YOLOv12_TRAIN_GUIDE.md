# YOLOv12-nano 数据集训练完整教程（中文版）

## 📚 目录

1. [准备工作](#一准备工作)
2. [数据集准备](#二数据集准备)
3. [数据标注](#三数据标注)
4. [训练模型](#四训练模型)
5. [模型转换](#五模型转换)
6. [模型移植到树莓派](#六模型移植到树莓派)
7. [常见问题](#七常见问题)

---

## 一、准备工作

### 1.1 硬件要求

- **训练用电脑/服务器**：需要GPU，推荐NVIDIA显卡，显存4GB以上
- **树莓派**：用于运行训练好的模型

### 1.2 软件环境 - 推荐用Anaconda（Windows/Mac/Linux）

#### Windows系统安装Anaconda

1. **下载地址**（清华大学镜像，速度快）：
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

2. **安装步骤**：
   - 下载对应版本的 `Anaconda3-xxxx.x-Windows-x86_64.exe`
   - 双击安装，一路下一步
   - 勾选"Add Anaconda to PATH"（很重要！）

#### Mac系统安装Anaconda

1. **下载地址**：
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

2. **安装步骤**：
   - 下载 `Anaconda3-xxxx.x-MacOSX-x86_64.pkg`
   - 双击安装，一路下一步

### 1.3 创建训练环境

打开 **Anaconda Prompt**（Windows）或 **Terminal**（Mac/Linux），执行：

```bash
# 创建名为 yolo_train 的环境，Python版本3.10
conda create -n yolo_train python=3.10

# 激活环境
conda activate yolo_train
```

### 1.4 安装PyTorch（用国内镜像）

**Windows/Mac/Linux通用命令**：

```bash
# 添加清华镜像源（国内必选！）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装PyTorch CPU版本（没有NVIDIA显卡用这个）
pip install torch torchvision

# 或者安装GPU版本（有NVIDIA显卡用这个，速度快很多）
# CUDA 11.8版本
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本（推荐最新显卡）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

### 1.5 安装Ultralytics（YOLOv12官方库）

```bash
pip install ultralytics
```

### 1.6 验证安装

```bash
python
>>> import torch
>>> import ultralytics
>>> print('PyTorch:', torch.__version__)
>>> print('Ultralytics:', ultralytics.__version__)
>>> print('CUDA可用:', torch.cuda.is_available())
```

应该看到类似输出：
```
PyTorch: 2.x.x
Ultralytics: 8.x.x
CUDA可用: True/False
```

---

## 二、数据集准备

### 2.1 创建文件夹结构

在任意位置创建以下文件夹结构：

```
yolo_dataset/
├── images/          # 存放图片
│   ├── train/       # 训练集图片（80%的图片放这里）
│   └── val/         # 验证集图片（20%的图片放这里）
├── labels/          # 存放标签（和图片一一对应）
│   ├── train/       # 训练集标签
│   └── val/         # 验证集标签
└── dataset.yaml     # 配置文件
```

### 2.2 收集图片

**建议**：
- 收集你想检测的物体图片，至少 **100-200张**
- 图片要多样化：不同角度、不同光照、不同背景
- 手机拍摄的照片完全可以

**存放位置**：
- 训练集：`yolo_dataset/images/train/` 
- 验证集：`yolo_dataset/images/val/`

### 2.3 创建数据集配置文件

创建文件 `yolo_dataset/dataset.yaml`：

```yaml
# 训练集图片路径
train: ./images/train

# 验证集图片路径
val: ./images/val

# 类别数量（根据你的需求填写）
nc: 3

# 类别名称（按顺序写）
names: ['person', 'car', 'obstacle']
```

**注意**：
- `nc` 的数字要和 `names` 的数量一致
- names 中的名称顺序很重要，从0开始编号

---

## 三、数据标注

### 3.1 安装标注工具 LabelImg

```bash
pip install labelImg
```

### 3.2 启动标注工具

**Windows**：
```bash
# 在Anaconda Prompt中
labelImg
```

**Mac/Linux**：
```bash
# 在终端中
labelImg
```

### 3.3 标注步骤（图解）

**第一步：设置路径**
1. 点击左侧 **Open Dir** 按钮
2. 选择 `yolo_dataset/images/train` 文件夹
3. 点击 **Change Save Dir** 按钮
4. 选择 `yolo_dataset/labels/train` 文件夹

**第二步：设置标注格式**
1. 点击左侧 **PascalVOC** 按钮
2. 选择 **YOLO** 格式（重要！）

**第三步：开始标注**
1. 按 `W` 键，开始画框
2. 鼠标拖动，框住目标
3. 弹出窗口选择类别（或输入新类别名）
4. 按 `D` 键下一张，按 `A` 键上一张
5. 按 `Ctrl+S` 保存（或开启自动保存）

**标注技巧**：
- 框要紧贴目标，不要留太多空白
- 一张图可以有多个目标
- 同一类别的物体用同一个名字

### 3.4 标注结果示例

标注完成后，在 `labels/train/` 文件夹中会生成 `.txt` 文件：

文件内容格式：
```
类别编号 x_center y_center width height
```

例如 `cat_001.txt` 内容：
```
0 0.5 0.5 0.3 0.2
1 0.7 0.3 0.1 0.1
```

含义：
- 第1行：`0` 号类别，中心在(0.5, 0.5)，宽高0.3×0.2
- 第2行：`1` 号类别，中心在(0.7, 0.3)，宽高0.1×0.1

**注意**：数值都是相对于图片宽高的比例，范围是0-1！

---

## 四、训练模型

### 4.1 下载预训练权重

```bash
# 创建 weights 文件夹
mkdir weights
cd weights

# 下载 YOLOv12n（nano版本，最轻量，推荐！）
wget https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt

# 或者下载其他版本
# YOLOv12s（small版本）
wget https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12s.pt
```

### 4.2 训练代码

创建训练脚本 `train.py`：

```python
from ultralytics import YOLO

# 加载预训练模型（使用YOLOv12n）
model = YOLO('weights/yolov12n.pt')

# 开始训练
results = model.train(
    data='yolo_dataset/dataset.yaml',  # 数据集配置
    epochs=100,                           # 训练轮数
    imgsz=320,                           # 图片大小（推荐320，更快！）
    batch=16,                            # 批次大小（根据显存调整，GPU用16-32，CPU用4-8）
    device=0,                            # GPU编号，CPU训练用 'cpu'
    workers=4,                           # 数据加载线程数
    patience=50,                         # 早停耐心值
    project='runs/train',                # 保存路径
    name='yolov12_obstacle',            # 实验名称
    exist_ok=True,                       # 覆盖已有结果
    pretrained=True,                     # 使用预训练权重
    optimizer='SGD',                     # 优化器
    lr0=0.01,                           # 初始学习率
    lrf=0.01,                           # 最终学习率
    momentum=0.937,                      # 动量
    weight_decay=0.0005,                 # 权重衰减
    save=True,                          # 保存模型
    cache=True,                          # 缓存图片（加快训练）
    val=True,                           # 验证
    plots=True,                          # 画图
    verbose=True,                        # 详细输出
)
```

### 4.3 开始训练

```bash
python train.py
```

### 4.4 训练过程说明

训练时会有进度条，显示：
- 当前epoch / 总epoch
- 损失值（box_loss, cls_loss, dfl_loss）
- 精度（precision）
- 召回率（recall）
- mAP50和mAP50-95

**训练时间**：
- GPU训练（RTX 3080+）：10-30分钟
- GPU训练（RTX 3060）：30-60分钟
- CPU训练：几小时到几天（**不推荐！**）

### 4.5 训练完成

训练完成后，模型保存在：
```
runs/train/yolov12_obstacle/weights/best.pt
runs/train/yolov12_obstacle/weights/last.pt
```

- `best.pt`：验证集上效果最好的模型
- `last.pt`：最后一轮的模型

---

## 五、模型转换

### 5.1 测试模型

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolov12_obstacle/weights/best.pt')

# 测试单张图片
results = model('test.jpg')

# 显示结果
results.show()
```

### 5.2 导出为ONNX格式（树莓派使用）

**YOLOv12推荐导出为ONNX格式**：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolov12_obstacle/weights/best.pt')

# 导出为ONNX格式（推荐！树莓派通用）
model.export(format='onnx', imgsz=320)

print("导出完成！生成文件：best.onnx")
```

**会生成文件**：`best.onnx`

### 5.3 其他导出格式（可选）

```python
# 导出为其他格式

# TorchScript格式
model.export(format='torchscript')

# TensorRT格式（需要TensorRT，精度最高）
model.export(format='engine', imgsz=320)

# CoreML格式（苹果设备用）
model.export(format='coreml')
```

---

## 六、模型移植到树莓派（详细步骤！）

### 6.1 先理清原理

**重要！**
- 电脑上训练出来的模型是 **`best.pt`**（PyTorch格式）
- 树莓派上用的是 **PyTorch**，**可以直接使用 `.pt` 格式！**
- **不需要转换格式！** YOLOv12支持直接加载.pt模型

**树莓派需要安装PyTorch，但不需要导出为ONNX！**

---

### 6.2 完整移植流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        电脑（训练环境）                           │
├─────────────────────────────────────────────────────────────────┤
│  收集图片 → 标注 → 训练 → best.pt                               │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
                              传到树莓派
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                        树莓派（运行环境）                          │
├─────────────────────────────────────────────────────────────────┤
│        pip install torch ultralytics → 直接用best.pt！           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6.3 第一步：树莓派安装PyTorch（关键！）

在树莓派上执行：

```bash
# 设置pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装PyTorch（CPU版本）
pip install torch torchvision

# 安装Ultralytics
pip install ultralytics
```

**安装时间**：5-15分钟（树莓派5）

---

### 6.4 第二步：把best.pt传到树莓派

把训练好的模型传到树莓派的项目目录：

```
raspberrypi_bizhang/          ← 你的项目目录
├── raspberrypi_bizhang.py    ← 主程序
└── best.pt                   ← 从训练电脑传过来 ⭐
```

**传输方法**：
- Windows/Mac：用 SCP 或 WinSCP 软件
- 命令行方式（从电脑传到树莓派）：
```bash
scp best.pt pi@192.168.x.x:/home/pi/raspberrypi_bizhang/
```

---

### 6.5 第三步：修改树莓派代码

在树莓派上编辑 `raspberrypi_bizhang.py`：

找到YOLO加载部分，修改为：

```python
# 使用YOLOv12（直接加载.pt文件！）
YOLO_MODEL = YOLO('best.pt')

# 确保使用YOLOv12版本
YOLO_VERSION = 'v12n'
```

---

### 6.6 第四步：运行测试

在树莓派上运行：

```bash
python raspberrypi_bizhang.py
```

应该就能正常运行了！

---

### 6.7 常见移植问题

#### Q1: 树莓派上需要导出为ONNX吗？

**答**：不需要！YOLOv12可以直接加载.pt文件！但如果你想导出也可以：

```python
model.export(format='onnx')
```

#### Q2: 树莓派内存不够怎么办？

**答**：
1. 减小输入尺寸：
```python
YOLO_IMGSZ = 320
```

2. 使用更小的batch：
```python
# 在推理时
model('test.jpg', imgsz=320, verbose=False)
```

#### Q3: 树莓派上PyTorch安装失败？

**答**：确保使用正确的Python版本：

```bash
# 检查Python版本
python --version  # 应该是3.8-3.11

# 如果版本不对，创建新环境
conda create -n yolo_pi python=3.10
conda activate yolo_pi
```

---

## 七、常见问题

### Q1: 训练很慢怎么办？

**答**：
1. **用GPU训练**（最重要！），速度快10-100倍
2. 减小 `batch` 值（显存不够就调小）
3. 减小 `imgsz` 值（从320改成256）
4. 用Google Colab免费GPU

### Q2: 标注软件打不开？

**答**：
```bash
# Windows
pip install pyqt5 lxml
labelImg

# Mac
pip install pyqt5 lxml
python -m labelImg

# Linux
sudo apt-get install pyqt5-dev-tools
pip install pyqt5 lxml pyqtgraph
labelImg
```

### Q3: 标注格式怎么选？

**答**：
- **YOLO格式**（推荐）：数值是0-1的比例，适合YOLO训练
- **PascalVOC格式**：数值是像素坐标，需要转换

### Q4: 标注的图片不够怎么办？

**答**：
1. 网上下载公开数据集（如COCO）
2. 对现有图片做数据增强（YOLO训练时会自动增强）
3. 合成图片（用Python程序自动生成）

### Q5: 怎么提高检测精度？

**答**：
1. 增加训练数据量（至少100-200张）
2. 数据要多样化
3. 增加训练轮数（epochs=200）
4. 使用YOLOv12s而不是YOLOv12n（更大更准）

### Q6: YOLOv12和YOLOv8/YOLOv3哪个好？

**答**：**YOLOv12是最新最强的！**

| 版本 | 模型大小 | 速度 | 精度 | 推荐 |
|------|---------|------|------|------|
| YOLOv12n | 2.5MB | 最快 | 中等 | ⭐树莓派推荐 |
| YOLOv12s | 5MB | 快 | 较高 | 电脑推荐 |
| YOLOv8n | 6MB | 快 | 较高 | 备选 |
| YOLOv3-tiny | 33MB | 慢 | 中等 | 不推荐 |

---

## 八、快速开始（简化版）

如果觉得上面太复杂，用这个简化流程：

### Step 1: 环境搭建
```bash
conda create -n yolo_train python=3.10
conda activate yolo_train
pip install torch torchvision ultralytics
```

### Step 2: 准备数据
- 把图片放到 `images/train/` 和 `images/val/`
- 用LabelImg标注，保存到 `labels/train/` 和 `labels/val/`
- 创建 `dataset.yaml`

### Step 3: 一键训练
```python
from ultralytics import YOLO
model = YOLO('yolov12n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=320)
```

### Step 4: 移植到树莓派
```bash
# 树莓派安装
pip install torch torchvision ultralytics

# 传到树莓派
scp best.pt pi@ip:/path/to/project/

# 运行
python raspberrypi_bizhang.py
```

---

## 九、参考资料

- [Ultralytics YOLOv12官方文档](https://docs.ultralytics.com/yolov12/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- [Google Colab免费GPU](https://colab.research.google.com/)
- [PyTorch官网](https://pytorch.org/)

---

## 十、版本对比总结

### YOLOv12 vs YOLOv3-tiny

| 对比项 | YOLOv12-nano | YOLOv3-tiny |
|--------|-------------|-------------|
| **模型大小** | 2.5MB | 33MB |
| **训练需要** | PyTorch | PyTorch |
| **树莓派运行** | 直接.pt | OpenCV DNN |
| **FPS（树莓派5）** | 15-25 | 5-10 |
| **精度** | 40.2% mAP | 33.1% mAP |
| **最新版本** | ✅ 是 | ❌ 否 |

**结论：YOLOv12-nano更好！推荐使用！**

---

**祝训练成功！🎉**
