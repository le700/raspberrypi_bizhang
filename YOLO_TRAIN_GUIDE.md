# YOLOv3-tiny 数据集训练完整教程（中文版）

## 📚 目录

1. [准备工作](#一准备工作)
2. [数据集准备](#二数据集准备)
3. [数据标注](#三数据标注)
4. [训练模型](#四训练模型)
5. [转换模型](#五转换模型)
6. [常见问题](#六常见问题)

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
# 创建名为 yolo_train 的环境，Python版本3.8
conda create -n yolo_train python=3.8

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
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### 1.5 安装Ultralytics（YOLO官方库）

```bash
pip install ultralytics
```

### 1.6 验证安装

```bash
python
>>> import torch
>>> import ultralytics
>>> print(torch.__version__)
>>> print(ultralytics.__version__)
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

### 4.1 下载预训练权重（推荐）

```bash
# 创建 weights 文件夹
mkdir weights
cd weights

# 下载 YOLOv3-tiny 预训练权重
wget https://github.com/ultralytics/yolov3/releases/download/yolov3-tiny/yolov3-tiny.pt

# 或者下载完整版 YOLOv3
wget https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt
```

### 4.2 训练代码

创建训练脚本 `train.py`：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('weights/yolov3-tiny.pt')

# 开始训练
results = model.train(
    data='yolo_dataset/dataset.yaml',  # 数据集配置
    epochs=100,                           # 训练轮数
    imgsz=416,                           # 图片大小
    batch=8,                             # 批次大小（根据显存调整）
    device=0,                            # GPU编号，CPU训练用 'cpu'
    workers=4,                           # 数据加载线程数
    patience=50,                         # 早停耐心值
    project='runs/train',                # 保存路径
    name='yolo_obstacle',                # 实验名称
    exist_ok=True,                       # 覆盖已有结果
    pretrained=True,                     # 使用预训练权重
    optimizer='SGD',                     # 优化器
    lr0=0.01,                            # 初始学习率
    lrf=0.01,                            # 最终学习率
    momentum=0.937,                      # 动量
    weight_decay=0.0005,                 # 权重衰减
    save=True,                           # 保存模型
    cache=True,                          # 缓存图片（加快训练）
    val=True,                            # 验证
    plots=True,                          # 画图
)
```

### 4.3 开始训练

```bash
python train.py
```

### 4.4 训练过程说明

训练时会有进度条，显示：
- 当前epoch / 总epoch
- 损失值（loss）
- 精度（precision）
- 召回率（recall）
- mAP（平均精度）

**训练时间**：
- GPU训练：几分钟到几小时
- CPU训练：几小时到几天

### 4.5 训练完成

训练完成后，模型保存在：
```
runs/train/yolo_obstacle/weights/best.pt
runs/train/yolo_obstacle/weights/last.pt
```

- `best.pt`：验证集上效果最好的模型
- `last.pt`：最后一轮的模型

---

## 五、转换模型

### 5.1 测试模型

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolo_obstacle/weights/best.pt')

# 测试单张图片
results = model('test.jpg')

# 显示结果
results.show()
```

### 5.2 导出为 OpenCV DNN 可用格式

**方法1：导出为 ONNX 格式**（推荐）

```python
# 导出为ONNX
model.export(format='onnx')
```

生成文件：`best.onnx`

**方法2：导出为 Darknet 格式**

```python
# 导出为Darknet（.weights + .cfg）
model.export(format='darknet')
```

生成文件：`best.weights` 和 `best.cfg`

---

## 六、常见问题

### Q1: 训练很慢怎么办？

**答**：
1. 用GPU训练，速度快10-100倍
2. 减小 `batch` 值（显存不够就调小）
3. 减小 `imgsz` 值（从416改成320）
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
2. 对现有图片做数据增强（旋转、缩放、裁剪）
3. 合成图片（用Python程序自动生成）

### Q5: 怎么提高检测精度？

**答**：
1. 增加训练数据量（至少100-200张）
2. 数据要多样化
3. 增加训练轮数（epochs）
4. 调整学习率

---

## 七、快速开始（简化版）

如果觉得上面太复杂，用这个简化流程：

### Step 1: 环境搭建
```bash
conda create -n yolo_train python=3.8
conda activate yolo_train
pip install ultralytics torch torchvision
```

### Step 2: 准备数据
- 把图片放到 `images/train/` 和 `images/val/`
- 用LabelImg标注，保存到 `labels/train/` 和 `labels/val/`
- 创建 `dataset.yaml`

### Step 3: 一键训练
```python
from ultralytics import YOLO
model = YOLO('yolov3-tiny.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=416)
```

### Step 4: 使用模型
```python
model = YOLO('runs/train/exp/weights/best.pt')
model('test.jpg').show()
```

---

## 八、参考资料

- [Ultralytics YOLOv3官方文档](https://docs.ultralytics.com/yolov3/)
- [YOLO官网](https://pjreddie.com/darknet/yolo/)
- [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- [Google Colab免费GPU](https://colab.research.google.com/)

---

**祝训练成功！🎉**
