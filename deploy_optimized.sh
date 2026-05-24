#!/bin/bash
# TonyPi 树莓派5优化版一键部署脚本

set -e

echo "======================================"
echo "TonyPi 树莓派5优化版部署脚本"
echo "======================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否为树莓派
check_raspberry_pi() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(tr -d '\0' < /proc/device-tree/model)
        if [[ $MODEL == *"Raspberry Pi"* ]]; then
            echo -e "${GREEN}检测到设备: $MODEL${NC}"
            return 0
        fi
    fi
    echo -e "${YELLOW}警告: 未检测到树莓派，将继续部署...${NC}"
    return 1
}

# 设置国内镜像源
setup_mirrors() {
    echo ""
    echo "步骤 1: 设置国内镜像源"
    echo "------------------------"
    
    # 备份原文件
    if [ -f /etc/apt/sources.list ]; then
        sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
    fi
    
    # 清华镜像源 (Debian Bookworm)
    sudo tee /etc/apt/sources.list > /dev/null << 'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
EOF
    
    echo -e "${GREEN}镜像源设置完成${NC}"
}

# 系统更新和依赖安装
install_system_deps() {
    echo ""
    echo "步骤 2: 更新系统并安装依赖"
    echo "----------------------------"
    
    sudo apt-get update -y
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        libopencv-dev \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        libqt5gui5 \
        libqt5webkit5 \
        libqt5test5 \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        curl
        
    echo -e "${GREEN}系统依赖安装完成${NC}"
}

# Python依赖安装
install_python_deps() {
    echo ""
    echo "步骤 3: 安装Python依赖"
    echo "-----------------------"
    
    # 设置pip镜像
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 升级pip
    pip3 install --upgrade pip
    
    # 安装基础依赖
    pip3 install numpy opencv-python
    
    # 安装性能监控依赖
    pip3 install psutil
    
    # 可选: 安装YOLOv12依赖 (需要时安装，比较大)
    echo ""
    read -p "是否安装YOLOv12依赖? (推荐用于树莓派5) [y/N]: " install_yolo
    if [[ $install_yolo == "y" || $install_yolo == "Y" ]]; then
        echo "正在安装YOLOv12依赖 (这可能需要一些时间)..."
        pip3 install torch torchvision ultralytics
        echo -e "${GREEN}YOLOv12依赖安装完成${NC}"
    fi
    
    echo -e "${GREEN}Python依赖安装完成${NC}"
}

# 系统配置优化
optimize_system() {
    echo ""
    echo "步骤 4: 系统性能优化"
    echo "---------------------"
    
    # 备份config.txt
    if [ -f /boot/firmware/config.txt ]; then
        sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.backup
    fi
    
    # 修改config.txt
    sudo tee -a /boot/firmware/config.txt > /dev/null << 'EOF'

# TonyPi优化配置
arm_boost=1
force_turbo=0
over_voltage=2
gpu_mem=128
dtparam=pciex1_gen=3
dtparam=usb_max_current_enable=1
dtparam=audio=off
EOF
    
    # 配置sysctl
    if [ -f /etc/sysctl.conf ]; then
        sudo cp /etc/sysctl.conf /etc/sysctl.conf.backup
    fi
    
    sudo tee -a /etc/sysctl.conf > /dev/null << 'EOF'

# TonyPi内存优化
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.dirty_background_ratio=5
vm.dirty_ratio=10
EOF
    
    # 应用sysctl配置
    sudo sysctl -p
    
    # 创建日志目录
    sudo mkdir -p /var/log/tonypi
    sudo chown pi:pi /var/log/tonypi
    
    echo -e "${GREEN}系统优化完成${NC}"
}

# 配置systemd服务
setup_service() {
    echo ""
    echo "步骤 5: 配置自启动服务"
    echo "------------------------"
    
    SCRIPT_PATH=$(realpath raspberrypi_bizhang_optimized.py)
    
    sudo tee /etc/systemd/system/tonypi.service > /dev/null << EOF
[Unit]
Description=TonyPi Optimized Robot Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $SCRIPT_PATH
Restart=always
RestartSec=10
StandardOutput=append:/var/log/tonypi/service.log
StandardError=append:/var/log/tonypi/error.log

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable tonypi.service
    
    echo ""
    echo "服务配置完成!"
    echo "使用以下命令管理服务:"
    echo "  启动: sudo systemctl start tonypi"
    echo "  停止: sudo systemctl stop tonypi"
    echo "  状态: sudo systemctl status tonypi"
    echo "  日志: journalctl -u tonypi -f"
}

# 主菜单
main_menu() {
    echo "请选择部署选项:"
    echo "1) 完整部署 (推荐)"
    echo "2) 仅安装Python依赖"
    echo "3) 仅系统优化"
    echo "4) 仅配置服务"
    echo "5) 退出"
    echo ""
    read -p "请输入选项 [1-5]: " choice
    
    case $choice in
        1)
            check_raspberry_pi
            setup_mirrors
            install_system_deps
            install_python_deps
            optimize_system
            setup_service
            echo ""
            echo -e "${GREEN}======================================"
            echo "部署完成!"
            echo "======================================${NC}"
            echo ""
            echo "下一步:"
            echo "1) 重启树莓派以应用系统优化"
            echo "2) 编辑 raspberrypi_bizhang_optimized.py 配置YOLO"
            echo "3) 启动服务: sudo systemctl start tonypi"
            echo ""
            ;;
        2)
            install_python_deps
            ;;
        3)
            optimize_system
            ;;
        4)
            setup_service
            ;;
        5)
            echo "退出"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            main_menu
            ;;
    esac
}

# 检查脚本是否以root运行
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}请不要以root用户运行此脚本${NC}"
    echo "使用普通用户运行，需要时会自动请求sudo权限"
    exit 1
fi

# 显示主菜单
main_menu
