#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TonyPi 树莓派5监控工具
提供实时性能监控和日志查看功能
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil未安装，部分功能不可用")

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

# 颜色定义
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_cpu_temp():
    """获取CPU温度"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 0.0

def get_system_info():
    """获取系统信息"""
    info = {}
    
    # CPU信息
    if PSUTIL_AVAILABLE:
        info['cpu_percent'] = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info['cpu_freq'] = cpu_freq.current
        else:
            info['cpu_freq'] = 0
    else:
        info['cpu_percent'] = 0
        info['cpu_freq'] = 0
    
    # 温度
    info['cpu_temp'] = get_cpu_temp()
    
    # 内存信息
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info['mem_total'] = mem.total / 1024 / 1024
        info['mem_used'] = mem.used / 1024 / 1024
        info['mem_percent'] = mem.percent
    else:
        info['mem_total'] = 0
        info['mem_used'] = 0
        info['mem_percent'] = 0
    
    # 磁盘信息
    if PSUTIL_AVAILABLE:
        disk = psutil.disk_usage('/')
        info['disk_total'] = disk.total / 1024 / 1024 / 1024
        info['disk_used'] = disk.used / 1024 / 1024 / 1024
        info['disk_percent'] = disk.percent
    else:
        info['disk_total'] = 0
        info['disk_used'] = 0
        info['disk_percent'] = 0
    
    return info

def check_service_status():
    """检查TonyPi服务状态"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'tonypi'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except:
        return "unknown"

def get_recent_logs(lines=20):
    """获取最近的日志"""
    log_files = [
        '/var/log/tonypi/service.log',
        '/var/log/tonypi/error.log',
        '/var/log/tonypi.log'
    ]
    
    logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines_list = f.readlines()
                    recent = lines_list[-lines:] if len(lines_list) > lines else lines_list
                    logs.append(f"=== {log_file} ===")
                    logs.extend(recent)
            except:
                pass
    
    return logs

def print_simple_monitor():
    """简单的监控界面"""
    while True:
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        info = get_system_info()
        service_status = check_service_status()
        
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}TonyPi 树莓派5监控工具{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}系统状态:{Colors.ENDC}")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  服务状态: {Colors.GREEN if service_status == 'active' else Colors.FAIL}{service_status}{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}CPU信息:{Colors.ENDC}")
        print(f"  使用率: {info['cpu_percent']:.1f}%")
        print(f"  频率: {info['cpu_freq']:.0f} MHz")
        temp_color = Colors.GREEN if info['cpu_temp'] < 60 else Colors.WARNING if info['cpu_temp'] < 75 else Colors.FAIL
        print(f"  温度: {temp_color}{info['cpu_temp']:.1f}°C{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}内存信息:{Colors.ENDC}")
        print(f"  已用: {info['mem_used']:.1f} MB / {info['mem_total']:.1f} MB")
        mem_color = Colors.GREEN if info['mem_percent'] < 70 else Colors.WARNING if info['mem_percent'] < 85 else Colors.FAIL
        print(f"  使用率: {mem_color}{info['mem_percent']:.1f}%{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}磁盘信息:{Colors.ENDC}")
        print(f"  已用: {info['disk_used']:.1f} GB / {info['disk_total']:.1f} GB")
        disk_color = Colors.GREEN if info['disk_percent'] < 70 else Colors.WARNING if info['disk_percent'] < 85 else Colors.FAIL
        print(f"  使用率: {disk_color}{info['disk_percent']:.1f}%{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}最近日志:{Colors.ENDC}")
        logs = get_recent_logs(10)
        if logs:
            for line in logs[-10:]:
                print(f"  {line.rstrip()}")
        else:
            print("  暂无日志")
        print()
        
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print("按 Ctrl+C 退出")
        print()
        
        time.sleep(2)

def print_menu():
    """打印菜单"""
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}TonyPi 管理工具{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print()
    print("1. 实时监控")
    print("2. 查看服务状态")
    print("3. 启动服务")
    print("4. 停止服务")
    print("5. 重启服务")
    print("6. 查看日志")
    print("7. 系统信息")
    print("0. 退出")
    print()

def view_service_status():
    """查看服务状态"""
    print(f"{Colors.HEADER}服务状态:{Colors.ENDC}")
    status = check_service_status()
    color = Colors.GREEN if status == 'active' else Colors.FAIL
    print(f"TonyPi服务: {color}{status}{Colors.ENDC}")
    print()
    
    try:
        result = subprocess.run(
            ['systemctl', 'status', 'tonypi', '--no-pager', '-n', '10'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except:
        print("无法获取详细状态")
    print()

def start_service():
    """启动服务"""
    print("正在启动TonyPi服务...")
    try:
        subprocess.run(['sudo', 'systemctl', 'start', 'tonypi'], check=True)
        print(f"{Colors.GREEN}服务启动成功{Colors.ENDC}")
    except:
        print(f"{Colors.FAIL}服务启动失败{Colors.ENDC}")
    print()

def stop_service():
    """停止服务"""
    print("正在停止TonyPi服务...")
    try:
        subprocess.run(['sudo', 'systemctl', 'stop', 'tonypi'], check=True)
        print(f"{Colors.GREEN}服务停止成功{Colors.ENDC}")
    except:
        print(f"{Colors.FAIL}服务停止失败{Colors.ENDC}")
    print()

def restart_service():
    """重启服务"""
    print("正在重启TonyPi服务...")
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'tonypi'], check=True)
        print(f"{Colors.GREEN}服务重启成功{Colors.ENDC}")
    except:
        print(f"{Colors.FAIL}服务重启失败{Colors.ENDC}")
    print()

def view_logs():
    """查看日志"""
    print(f"{Colors.HEADER}查看日志:{Colors.ENDC}")
    print("1. 服务日志")
    print("2. 错误日志")
    print("3. 所有日志")
    print("4. 返回")
    print()
    
    choice = input("请选择: ").strip()
    
    log_file = None
    if choice == '1':
        log_file = '/var/log/tonypi/service.log'
    elif choice == '2':
        log_file = '/var/log/tonypi/error.log'
    elif choice == '3':
        try:
            subprocess.run(['journalctl', '-u', 'tonypi', '-f'])
        except:
            pass
        return
    
    if log_file and os.path.exists(log_file):
        try:
            subprocess.run(['tail', '-f', log_file])
        except KeyboardInterrupt:
            pass
    else:
        print("日志文件不存在")
    print()

def print_system_info():
    """打印系统信息"""
    info = get_system_info()
    print(f"{Colors.HEADER}系统信息:{Colors.ENDC}")
    print(f"CPU使用率: {info['cpu_percent']:.1f}%")
    print(f"CPU频率: {info['cpu_freq']:.0f} MHz")
    print(f"CPU温度: {info['cpu_temp']:.1f}°C")
    print(f"内存: {info['mem_used']:.1f} MB / {info['mem_total']:.1f} MB ({info['mem_percent']:.1f}%)")
    print(f"磁盘: {info['disk_used']:.1f} GB / {info['disk_total']:.1f} GB ({info['disk_percent']:.1f}%)")
    print()

def main():
    """主函数"""
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_menu()
        
        choice = input("请选择操作 [0-7]: ").strip()
        
        if choice == '0':
            print("退出")
            break
        elif choice == '1':
            try:
                print_simple_monitor()
            except KeyboardInterrupt:
                pass
        elif choice == '2':
            view_service_status()
        elif choice == '3':
            start_service()
        elif choice == '4':
            stop_service()
        elif choice == '5':
            restart_service()
        elif choice == '6':
            view_logs()
        elif choice == '7':
            print_system_info()
        else:
            print(f"{Colors.FAIL}无效选项{Colors.ENDC}")
        
        if choice != '1':
            input("按 Enter 继续...")

if __name__ == "__main__":
    main()
