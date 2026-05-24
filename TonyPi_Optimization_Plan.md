# TonyPi树莓派避障机器人优化方案

## 一、现有算法分析

### 1.1 当前系统架构

**硬件配置：**
- 摄像头（640×480分辨率）
- 超声波传感器
- 双足人形机器人动作组

**软件实现：**
- 超声波测距：6帧滑动平均滤波
- YOLO目标检测：支持v3-tiny（OpenCV DNN）和v12n（PyTorch）
- 决策逻辑：简单的状态机，固定避障流程

### 1.2 现有算法流程

```
1. 超声波测距（每10ms读取，6帧平均）
   ↓
2. 视觉线程：巡线 + YOLO检测（跳帧处理）
   ↓
3. 主决策循环：
   - 距离 < 250mm → 触发避障
   - 根据YOLO方位选择左绕/右绕
   - 执行固定避障动作序列
```

### 1.3 现有问题分析

| 问题类别 | 具体问题 | 影响 |
|---------|---------|------|
| 传感器融合 | 简单的条件判断，无真正融合 | 信息利用率低，决策可靠性差 |
| 路径规划 | 固定动作序列，无动态调整 | 复杂场景适应性差 |
| 决策逻辑 | 只有左/右两种选择 | 缺乏灵活性 |
| 边缘情况 | 无动态障碍物检测、无环境记忆 | 易陷入局部死循环 |
| 性能优化 | 跳帧处理简单，无自适应策略 | 性能和精度难以平衡 |

---

## 二、传感器融合策略改进

### 2.1 多传感器数据融合架构

```
┌─────────────────┐    ┌─────────────────┐
│  超声波传感器    │    │   摄像头+YOLO   │
│  (距离数据)      │    │   (目标检测)     │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  卡尔曼滤波     │    │  目标跟踪+     │
│  (距离平滑)     │    │  距离估算      │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────┐
         │  数据融合模块   │
         │  (贝叶斯网络)   │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  环境地图构建   │
         └─────────────────┘
```

### 2.2 具体改进方案

#### 2.2.1 超声波数据增强

**现有问题：** 仅使用简单平均滤波，抗干扰能力弱

**改进方案：**
```python
# 改进的超声波滤波
class UltrasonicFilter:
    def __init__(self, window_size=10):
        self.window = deque(maxlen=window_size)
        self.last_valid = 0
    
    def update(self, value):
        if 0 < value < 5000:
            # 异常值检测（3σ原则）
            if len(self.window) >= 3:
                mean = np.mean(self.window)
                std = np.std(self.window)
                if abs(value - mean) > 3 * std:
                    return self.last_valid
            
            self.window.append(value)
            self.last_valid = np.median(self.window)  # 中值滤波更鲁棒
        
        return self.last_valid
```

#### 2.2.2 YOLO信息增强

**现有问题：** 仅使用目标方位，无距离、大小、类别信息

**改进方案：**
1. 利用边界框大小估算目标距离
2. 目标类别识别（区分静态/动态障碍物）
3. 多目标跟踪（匈牙利算法）

```python
def estimate_distance(bbox_width, real_width=0.5, focal_length=600):
    """
    通过边界框宽度估算距离
    real_width: 目标实际宽度（米）
    focal_length: 相机焦距（像素）
    """
    if bbox_width <= 0:
        return float('inf')
    return (real_width * focal_length) / bbox_width

class MultiObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections):
        # 匈牙利算法匹配
        # ...
        pass
```

#### 2.2.3 贝叶斯融合

将超声波和视觉信息用贝叶斯网络融合：

```python
class BayesianFusion:
    def __init__(self):
        # 先验概率
        self.p_obstacle = 0.5
    
    def fuse(self, ultrasonic_dist, yolo_detections):
        """
        融合超声波和YOLO信息
        返回：障碍物存在概率、距离估计
        """
        # 超声波似然
        p_us_given_obstacle = self._us_likelihood(ultrasonic_dist)
        p_us_given_no_obstacle = 1 - p_us_given_obstacle
        
        # YOLO似然
        p_yolo_given_obstacle = self._yolo_likelihood(yolo_detections)
        p_yolo_given_no_obstacle = 1 - p_yolo_given_obstacle
        
        # 贝叶斯更新
        numerator = self.p_obstacle * p_us_given_obstacle * p_yolo_given_obstacle
        denominator = numerator + (1 - self.p_obstacle) * p_us_given_no_obstacle * p_yolo_given_no_obstacle
        
        p_obstacle_posterior = numerator / denominator if denominator > 0 else 0
        
        # 距离估计（加权平均）
        dist_estimate = self._weighted_distance(ultrasonic_dist, yolo_detections, p_obstacle_posterior)
        
        return p_obstacle_posterior, dist_estimate
```

---

## 三、避障路径规划优化

### 3.1 动态窗口法（DWA）适配

针对双足机器人特点，简化DWA：

```python
class DynamicWindowApproach:
    def __init__(self):
        # 动作空间：TonyPi支持的动作
        self.actions = {
            'forward': {'cost': 1, 'dx': 1, 'dtheta': 0},
            'turn_left': {'cost': 2, 'dx': 0, 'dtheta': -30},
            'turn_right': {'cost': 2, 'dx': 0, 'dtheta': 30},
            'left_step': {'cost': 3, 'dx': 0.5, 'dtheta': -15},
            'right_step': {'cost': 3, 'dx': 0.5, 'dtheta': 15}
        }
    
    def plan(self, obstacle_map, target_dir):
        """
        规划下一步动作
        obstacle_map: 障碍物网格地图
        target_dir: 目标方向
        """
        best_action = None
        best_score = -float('inf')
        
        for action, params in self.actions.items():
            # 预测下一步状态
            next_state = self._predict_state(action)
            
            # 计算代价
            obstacle_cost = self._obstacle_cost(next_state, obstacle_map)
            target_cost = self._target_cost(next_state, target_dir)
            total_cost = obstacle_cost + target_cost + params['cost']
            
            score = -total_cost  # 代价越小越好
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
```

### 3.2 局部势场法

```python
class PotentialField:
    def __init__(self):
        self.k_att = 0.5  # 吸引力系数
        self.k_rep = 2.0  # 排斥力系数
        self.rep_threshold = 300  # 排斥力作用距离（mm）
    
    def compute_force(self, robot_pos, target_pos, obstacles):
        """
        计算合力
        """
        # 吸引力
        att_force = self._attractive_force(robot_pos, target_pos)
        
        # 排斥力
        rep_force = np.array([0.0, 0.0])
        for obs in obstacles:
            rep_force += self._repulsive_force(robot_pos, obs)
        
        total_force = att_force + rep_force
        
        # 转换为动作
        return self._force_to_action(total_force)
```

### 3.3 改进的避障流程

不再是固定动作序列，而是动态调整：

```python
def enhanced_obstacle_avoidance(robot_state, fusion_result):
    """
    增强的避障决策
    """
    p_obstacle, dist_estimate = fusion_result
    
    # 多级避障策略
    if dist_estimate < 100:
        # 紧急避障
        return 'emergency_stop'
    elif dist_estimate < 200:
        # 快速转向
        return choose_turn_direction(robot_state)
    elif dist_estimate < 350:
        # 规划绕行
        return dwa.plan(robot_state.map, robot_state.target)
    else:
        # 继续前进
        return 'forward'
```

---

## 四、决策逻辑改进

### 4.1 有限状态机（FSM）增强

```
┌──────────┐
│  待机    │
└────┬─────┘
     │ 启动
     ▼
┌──────────────────┐
│  巡线模式        │←──────────┐
└────┬─────────────┘          │
     │ 检测到障碍              │
     ▼                         │
┌──────────────────┐          │
│  评估场景        │          │
└────┬─────────────┘          │
     │                         │
     ├─→ 静态障碍 → 绕行模式 ──┤
     │                         │
     ├─→ 动态障碍 → 等待模式 ──┤
     │                         │
     └─→ 复杂场景 → 探索模式 ──┘
```

### 4.2 实现代码

```python
class StateMachine:
    STATE_IDLE = 'idle'
    STATE_PATROL = 'patrol'
    STATE_EVALUATE = 'evaluate'
    STATE_AVOID_STATIC = 'avoid_static'
    STATE_WAIT_DYNAMIC = 'wait_dynamic'
    STATE_EXPLORE = 'explore'
    
    def __init__(self):
        self.state = self.STATE_IDLE
        self.state_history = deque(maxlen=10)
        self.decision_count = 0
    
    def transition(self, sensor_data):
        """
        状态转换逻辑
        """
        self.state_history.append(self.state)
        self.decision_count += 1
        
        obstacle_info = self._analyze_obstacles(sensor_data)
        
        if self.state == self.STATE_PATROL:
            if obstacle_info['detected']:
                if obstacle_info['is_dynamic']:
                    return self.STATE_WAIT_DYNAMIC
                elif obstacle_info['complexity'] > 0.7:
                    return self.STATE_EXPLORE
                else:
                    return self.STATE_AVOID_STATIC
        
        # 其他状态转换...
        
        return self.state
```

---

## 五、边缘情况处理

### 5.1 检测和处理策略

| 边缘情况 | 检测方法 | 处理策略 |
|---------|---------|---------|
| 动态障碍物 | YOLO跟踪 + 光流法 | 预测轨迹，等待或绕行 |
| 狭窄通道 | 超声波两侧测距（需添加硬件）或视觉分析 | 慢速通过，精细调整 |
| 死胡同 | 历史轨迹记忆 + 无出口检测 | 掉头返回 |
| 低光照 | 摄像头曝光检测 | 依赖超声波为主 |
| 跌倒 | 原有检测逻辑 | 自动起立 + 环境重评估 |

### 5.2 动态障碍物处理

```python
class DynamicObstacleHandler:
    def __init__(self):
        self.track_history = {}  # id → [(pos, time), ...]
    
    def predict_trajectory(self, track_id, steps=5):
        """
        预测障碍物轨迹
        """
        if track_id not in self.track_history:
            return None
        
        history = self.track_history[track_id]
        if len(history) < 3:
            return None
        
        # 卡尔曼预测或简单线性外推
        velocities = []
        for i in range(1, len(history)):
            dt = history[i][1] - history[i-1][1]
            dx = history[i][0][0] - history[i-1][0][0]
            dy = history[i][0][1] - history[i-1][0][1]
            velocities.append((dx/dt, dy/dt))
        
        avg_vel = np.mean(velocities, axis=0)
        last_pos, last_time = history[-1]
        
        trajectory = []
        for i in range(1, steps+1):
            pred_time = last_time + i * 0.1
            pred_pos = (last_pos[0] + avg_vel[0] * i * 0.1,
                       last_pos[1] + avg_vel[1] * i * 0.1)
            trajectory.append((pred_pos, pred_time))
        
        return trajectory
    
    def check_collision(self, robot_traj, obstacle_traj):
        """
        碰撞检测
        """
        safety_distance = 200  # mm
        for (r_pos, r_time), (o_pos, o_time) in zip(robot_traj, obstacle_traj):
            dist = np.linalg.norm(np.array(r_pos) - np.array(o_pos))
            if dist < safety_distance:
                return True, r_time
        return False, None
```

### 5.3 死胡同检测

```python
class HistoryAnalyzer:
    def __init__(self):
        self.position_history = deque(maxlen=100)
        self.direction_history = deque(maxlen=50)
    
    def detect_dead_end(self, current_sensor_data):
        """
        检测是否陷入死胡同
        """
        # 条件1：前方、左、右都有障碍
        front_blocked = current_sensor_data['distance'] < 200
        left_blocked = current_sensor_data.get('left_distance', 999) < 200
        right_blocked = current_sensor_data.get('right_distance', 999) < 200
        
        # 条件2：在小范围内打转
        if len(self.position_history) > 20:
            recent_positions = list(self.position_history)[-20:]
            variance = np.var(recent_positions, axis=0)
            area_covered = np.sqrt(variance[0] + variance[1])
            looping = area_covered < 100  # mm
        else:
            looping = False
        
        return front_blocked and (left_blocked or right_blocked) and looping
    
    def escape_dead_end(self):
        """
        死胡同逃脱策略
        """
        return ['turn_left', 'turn_left', 'go_forward_fast', 'go_forward_fast']
```

---

## 六、完整实现建议

### 6.1 文件结构

```
tonypi_improved/
├── main.py                    # 主程序入口
├── sensor_fusion.py           # 传感器融合模块
├── path_planning.py           # 路径规划模块
├── decision_logic.py          # 决策逻辑模块
├── edge_cases.py              # 边缘情况处理
├── utils.py                   # 工具函数
└── config.py                  # 配置参数
```

### 6.2 硬件建议

1. **如预算允许，增加：**
   - 左侧超声波传感器
   - 右侧超声波传感器
   - IMU（惯性测量单元）

2. **当前硬件优化：**
   - 摄像头帧率优化
   - 超声波采样频率调整

### 6.3 性能优化

- 多线程/多进程架构
- 自适应跳帧（根据CPU负载）
- 模型量化（YOLOv12 INT8）

---

## 七、总结

本优化方案从五个方面对TonyPi避障系统进行了增强：

1. **传感器融合：** 从简单条件判断升级为贝叶斯网络融合
2. **路径规划：** 从固定动作序列升级为DWA+势场法动态规划
3. **决策逻辑：** 增强的有限状态机，支持多种场景
4. **边缘情况：** 动态障碍物、死胡同等专门处理
5. **机器人特性：** 充分考虑双足机器人的动作限制

这些改进可以在现有硬件基础上逐步实现，建议分阶段部署：
- Phase 1: 传感器融合增强
- Phase 2: 决策逻辑升级
- Phase 3: 路径规划优化
- Phase 4: 边缘情况完善
