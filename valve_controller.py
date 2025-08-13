import nidaqmx
import threading
import time
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum
import numpy as np

class OdorType(Enum):
    """气味类型"""
    A = "A"  # 气味A
    B = "B"  # 气味B  
    C = "C"  # 气味C

@dataclass
class ValveAddress:
    """阀门物理地址"""
    device: str      # 例如 "Dev1"
    port: int        # 0, 1, 2
    line: int        # 0-7
    channel_index: int  # 0-23 (在该设备中的索引)
    
    def __post_init__(self):
        self.channel_index = self.port * 8 + self.line
        
    @property
    def physical_address(self) -> str:
        """返回物理地址字符串"""
        return f"{self.device}/port{self.port}/line{self.line}"

@dataclass
class ValveInfo:
    """阀门完整信息"""
    name: str           # 例如 "A_1_1"
    odor_type: str      # A, B, 或 C
    main_valve: int     # 主阀编号 1-3
    sub_valve: int      # 子阀编号 1-6
    maze_id: Optional[int]  # 所属迷宫 (1-6, 可自定义)
    address: ValveAddress  # 物理地址

class MultiDeviceValveController:
    """跨设备阀门控制器 - 支持54个阀门的独立和同步控制"""
    
    def __init__(self, devices: List[str] = None):
        """
        初始化控制器
        
        Args:
            devices: 设备列表，如 ['Dev1', 'Dev2', 'Dev3']
        """
        self.devices = devices or ['Dev1', 'Dev2', 'Dev3']
        self.tasks: Dict[str, nidaqmx.Task] = {}
        self.device_locks: Dict[str, threading.RLock] = {}
        self.valve_mapping: Dict[str, ValveInfo] = {}
        self.maze_valve_mapping: Dict[int, List[str]] = {i: [] for i in range(1, 7)}  # 6个迷宫
        self.device_states: Dict[str, List[bool]] = {}
        self.global_lock = threading.RLock()
        
        # 初始化设备
        self._initialize_devices()
        
        # 加载或创建映射
        self._initialize_mapping()
        
    def _initialize_devices(self):
        """初始化所有NI设备"""
        for device in self.devices:
            try:
                # 创建任务
                task = nidaqmx.Task(f"{device}_task")
                
                # 添加24个通道
                for port in range(3):
                    for line in range(8):
                        channel = f"{device}/port{port}/line{line}"
                        task.do_channels.add_do_chan(channel)
                
                # 初始化状态
                self.tasks[device] = task
                self.device_locks[device] = threading.RLock()
                self.device_states[device] = [False] * 24
                
                # 初始化所有通道为关闭
                task.write([False] * 24)
                
                print(f"✓ 初始化设备 {device} 成功")
                
            except Exception as e:
                print(f"✗ 初始化设备 {device} 失败: {e}")
                
    def _initialize_mapping(self, mapping_file: Optional[str] = None):
        """初始化阀门映射表"""
        if mapping_file:
            self.load_mapping_from_file(mapping_file)
        else:
            self.create_default_mapping()
            
    def create_default_mapping(self):
        """
        创建默认映射表
        将54个阀门分配到3个设备上
        默认每个迷宫分配9个阀门（可后续自定义）
        """
        valve_index = 0
        maze_index = 1
        valves_per_maze = 9  # 默认每个迷宫9个阀门
        valves_assigned = 0
        
        for odor in ['A', 'B', 'C']:
            for main_valve in range(1, 4):  # 1-3
                for sub_valve in range(1, 7):  # 1-6
                    # 计算设备分配
                    device_idx = valve_index // 24
                    channel_in_device = valve_index % 24
                    port = channel_in_device // 8
                    line = channel_in_device % 8
                    
                    # 计算默认迷宫分配
                    current_maze = maze_index
                    valves_assigned += 1
                    
                    if valves_assigned >= valves_per_maze and maze_index < 6:
                        maze_index += 1
                        valves_assigned = 0
                    
                    # 创建阀门信息
                    valve_name = f"{odor}_{main_valve}_{sub_valve}"
                    
                    self.valve_mapping[valve_name] = ValveInfo(
                        name=valve_name,
                        odor_type=odor,
                        main_valve=main_valve,
                        sub_valve=sub_valve,
                        maze_id=current_maze,
                        address=ValveAddress(
                            device=self.devices[min(device_idx, len(self.devices)-1)],
                            port=port,
                            line=line,
                            channel_index=channel_in_device
                        )
                    )
                    
                    # 更新迷宫-阀门映射
                    if current_maze:
                        self.maze_valve_mapping[current_maze].append(valve_name)
                    
                    valve_index += 1
                    
        print(f"✓ 创建默认映射，共{len(self.valve_mapping)}个阀门，6个迷宫")
        
    def set_valve_maze_mapping(self, valve_name: str, maze_id: int):
        """
        设置阀门的迷宫归属
        
        Args:
            valve_name: 阀门名称
            maze_id: 迷宫ID (1-6)
        """
        if valve_name not in self.valve_mapping:
            raise ValueError(f"阀门 {valve_name} 不存在")
        
        if maze_id < 1 or maze_id > 6:
            raise ValueError(f"迷宫ID必须在1-6之间")
            
        # 从旧迷宫中移除
        old_maze = self.valve_mapping[valve_name].maze_id
        if old_maze and old_maze in self.maze_valve_mapping:
            if valve_name in self.maze_valve_mapping[old_maze]:
                self.maze_valve_mapping[old_maze].remove(valve_name)
        
        # 添加到新迷宫
        self.valve_mapping[valve_name].maze_id = maze_id
        if maze_id not in self.maze_valve_mapping:
            self.maze_valve_mapping[maze_id] = []
        self.maze_valve_mapping[maze_id].append(valve_name)
        
    def get_maze_valves(self, maze_id: int) -> List[str]:
        """获取指定迷宫的所有阀门"""
        return self.maze_valve_mapping.get(maze_id, [])
        
    def set_maze_state(self, maze_id: int, state: bool):
        """设置整个迷宫的所有阀门状态"""
        valves = self.get_maze_valves(maze_id)
        if valves:
            valve_states = {valve: state for valve in valves}
            self.set_valves(valve_states)
            print(f"迷宫 {maze_id} 的 {len(valves)} 个阀门已{'开启' if state else '关闭'}")
        else:
            print(f"迷宫 {maze_id} 没有分配阀门")
            
    def load_mapping_from_file(self, filename: str):
        """从文件加载映射表"""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # 清空现有映射
        self.valve_mapping.clear()
        self.maze_valve_mapping = {i: [] for i in range(1, 7)}
        
        # 加载阀门映射
        for valve_name, info in data['valve_mapping'].items():
            odor, main, sub = valve_name.split('_')
            maze_id = info.get('maze_id', None)
            
            self.valve_mapping[valve_name] = ValveInfo(
                name=valve_name,
                odor_type=odor,
                main_valve=int(main),
                sub_valve=int(sub),
                maze_id=maze_id,
                address=ValveAddress(
                    device=info['device'],
                    port=info['port'],
                    line=info['line'],
                    channel_index=info['port'] * 8 + info['line']
                )
            )
            
            # 更新迷宫映射
            if maze_id:
                self.maze_valve_mapping[maze_id].append(valve_name)
                
    def save_mapping_to_file(self, filename: str):
        """保存映射表到文件"""
        data = {
            'valve_mapping': {},
            'maze_mapping': {}
        }
        
        # 保存阀门映射
        for valve_name, info in self.valve_mapping.items():
            data['valve_mapping'][valve_name] = {
                'device': info.address.device,
                'port': info.address.port,
                'line': info.address.line,
                'maze_id': info.maze_id
            }
        
        # 保存迷宫映射
        data['maze_mapping'] = {str(k): v for k, v in self.maze_valve_mapping.items()}
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✓ 映射表已保存到 {filename}")
        
    # ==================== 核心控制方法 ====================
    
    def set_valve(self, valve_name: str, state: bool, immediate: bool = True):
        """设置单个阀门状态"""
        if valve_name not in self.valve_mapping:
            raise ValueError(f"阀门 {valve_name} 不存在")
            
        valve_info = self.valve_mapping[valve_name]
        device = valve_info.address.device
        channel = valve_info.address.channel_index
        
        with self.device_locks[device]:
            self.device_states[device][channel] = state
            
            if immediate:
                self.tasks[device].write(self.device_states[device])
                print(f"阀门 {valve_name} -> {'开' if state else '关'}")
                
    def set_valves(self, valve_states: Dict[str, bool]):
        """批量设置多个阀门（可能跨设备）"""
        device_updates: Dict[str, Dict[int, bool]] = {}
        
        for valve_name, state in valve_states.items():
            if valve_name not in self.valve_mapping:
                print(f"警告：阀门 {valve_name} 不存在")
                continue
                
            valve_info = self.valve_mapping[valve_name]
            device = valve_info.address.device
            channel = valve_info.address.channel_index
            
            if device not in device_updates:
                device_updates[device] = {}
            device_updates[device][channel] = state
            
        self._parallel_device_update(device_updates)
        
    def _parallel_device_update(self, device_updates: Dict[str, Dict[int, bool]]):
        """并行更新多个设备"""
        threads = []
        
        def update_device(device: str, updates: Dict[int, bool]):
            with self.device_locks[device]:
                for channel, state in updates.items():
                    self.device_states[device][channel] = state
                self.tasks[device].write(self.device_states[device])
                
        for device, updates in device_updates.items():
            thread = threading.Thread(
                target=update_device,
                args=(device, updates)
            )
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join(timeout=0.1)
            
    def get_valve_state(self, valve_name: str) -> bool:
        """获取阀门当前状态"""
        if valve_name not in self.valve_mapping:
            raise ValueError(f"阀门 {valve_name} 不存在")
            
        valve_info = self.valve_mapping[valve_name]
        device = valve_info.address.device
        channel = valve_info.address.channel_index
        
        return self.device_states[device][channel]
        
    def print_status(self):
        """打印当前状态"""
        print("\n=== 阀门状态 ===")
        
        # 按迷宫显示
        for maze_id in range(1, 7):
            valves = self.get_maze_valves(maze_id)
            if valves:
                print(f"\n迷宫 {maze_id} ({len(valves)}个阀门):")
                for valve in sorted(valves):
                    state = self.get_valve_state(valve)
                    status = "●" if state else "○"
                    print(f"  {valve}: {status}")
                    
    def cleanup(self):
        """清理资源"""
        for task in self.tasks.values():
            task.write([False] * 24)
            task.close()
        print("✓ 资源已清理")