#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪器集成模块 - 修复版，避免重复初始化相机
"""

import threading
import time
import cv2
import numpy as np
from collections import deque
import datetime
import os
from queue import Queue

# 导入现有模块
from test import livetracker, MindVisionCamera
from multi_lightsvalvesobject_2odors_20240611 import LightsValves
import nidaqmx


class TrackerWrapper(threading.Thread):
    """
    包装现有的livetracker类，使其能够与UI通信
    """
    def __init__(self, camera_id, maze_id, camera_obj, task, data_queue, name=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.maze_id = maze_id
        self.camera_obj = camera_obj
        self.task = task
        self.data_queue = data_queue
        self.name = name or f"Cam{camera_id}_{chr(65+maze_id)}"
        
        # 控制标志
        self.running = False
        self.paused = False
        self.stop_flag = False
        
        # 数据存储
        self.current_frame = None
        self.larva_position = (0, 0)
        self.decision_count = {'odor_a': 0, 'odor_b': 0, 'air': 0}
        self.valve_states = {}
        self.trajectory = deque(maxlen=500)
        
        # 创建livetracker实例
        self.tracker = None
        
    def run(self):
        """运行追踪器"""
        try:
            # 创建自定义的livetracker
            self.tracker = CustomLiveTracker(
                name=self.name,
                camera_id=self.camera_id,
                maze_id=self.maze_id,
                data_callback=self.handle_tracker_data
            )
            
            # 设置相机和任务
            self.tracker.camera_obj = self.camera_obj
            self.tracker.task = self.task
            
            # 运行追踪器
            self.running = True
            self.tracker.run()
            
        except Exception as e:
            print(f"追踪器 {self.name} 出错: {e}")
            self.send_error(str(e))
            
    def handle_tracker_data(self, data):
        """处理来自追踪器的数据（修复版 - 确保数据正确发送）"""
        if self.paused or self.stop_flag:
            return
            
        # 更新本地数据
        if 'frame' in data:
            self.current_frame = data['frame']
            
        if 'position' in data:
            self.larva_position = data['position']
            self.trajectory.append(data['position'])
            
        if 'decision' in data:
            decision_type = data['decision']
            if decision_type in self.decision_count:
                self.decision_count[decision_type] += 1
                
        if 'valve_states' in data:
            self.valve_states = data['valve_states']
            
        # 发送数据到UI（统一数据流）
        try:
            ui_data = {
                'camera_id': self.camera_id,
                'maze_id': self.maze_id,
                'frame': self.current_frame.copy() if self.current_frame is not None else None,
                'position': self.larva_position,
                'stats': self.decision_count.copy(),
                'valve_states': self.valve_states.copy(),
                'trajectory': list(self.trajectory),
                'timestamp': time.time(),
                'source': 'tracker'  # 标记数据源
            }
            
            # 非阻塞发送到队列
            if hasattr(self, 'data_queue') and self.data_queue is not None:
                try:
                    self.data_queue.put_nowait(ui_data)
                except:
                    # 队列满时跳过当前数据，不影响追踪
                    pass
            else:
                print(f"Warning: No data_queue available for {self.name}")
                
        except Exception as e:
            print(f"Error sending tracker data: {e}")
            # 发送错误不影响追踪继续
        
    def pause(self):
        """暂停追踪"""
        self.paused = True
        if self.tracker:
            self.tracker.pause()
            
    def resume(self):
        """恢复追踪"""
        self.paused = False
        if self.tracker:
            self.tracker.resume()
            
    def stop(self):
        """停止追踪"""
        self.stop_flag = True
        self.running = False
        if self.tracker:
            self.tracker.stop()
            
    def send_error(self, error_msg):
        """发送错误信息到UI"""
        error_data = {
            'type': 'error',
            'camera_id': self.camera_id,
            'maze_id': self.maze_id,
            'message': error_msg,
            'timestamp': time.time()
        }
        self.data_queue.put(error_data)


class CustomLiveTracker(livetracker):
    """
    自定义的livetracker类，添加数据回调功能
    """
    def __init__(self, name, camera_id, maze_id, data_callback):
        # 不调用父类的__init__，因为它会启动线程
        threading.Thread.__init__(self, name=name, daemon=True)
        self.camera_id = camera_id
        self.maze_id = maze_id
        self.data_callback = data_callback
        self.paused = False
        self.stop_flag = False
        
    def run(self):
        """重写run方法，添加数据回调"""
        # 这里应该是原始livetracker的run方法的修改版本
        # 在关键位置添加数据回调
        
        # 示例实现（应该根据实际的livetracker代码修改）
        try:
            # 初始化
            self.setup_tracking()
            
            # 主循环
            while not self.stop_flag:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                # 获取帧
                frame = self.get_frame()
                if frame is None:
                    continue
                    
                # 处理帧
                result = self.process_frame(frame)
                
                # 发送数据到UI
                if self.data_callback:
                    self.data_callback(result)
                    
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"追踪器错误: {e}")
            
    def setup_tracking(self):
        """设置追踪（简化版本）"""
        # 这里应该包含原始的设置代码
        pass
        
    def get_frame(self):
        """获取帧（简化版本）"""
        # 使用模拟数据进行测试
        frame = np.random.randint(0, 255, (450, 450), dtype=np.uint8)
        return frame
        
    def process_frame(self, frame):
        """处理帧（简化版本）"""
        # 模拟处理结果
        result = {
            'frame': frame,
            'position': (np.random.randint(0, 450), np.random.randint(0, 450)),
            'valve_states': {
                'odor1': np.random.random() < 0.3,
                'odor2': np.random.random() < 0.3,
                'air': True
            }
        }
        
        # 随机产生决策
        if np.random.random() < 0.05:
            decision = np.random.choice(['odor_a', 'odor_b', 'air'])
            result['decision'] = decision
            
        return result
        
    def pause(self):
        """暂停追踪"""
        self.paused = True
        
    def resume(self):
        """恢复追踪"""
        self.paused = False
        
    def stop(self):
        """停止追踪"""
        self.stop_flag = True


class CameraManager:
    """
    相机管理器 - 修复版，不重复初始化相机
    """
    def __init__(self):
        self.cameras = {}  # 存储相机对象
        self.camera_threads = {}
        self.frame_buffers = {}
        
    def set_cameras(self, cameras):
        """设置已经初始化的相机对象"""
        self.cameras = cameras
        self.logger.info(f"使用 {len(cameras)} 个已初始化的相机")
        
        # 为每个相机创建帧缓冲区和采集线程
        for camera_id, camera in cameras.items():
            # 创建帧缓冲区
            self.frame_buffers[camera_id] = Queue(maxsize=10)
            
            # 启动采集线程
            thread = threading.Thread(
                target=self.capture_thread,
                args=(camera_id, camera),
                daemon=True
            )
            thread.start()
            self.camera_threads[camera_id] = thread
            
        return len(self.cameras)
        
    def initialize_cameras(self):
        """初始化所有相机（修复版 - 检查是否已经初始化）"""
        if self.cameras:
            print(f"相机已经初始化，共 {len(self.cameras)} 个")
            return len(self.cameras)
            
        try:
            import mvsdk
            
            # 枚举设备
            DevList = mvsdk.CameraEnumerateDevice()
            print(f"发现 {len(DevList)} 个相机")
            
            # 初始化每个相机（最多4个）
            for i in range(min(len(DevList), 4)):
                try:
                    camera = MindVisionCamera()
                    camera.open()
                    self.cameras[i] = camera
                    
                    # 创建帧缓冲区
                    self.frame_buffers[i] = Queue(maxsize=10)
                    
                    # 启动采集线程
                    thread = threading.Thread(
                        target=self.capture_thread,
                        args=(i, camera),
                        daemon=True
                    )
                    thread.start()
                    self.camera_threads[i] = thread
                    
                    print(f"相机 {i} 初始化成功")
                    
                except Exception as e:
                    print(f"相机 {i} 初始化失败: {e}")
                    
            return len(self.cameras)
            
        except ImportError:
            print("无法导入相机SDK，使用模拟模式")
            return 0
            
    def capture_thread(self, camera_id, camera):
        """相机采集线程"""
        while True:
            try:
                frame = camera.grab()
                if frame is not None:
                    # 将帧放入缓冲区
                    if not self.frame_buffers[camera_id].full():
                        self.frame_buffers[camera_id].put(frame)
                        
            except Exception as e:
                print(f"相机 {camera_id} 采集错误: {e}")
                time.sleep(0.1)
                
    def get_frame(self, camera_id):
        """获取指定相机的最新帧"""
        if camera_id in self.frame_buffers:
            try:
                return self.frame_buffers[camera_id].get_nowait()
            except:
                return None
        return None
        
    def close_all(self):
        """关闭所有相机（注意：不在这里关闭，由系统统一管理）"""
        # 不关闭相机，只清理线程
        print("清理相机采集线程...")


class DataRecorder:
    """
    数据记录器 - 负责记录实验数据
    """
    def __init__(self, base_path="D:/Behaviour/data"):
        self.base_path = base_path
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_handles = {}
        
        # 创建数据目录
        self.data_dir = os.path.join(base_path, self.current_date)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def start_recording(self, camera_id, maze_id):
        """开始记录指定迷宫的数据"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        
        # 创建文件
        base_filename = f"{maze_name}_{self.session_id}"
        
        files = {
            'decision': os.path.join(self.data_dir, f"{base_filename}_decisionlog.txt"),
            'valve': os.path.join(self.data_dir, f"{base_filename}_valvelog.txt"),
            'position': os.path.join(self.data_dir, f"{base_filename}_positionframe.txt")
        }
        
        # 打开文件句柄
        handles = {}
        for key, path in files.items():
            handles[key] = open(path, 'w', encoding='utf-8')
            
        self.file_handles[maze_name] = handles
        
        # 写入头部信息
        handles['decision'].write("Event,Frame,ElapsedTime,VideoTime\n")
        handles['valve'].write("Event,Frame,ElapsedTime,VideoTime\n")
        handles['position'].write("Position,Frame,ElapsedTime,Velocity,InRegion\n")
        
    def record_decision(self, camera_id, maze_id, decision, frame_count, elapsed_time):
        """记录决策"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        if maze_name in self.file_handles:
            video_time = self.format_video_time(frame_count)
            line = f"{decision},{frame_count},{elapsed_time:.2f},{video_time}\n"
            self.file_handles[maze_name]['decision'].write(line)
            self.file_handles[maze_name]['decision'].flush()
            
    def record_valve_event(self, camera_id, maze_id, event, frame_count, elapsed_time):
        """记录阀门事件"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        if maze_name in self.file_handles:
            video_time = self.format_video_time(frame_count)
            line = f"{event},{frame_count},{elapsed_time:.2f},{video_time}\n"
            self.file_handles[maze_name]['valve'].write(line)
            self.file_handles[maze_name]['valve'].flush()
            
    def record_position(self, camera_id, maze_id, position, frame_count, elapsed_time, velocity, in_region):
        """记录位置"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        if maze_name in self.file_handles:
            pos_str = f"[{position[0]},{position[1]}]"
            line = f"{pos_str},{frame_count},{elapsed_time:.2f},{velocity:.2f},{in_region}\n"
            self.file_handles[maze_name]['position'].write(line)
            self.file_handles[maze_name]['position'].flush()
            
    def format_video_time(self, frame_count, fps=30):
        """格式化视频时间"""
        total_seconds = frame_count / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def stop_recording(self, camera_id, maze_id):
        """停止记录"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        if maze_name in self.file_handles:
            # 写入统计信息
            # ...
            
            # 关闭文件
            for handle in self.file_handles[maze_name].values():
                handle.close()
                
            del self.file_handles[maze_name]
            
    def close_all(self):
        """关闭所有文件"""
        for handles in self.file_handles.values():
            for handle in handles.values():
                try:
                    handle.close()
                except:
                    pass
                    

class ExperimentController:
    """
    实验控制器 - 修复版，避免重复初始化相机
    """
    def __init__(self, ui_callback=None):
        self.ui_callback = ui_callback
        self.camera_manager = CameraManager()
        self.data_recorder = DataRecorder()
        self.trackers = {}
        self.data_queue = Queue()
        
        # 阀门任务
        self.valve_tasks = self.init_valve_tasks()
        
    def set_cameras(self, cameras):
        """设置已经初始化的相机对象"""
        self.camera_manager.set_cameras(cameras)
        
    def init_valve_tasks(self):
        """初始化阀门任务"""
        tasks = {}
        try:
            # 创建任务1
            task1 = nidaqmx.Task()
            for i in range(24):
                channel = f"Dev1/port{i//8}/line{i%8}"
                task1.do_channels.add_do_chan(channel)
            tasks['task1'] = task1
            
            # 创建任务2
            task2 = nidaqmx.Task()
            for i in range(24):
                channel = f"Dev2/port{i//8}/line{i%8}"
                task2.do_channels.add_do_chan(channel)
            tasks['task2'] = task2
            
        except Exception as e:
            print(f"阀门初始化失败: {e}")
            
        return tasks
        
    def initialize(self):
        """初始化系统（修复版 - 不重复初始化相机）"""
        # 如果没有预设相机，则初始化
        if not self.camera_manager.cameras:
            num_cameras = self.camera_manager.initialize_cameras()
        else:
            num_cameras = len(self.camera_manager.cameras)
            
        # 启动数据处理线程
        self.data_thread = threading.Thread(
            target=self.process_data,
            daemon=True
        )
        self.data_thread.start()
        
        return num_cameras
        
    def start_maze(self, camera_id, maze_id):
        """启动单个迷宫"""
        key = f"{camera_id}_{maze_id}"
        
        if key not in self.trackers:
            # 确定使用哪个任务
            task = self.valve_tasks.get('task1' if camera_id < 2 else 'task2')
            
            # 获取相机对象
            camera_obj = self.camera_manager.cameras.get(camera_id)
            if not camera_obj:
                print(f"相机 {camera_id} 不存在")
                return
                
            # 创建追踪器
            tracker = TrackerWrapper(
                camera_id=camera_id,
                maze_id=maze_id,
                camera_obj=camera_obj,
                task=task,
                data_queue=self.data_queue
            )
            
            # 开始记录
            self.data_recorder.start_recording(camera_id, maze_id)
            
            # 启动追踪器
            tracker.start()
            self.trackers[key] = tracker
            
    def stop_maze(self, camera_id, maze_id):
        """停止单个迷宫"""
        key = f"{camera_id}_{maze_id}"
        
        if key in self.trackers:
            # 停止追踪器
            self.trackers[key].stop()
            
            # 停止记录
            self.data_recorder.stop_recording(camera_id, maze_id)
            
            # 移除追踪器
            del self.trackers[key]
            
    def process_data(self):
        """处理数据队列"""
        while True:
            try:
                data = self.data_queue.get(timeout=0.1)
                
                # 记录数据
                if 'type' not in data or data['type'] != 'error':
                    self.record_data(data)
                    
                # 发送到UI
                if self.ui_callback:
                    self.ui_callback(data)
                    
            except:
                pass
                
    def record_data(self, data):
        """记录数据到文件"""
        camera_id = data.get('camera_id')
        maze_id = data.get('maze_id')
        
        # 这里应该调用data_recorder的相应方法
        # 根据数据类型记录不同的信息
        
    def shutdown(self):
        """关闭系统（修复版）"""
        # 停止所有追踪器
        for tracker in list(self.trackers.values()):
            try:
                tracker.stop()
            except:
                pass
                
        # 清理相机管理器（但不关闭相机）
        self.camera_manager.close_all()
        
        # 关闭数据记录
        self.data_recorder.close_all()
        
        # 关闭阀门任务
        for task in self.valve_tasks.values():
            try:
                task.close()
            except:
                pass