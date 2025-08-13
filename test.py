#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幼虫行为追踪集成系统 - 修正相机调用版本
支持单相机6迷宫监控
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
import sys
import random
import math
import platform
from datetime import datetime
from collections import deque, defaultdict
from PIL import Image, ImageTk
from typing import Dict, List, Tuple, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要模块
try:
    import mvsdk
    HARDWARE_MODE = True
except ImportError as e:
    print(f"警告：相机SDK未安装 ({e})，使用模拟模式")
    HARDWARE_MODE = False

# 导入其他模块
from BakCreator import BakCreator
from region_shihan import Region

# 状态机相关
class StateMachine:
    """简化的状态机实现"""
    def __init__(self, initial, locs):
        self.locs = locs
        self.state = 0
        
    def on_input(self, input_pos):
        if input_pos == (-1, -1):
            return self.state, self.state, self.state
            
        # 计算当前状态
        new_state = self.get_state(input_pos)
        prev_state = self.state
        self.state = new_state
        return prev_state, new_state, new_state
    
    def get_state(self, pos):
        """根据位置计算状态（0-6）"""
        if pos == (-1, -1):
            return 0
            
        # 简化的状态判断
        distances = []
        for loc in self.locs:
            dist = np.linalg.norm(np.array(pos) - np.array(loc))
            distances.append(dist)
        
        min_idx = np.argmin(distances)
        return min_idx

class CameraManager:
    """相机管理器 - 修正版本"""
    def __init__(self):
        self.hCamera = None
        self.pFrameBuffer = None
        self.cap = None
        self.monoCamera = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.initialized = False
        
    def initialize(self):
        """初始化相机"""
        try:
            # 枚举相机
            DevList = mvsdk.CameraEnumerateDevice()
            if len(DevList) < 1:
                print("未找到相机，使用模拟模式")
                return False
            
            # 显示找到的相机
            for i, DevInfo in enumerate(DevList):
                print(f"找到相机 {i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
            
            # 使用第一个相机
            DevInfo = DevList[0]
            
            # 打开相机
            try:
                self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            except mvsdk.CameraException as e:
                print(f"相机初始化失败({e.error_code}): {e.message}")
                return False
            
            # 获取相机特性描述
            self.cap = mvsdk.CameraGetCapability(self.hCamera)
            
            # 判断是黑白相机还是彩色相机
            self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)
            
            # 设置输出格式
            if self.monoCamera:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            
            # 相机模式切换成连续采集
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            
            # 手动曝光，曝光时间30ms
            mvsdk.CameraSetAeState(self.hCamera, 0)
            mvsdk.CameraSetExposureTime(self.hCamera, 30 * 1000)
            
            # 让SDK内部取图线程开始工作
            mvsdk.CameraPlay(self.hCamera)
            
            # 计算RGB buffer所需的大小
            FrameBufferSize = self.cap.sResolutionRange.iWidthMax * \
                            self.cap.sResolutionRange.iHeightMax * \
                            (1 if self.monoCamera else 3)
            
            # 分配RGB buffer
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            
            self.initialized = True
            print("相机初始化成功")
            return True
            
        except Exception as e:
            print(f"相机初始化异常: {e}")
            return False
    
    def grab_frame(self):
        """抓取一帧图像"""
        if not self.initialized or not self.hCamera:
            return None
        
        try:
            # 从相机取一帧图片
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            
            # 处理图像
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            
            # 释放缓冲区
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            
            # Windows下图像是上下颠倒的，需要翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
            
            # 转换为numpy数组
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            if self.monoCamera:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth))
            else:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))
            
            return frame
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"获取图像失败({e.error_code}): {e.message}")
            return None
    
    def close(self):
        """关闭相机"""
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
        
        if self.pFrameBuffer:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = None
        
        self.initialized = False
        print("相机已关闭")

class MazeTracker:
    """单个迷宫的追踪器 - 简化版"""
    def __init__(self, maze_id, roi_coords):
        self.maze_id = maze_id
        self.roi_coords = roi_coords  # (x1, y1, x2, y2)
        self.name = f"Maze_{maze_id}"
        
        # 状态机
        self.state_machine = None
        self.locs = []
        self.current_state = 0
        
        # 追踪数据
        self.frame = None
        self.background = None
        self.position = (-1, -1)
        self.trajectory = deque(maxlen=500)
        self.frame_count = 0
        
        # 决策统计
        self.decisions = {'A': 0, 'B': 0, 'C': 0}
        
        # 背景建模
        self.bg_creator = None
        self.bg_frames = deque(maxlen=30)
        
    def initialize_regions(self, locs):
        """初始化7个区域点"""
        if len(locs) != 7:
            print(f"警告：需要7个点，当前{len(locs)}个")
            return False
        
        self.locs = locs
        self.state_machine = StateMachine(locs[0], locs)
        return True
    
    def process_frame(self, full_frame):
        """处理一帧图像"""
        if full_frame is None:
            return
        
        # 提取ROI
        x1, y1, x2, y2 = self.roi_coords
        
        # 确保坐标在图像范围内
        h, w = full_frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        roi = full_frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return
        
        # 转换为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        self.frame = gray
        
        # 背景建模
        self.bg_frames.append(gray)
        
        if len(self.bg_frames) == 30 and self.background is None:
            # 初始背景
            self.background = np.median(list(self.bg_frames), axis=0).astype(np.uint8)
            self.bg_creator = BakCreator(stacklen=60, alpha=0.02, bgim=self.background)
        
        if self.background is not None:
            # 前景提取
            fg = cv2.absdiff(gray, self.background)
            _, fgthresh = cv2.threshold(fg, 15, 255, cv2.THRESH_BINARY)
            
            # 形态学处理
            kernel = np.ones((3, 3), np.uint8)
            fgthresh = cv2.morphologyEx(fgthresh, cv2.MORPH_CLOSE, kernel)
            fgthresh = cv2.morphologyEx(fgthresh, cv2.MORPH_OPEN, kernel)
            
            # 查找幼虫
            self.position = self.find_larva(fgthresh)
            
            # 更新背景
            self.background = self.bg_creator.run(gray, fgthresh)
            
            # 更新状态
            if self.state_machine and self.position != (-1, -1):
                prev, curr, next = self.state_machine.on_input(self.position)
                
                if curr != self.current_state:
                    self.on_state_change(self.current_state, curr)
                    self.current_state = curr
            
            # 记录轨迹
            if self.position != (-1, -1):
                self.trajectory.append(self.position)
        
        self.frame_count += 1
    
    def find_larva(self, binary_img):
        """查找幼虫位置"""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找最大轮廓
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        return (-1, -1)
    
    def on_state_change(self, old_state, new_state):
        """状态变化处理"""
        print(f"[{self.name}] 状态: {old_state} -> {new_state}")
        
        # 简化的决策记录
        if new_state in [4, 5, 6]:  # 进入圆形区域
            choice = random.choice(['A', 'B', 'C'])
            self.decisions[choice] += 1
            print(f"[{self.name}] 决策: {choice}")
    
    def get_display_frame(self):
        """获取显示用的帧"""
        if self.frame is None:
            return None
        
        # 转换为彩色
        if len(self.frame.shape) == 2:
            display = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        else:
            display = self.frame.copy()
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            points = list(self.trajectory)
            for i in range(1, len(points)):
                cv2.line(display, points[i-1], points[i], (0, 255, 0), 1)
        
        # 绘制当前位置
        if self.position != (-1, -1):
            cv2.circle(display, self.position, 5, (0, 0, 255), -1)
        
        # 添加文本信息
        cv2.putText(display, f"State: {self.current_state}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"A:{self.decisions['A']} B:{self.decisions['B']} C:{self.decisions['C']}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display

class SimplifiedLarvaSystem:
    """简化的单相机6迷宫系统"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("幼虫追踪系统 - 单相机6迷宫")
        self.root.geometry("1200x800")
        
        # 系统参数
        self.num_mazes = 6
        self.camera = None
        self.trackers = {}
        
        # ROI配置 (将图像分为2x3网格)
        self.roi_configs = {}
        self.region_configs = {}
        
        # 运行状态
        self.running = False
        self.capture_thread = None
        
        # UI组件
        self.canvas_widgets = {}
        self.status_labels = {}
        
        # 初始化
        self.setup_ui()
        self.initialize_camera()
        self.initialize_roi()
        
    def setup_ui(self):
        """创建用户界面"""
        # 创建工具栏
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="开始", command=self.start_tracking).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="停止", command=self.stop_tracking).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="标注区域", command=self.start_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="测试相机", command=self.test_camera).pack(side=tk.LEFT, padx=5)
        
        self.fps_label = ttk.Label(toolbar, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # 创建迷宫显示网格 (2x3)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            # 迷宫框架
            maze_frame = ttk.LabelFrame(main_frame, text=f"迷宫 {i+1}")
            maze_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # 画布
            canvas = tk.Canvas(maze_frame, width=350, height=300, bg='gray')
            canvas.pack(padx=5, pady=5)
            
            # 绑定点击事件
            canvas.bind("<Button-1>", lambda e, m=i: self.on_canvas_click(e, m))
            
            # 状态标签
            status_label = ttk.Label(maze_frame, text="状态: 未初始化")
            status_label.pack()
            
            # 保存引用
            self.canvas_widgets[i] = canvas
            self.status_labels[i] = status_label
        
        # 配置网格权重
        for i in range(3):
            main_frame.columnconfigure(i, weight=1)
        for i in range(2):
            main_frame.rowconfigure(i, weight=1)
        
        # 状态栏
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = tk.StringVar(value="系统就绪")
        ttk.Label(status_bar, textvariable=self.status_text).pack(side=tk.LEFT, padx=10)
    
    def initialize_camera(self):
        """初始化相机"""
        if HARDWARE_MODE:
            self.camera = CameraManager()
            if self.camera.initialize():
                self.status_text.set("相机初始化成功")
            else:
                self.camera = None
                self.status_text.set("相机初始化失败，使用模拟模式")
        else:
            self.status_text.set("模拟模式")
    
    def initialize_roi(self):
        """初始化ROI (2x3网格分割)"""
        # 假设相机图像尺寸
        img_width = 2064
        img_height = 3088
        
        roi_width = img_width // 3
        roi_height = img_height // 2
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            x1 = col * roi_width
            y1 = row * roi_height
            x2 = x1 + roi_width
            y2 = y1 + roi_height
            
            self.roi_configs[i] = (x1, y1, x2, y2)
            
            # 创建追踪器
            self.trackers[i] = MazeTracker(i, (x1, y1, x2, y2))
    
    def test_camera(self):
        """测试相机连接"""
        if self.camera and self.camera.initialized:
            frame = self.camera.grab_frame()
            if frame is not None:
                # 显示完整图像
                display = cv2.resize(frame, (800, 600))
                cv2.imshow("Camera Test", display)
                cv2.waitKey(2000)
                cv2.destroyWindow("Camera Test")
                messagebox.showinfo("成功", f"相机工作正常\n图像尺寸: {frame.shape}")
            else:
                messagebox.showerror("错误", "无法获取图像")
        else:
            messagebox.showwarning("警告", "相机未初始化")
    
    def start_annotation(self):
        """开始标注"""
        self.annotation_mode = True
        self.annotation_points = defaultdict(list)
        self.current_annotation_maze = None
        messagebox.showinfo("标注", "点击每个迷宫画布标注7个点")
    
    def on_canvas_click(self, event, maze_id):
        """处理画布点击"""
        if not hasattr(self, 'annotation_mode') or not self.annotation_mode:
            return
        
        canvas = self.canvas_widgets[maze_id]
        x, y = event.x, event.y
        
        # 记录点
        self.annotation_points[maze_id].append((x, y))
        
        # 绘制点
        canvas.create_oval(x-3, y-3, x+3, y+3, fill='red', tags="annotation")
        canvas.create_text(x+10, y, text=str(len(self.annotation_points[maze_id])), 
                          fill='blue', tags="annotation")
        
        # 如果收集够7个点
        if len(self.annotation_points[maze_id]) == 7:
            # 初始化追踪器的区域
            self.trackers[maze_id].initialize_regions(self.annotation_points[maze_id])
            self.status_labels[maze_id].config(text="状态: 已标注")
            
            # 绘制区域
            if len(self.annotation_points[maze_id]) == 7:
                points = self.annotation_points[maze_id]
                # 绘制连线
                canvas.create_line(points[0][0], points[0][1], 
                                  points[1][0], points[1][1], 
                                  fill='green', width=2, tags="region")
                canvas.create_line(points[0][0], points[0][1], 
                                  points[2][0], points[2][1], 
                                  fill='green', width=2, tags="region")
                canvas.create_line(points[0][0], points[0][1], 
                                  points[3][0], points[3][1], 
                                  fill='green', width=2, tags="region")
            
            messagebox.showinfo("完成", f"迷宫 {maze_id+1} 标注完成")
    
    def start_tracking(self):
        """开始追踪"""
        if self.running:
            return
        
        self.running = True
        self.status_text.set("追踪运行中...")
        
        # 启动采集线程
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        # 启动显示更新
        self.update_display()
    
    def stop_tracking(self):
        """停止追踪"""
        self.running = False
        self.status_text.set("追踪已停止")
    
    def capture_loop(self):
        """图像采集循环"""
        fps_counter = 0
        fps_time = time.time()
        
        while self.running:
            try:
                # 获取图像
                if self.camera and self.camera.initialized:
                    frame = self.camera.grab_frame()
                else:
                    # 模拟模式
                    frame = self.generate_simulation_frame()
                
                if frame is not None:
                    # 处理每个迷宫
                    for maze_id, tracker in self.trackers.items():
                        tracker.process_frame(frame)
                    
                    # 计算FPS
                    fps_counter += 1
                    if time.time() - fps_time > 1.0:
                        fps = fps_counter / (time.time() - fps_time)
                        self.fps_label.config(text=f"FPS: {fps:.1f}")
                        fps_counter = 0
                        fps_time = time.time()
                
                time.sleep(0.03)  # ~33ms
                
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(0.1)
    
    def generate_simulation_frame(self):
        """生成模拟图像"""
        # 创建模拟图像 (2064x3088)
        frame = np.ones((3088, 2064), dtype=np.uint8) * 128
        
        # 添加模拟幼虫
        t = time.time()
        for i in range(6):
            row = i // 3
            col = i % 3
            
            cx = col * 688 + 344
            cy = row * 1544 + 772
            
            x = int(cx + 200 * np.sin(t * 0.5 + i))
            y = int(cy + 200 * np.cos(t * 0.5 + i))
            
            cv2.circle(frame, (x, y), 20, 255, -1)
        
        return frame
    
    def update_display(self):
        """更新显示"""
        if not self.running:
            return
        
        # 更新每个迷宫的显示
        for maze_id, tracker in self.trackers.items():
            display_frame = tracker.get_display_frame()
            
            if display_frame is not None:
                # 调整大小
                display_frame = cv2.resize(display_frame, (350, 300))
                
                # 转换为RGB
                if len(display_frame.shape) == 3:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                else:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
                
                # 转换为PhotoImage
                img = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(img)
                
                # 更新画布
                canvas = self.canvas_widgets[maze_id]
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo  # 保持引用
                
                # 更新状态
                state_text = f"状态: {tracker.current_state}"
                self.status_labels[maze_id].config(text=state_text)
        
        # 继续更新
        self.root.after(100, self.update_display)
    
    def quit(self):
        """退出程序"""
        self.running = False
        
        if self.camera:
            self.camera.close()
        
        self.root.quit()
    
    def run(self):
        """运行主程序"""
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.mainloop()

# 主程序入口
if __name__ == "__main__":
    app = SimplifiedLarvaSystem()
    app.run()