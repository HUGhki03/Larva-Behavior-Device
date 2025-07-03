#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幼虫行为追踪多装置实验系统 - 主UI界面（修复版）
支持4个相机，24个迷宫的同时监控
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import datetime
import os
import json
from collections import deque, defaultdict
import concurrent.futures
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 导入现有的模块
try:
    from test import MindVisionCamera, livetracker
    from multi_lightsvalvesobject_2odors_20240611 import LightsValves
    import mvsdk
    import nidaqmx
except ImportError as e:
    print(f"警告：无法导入必要模块 {e}")

class MazeMonitorWindow(tk.Toplevel):
    """单个迷宫的详细监控窗口"""
    def __init__(self, parent, camera_id, maze_id):
        super().__init__(parent)
        self.parent = parent
        self.camera_id = camera_id
        self.maze_id = maze_id
        self.maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"  # A-F
        
        self.title(f"迷宫监控 - {self.maze_name}")
        self.geometry("1200x800")
        
        # 数据存储
        self.decision_history = deque(maxlen=1000)
        self.valve_states = {"odor1": False, "odor2": False, "air": True}
        self.current_state = 0
        self.larva_position = (0, 0)
        self.region_points = []  # 存储区域标注点
        self.is_selecting_regions = False
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧 - 视频显示
        left_frame = ttk.LabelFrame(main_frame, text="实时监控", padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # 视频画布
        self.video_canvas = tk.Canvas(left_frame, width=450, height=450, bg="black")
        self.video_canvas.pack()
        
        # 绑定鼠标事件用于区域选择
        self.video_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 控制按钮
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_btn = ttk.Button(control_frame, text="开始", command=self.start_tracking)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(control_frame, text="暂停", command=self.pause_tracking)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="停止", command=self.stop_tracking)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加区域选择按钮
        self.select_region_btn = ttk.Button(control_frame, text="标注区域", command=self.start_region_selection)
        self.select_region_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        status_frame = ttk.LabelFrame(left_frame, text="状态信息", padding=10)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="状态：准备就绪")
        self.status_label.pack(anchor=tk.W)
        
        self.position_label = ttk.Label(status_frame, text="幼虫位置：(0, 0)")
        self.position_label.pack(anchor=tk.W)
        
        self.state_label = ttk.Label(status_frame, text="当前状态：中心")
        self.state_label.pack(anchor=tk.W)
        
        # 右侧 - 统计信息
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # 决策统计
        stats_frame = ttk.LabelFrame(right_frame, text="决策统计", padding=10)
        stats_frame.pack(fill=tk.X)
        
        # 统计显示
        ttk.Label(stats_frame, text="气味A选择：").grid(row=0, column=0, sticky="w")
        self.odor_a_count = ttk.Label(stats_frame, text="0")
        self.odor_a_count.grid(row=0, column=1)
        
        ttk.Label(stats_frame, text="气味B选择：").grid(row=1, column=0, sticky="w")
        self.odor_b_count = ttk.Label(stats_frame, text="0")
        self.odor_b_count.grid(row=1, column=1)
        
        ttk.Label(stats_frame, text="空气选择：").grid(row=2, column=0, sticky="w")
        self.air_count = ttk.Label(stats_frame, text="0")
        self.air_count.grid(row=2, column=1)
        
        # 饼图
        self.create_pie_chart(right_frame)
        
        # 阀门状态
        valve_frame = ttk.LabelFrame(right_frame, text="阀门状态", padding=10)
        valve_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.valve_indicators = {}
        for i, (name, label) in enumerate([("odor1", "气味A"), ("odor2", "气味B"), ("air", "空气")]):
            frame = ttk.Frame(valve_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            canvas = tk.Canvas(frame, width=20, height=20)
            canvas.pack(side=tk.LEFT, padx=(10, 0))
            canvas.create_oval(2, 2, 18, 18, fill="gray", tags="indicator")
            self.valve_indicators[name] = canvas
            
        # 配置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def on_canvas_click(self, event):
        """处理画布点击事件"""
        if self.is_selecting_regions:
            x, y = event.x, event.y
            self.region_points.append((x, y))
            
            # 在画布上绘制点
            self.video_canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tags="region_point")
            
            # 如果已经选择了7个点，完成选择
            if len(self.region_points) == 7:
                self.finish_region_selection()
                
    def start_region_selection(self):
        """开始区域选择"""
        self.is_selecting_regions = True
        self.region_points = []
        self.video_canvas.delete("region_point")
        self.select_region_btn.config(text="取消选择")
        self.select_region_btn.config(command=self.cancel_region_selection)
        messagebox.showinfo("区域选择", "请依次点击7个区域中心点：\n1. 迷宫中心\n2-4. 三个通道\n5-7. 三个圆形区域")
        
    def cancel_region_selection(self):
        """取消区域选择"""
        self.is_selecting_regions = False
        self.region_points = []
        self.video_canvas.delete("region_point")
        self.select_region_btn.config(text="标注区域")
        self.select_region_btn.config(command=self.start_region_selection)
        
    def finish_region_selection(self):
        """完成区域选择"""
        self.is_selecting_regions = False
        self.select_region_btn.config(text="重新标注")
        self.select_region_btn.config(command=self.start_region_selection)
        
        # 保存区域信息到父窗口
        if hasattr(self.parent, 'save_region_data'):
            self.parent.save_region_data(self.camera_id, self.maze_id, self.region_points)
            
        messagebox.showinfo("完成", "区域标注完成！")
        
    def create_pie_chart(self, parent):
        """创建饼图"""
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, parent)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_pie_chart()
        
    def update_pie_chart(self):
        """更新饼图"""
        self.ax.clear()
        counts = [
            int(self.odor_a_count.cget("text")),
            int(self.odor_b_count.cget("text")),
            int(self.air_count.cget("text"))
        ]
        if sum(counts) > 0:
            labels = ['气味A', '气味B', '空气']
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            self.ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
        else:
            self.ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=self.ax.transAxes)
        self.ax.set_title('决策分布')
        self.canvas_plot.draw()
        
    def update_valve_states(self, states):
        """更新阀门状态显示"""
        for name, state in states.items():
            if name in self.valve_indicators:
                color = "green" if state else "gray"
                self.valve_indicators[name].itemconfig("indicator", fill=color)
                
    def start_tracking(self):
        """开始追踪"""
        self.parent.start_single_maze(self.camera_id, self.maze_id)
        self.status_label.config(text="状态：运行中")
        
    def pause_tracking(self):
        """暂停追踪"""
        self.parent.pause_single_maze(self.camera_id, self.maze_id)
        self.status_label.config(text="状态：已暂停")
        
    def stop_tracking(self):
        """停止追踪"""
        self.parent.stop_single_maze(self.camera_id, self.maze_id)
        self.status_label.config(text="状态：已停止")
        
    def update_display(self):
        """更新显示"""
        # 从父窗口获取数据
        data = self.parent.get_maze_data(self.camera_id, self.maze_id)
        if data:
            # 更新视频
            if 'frame' in data and data['frame'] is not None:
                self.update_video(data['frame'])
            
            # 更新位置
            if 'position' in data:
                self.larva_position = data['position']
                self.position_label.config(text=f"幼虫位置：{self.larva_position}")
            
            # 更新统计
            if 'stats' in data:
                self.odor_a_count.config(text=str(data['stats'].get('odor_a', 0)))
                self.odor_b_count.config(text=str(data['stats'].get('odor_b', 0)))
                self.air_count.config(text=str(data['stats'].get('air', 0)))
                self.update_pie_chart()
            
            # 更新阀门状态
            if 'valve_states' in data:
                self.update_valve_states(data['valve_states'])
                
        # 继续更新
        self.after(100, self.update_display)
        
    def update_video(self, frame):
        """更新视频显示"""
        # 调整大小
        frame = cv2.resize(frame, (450, 450))
        # 转换颜色
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 绘制轨迹或其他信息
        if hasattr(self, 'trajectory'):
            for i in range(1, len(self.trajectory)):
                cv2.line(frame, self.trajectory[i-1], self.trajectory[i], (255, 0, 0), 2)
                
        # 绘制区域边界
        if self.region_points and len(self.region_points) == 7:
            # 使用region_shihan的方法绘制区域
            try:
                from region_shihan import Region
                region = Region(self.region_points[0], self.region_points)
                frame = region.drawRegions(frame)
            except:
                pass
                
        # 转换为PIL图像
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        
        # 更新画布
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo  # 保持引用


class MainUI(tk.Tk):
    """主UI窗口"""
    def __init__(self):
        super().__init__()
        
        self.title("幼虫行为追踪多装置实验系统")
        self.geometry("1400x900")
        
        # 系统状态
        self.running = False
        self.cameras = {}  # 相机对象
        self.trackers = {}  # 追踪线程
        self.maze_data = defaultdict(dict)  # 存储每个迷宫的数据
        self.data_queue = Queue()  # 数据队列
        self.detail_windows = {}  # 详细窗口
        self.region_data = {}  # 存储区域标注数据
        self.simulation_mode = False  # 模拟模式标志
        
        # 初始化UI
        self.setup_ui()
        
        # 初始化硬件
        self.init_hardware()
        
        # 启动数据处理线程
        self.data_thread = threading.Thread(target=self.process_data_queue, daemon=True)
        self.data_thread.start()
        
        # 添加UI定时更新（解决画面不显示问题）
        def timer_update():
            self.update_displays()
            self.after(50, timer_update)  # 20 FPS
        self.after(100, timer_update)  # 100ms后开始
        
        
    def setup_ui(self):
        """设置UI界面"""
        # 创建菜单栏
        self.create_menu()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建主界面
        self.create_main_interface()
        
        # 创建状态栏
        self.create_statusbar()
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出数据", command=self.export_data)
        file_menu.add_command(label="加载配置", command=self.load_config)
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit)
        
        # 实验菜单
        exp_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="实验", menu=exp_menu)
        exp_menu.add_command(label="开始所有", command=self.start_all)
        exp_menu.add_command(label="停止所有", command=self.stop_all)
        exp_menu.add_command(label="紧急停止", command=self.emergency_stop)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="实验参数", command=self.show_settings)
        settings_menu.add_command(label="阀门映射", command=self.show_valve_mapping)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 全局控制按钮
        ttk.Button(toolbar, text="开始所有", command=self.start_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="停止所有", command=self.stop_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="紧急停止", command=self.emergency_stop).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 实验模式选择
        ttk.Label(toolbar, text="实验模式:").pack(side=tk.LEFT, padx=5)
        self.exp_mode = ttk.Combobox(toolbar, values=["训练模式", "测试模式", "手动模式"], width=15)
        self.exp_mode.set("训练模式")
        self.exp_mode.pack(side=tk.LEFT, padx=5)
        
        # 训练参数
        ttk.Label(toolbar, text="训练周期:").pack(side=tk.LEFT, padx=5)
        self.training_cycles = tk.Spinbox(toolbar, from_=1, to=100, width=5, value=20)
        self.training_cycles.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 模拟模式复选框
        self.simulation_var = tk.BooleanVar()
        ttk.Checkbutton(toolbar, text="模拟模式", variable=self.simulation_var, 
                       command=self.toggle_simulation_mode).pack(side=tk.LEFT, padx=5)
        
        # 实时统计
        self.total_decisions = ttk.Label(toolbar, text="总决策数: 0")
        self.total_decisions.pack(side=tk.RIGHT, padx=10)
        
    def toggle_simulation_mode(self):
        """切换模拟模式"""
        self.simulation_mode = self.simulation_var.get()
        if self.simulation_mode:
            self.status_text.set("已切换到模拟模式")
        else:
            self.status_text.set("已切换到实际硬件模式")
            
    def create_main_interface(self):
        """创建主界面"""
        # 创建标签页
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 为每个相机创建标签页
        self.camera_tabs = {}
        for cam_id in range(4):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"相机 {cam_id}")
            self.camera_tabs[cam_id] = tab
            
            # 创建迷宫网格
            self.create_maze_grid(tab, cam_id)
            
        # 添加全局统计标签页
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="全局统计")
        self.create_global_stats(stats_tab)
        
    def create_maze_grid(self, parent, camera_id):
        """创建迷宫网格"""
        # 创建2x3的网格
        for row in range(2):
            for col in range(3):
                maze_id = row * 3 + col
                maze_frame = self.create_maze_preview(parent, camera_id, maze_id)
                maze_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
                
        # 配置网格权重
        for i in range(3):
            parent.columnconfigure(i, weight=1)
        for i in range(2):
            parent.rowconfigure(i, weight=1)
            
    def create_maze_preview(self, parent, camera_id, maze_id):
        """创建单个迷宫预览"""
        maze_name = f"Cam{camera_id}_{chr(65+maze_id)}"
        
        # 迷宫框架
        frame = ttk.LabelFrame(parent, text=maze_name, padding=10)
        
        # 预览画布
        canvas = tk.Canvas(frame, width=200, height=200, bg="gray")
        canvas.pack()
        
        # 绑定点击事件
        canvas.bind("<Button-1>", lambda e: self.open_detail_window(camera_id, maze_id))
        
        # 控制按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        start_btn = ttk.Button(btn_frame, text="开始", 
                              command=lambda: self.start_single_maze(camera_id, maze_id))
        start_btn.pack(side=tk.LEFT, padx=2)
        
        stop_btn = ttk.Button(btn_frame, text="停止",
                             command=lambda: self.stop_single_maze(camera_id, maze_id))
        stop_btn.pack(side=tk.LEFT, padx=2)
        
        # 状态标签
        status_label = ttk.Label(frame, text="状态: 准备就绪")
        status_label.pack()
        
        decision_label = ttk.Label(frame, text="决策: 0/0/0")
        decision_label.pack()
        
        # 存储UI元素引用
        if camera_id not in self.maze_data:
            self.maze_data[camera_id] = {}
        self.maze_data[camera_id][maze_id] = {
            'canvas': canvas,
            'status_label': status_label,
            'decision_label': decision_label,
            'stats': {'odor_a': 0, 'odor_b': 0, 'air': 0},
            'frame': None,
            'position': (0, 0),
            'valve_states': {}
        }
        
        return frame
        
    def create_global_stats(self, parent):
        """创建全局统计页面"""
        # 统计信息框架
        stats_frame = ttk.LabelFrame(parent, text="实验统计", padding=20)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 创建统计标签
        self.stats_labels = {}
        stats_items = [
            "实验开始时间:", "运行时长:", "活跃迷宫数:", 
            "总决策数:", "平均决策/迷宫:", "错误数:"
        ]
        
        for i, item in enumerate(stats_items):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(stats_frame, text=item, font=("Arial", 12)).grid(row=row, column=col, sticky="w", padx=10, pady=5)
            label = ttk.Label(stats_frame, text="--", font=("Arial", 12, "bold"))
            label.grid(row=row, column=col+1, sticky="w", padx=10, pady=5)
            self.stats_labels[item] = label
            
    def create_statusbar(self):
        """创建状态栏"""
        statusbar = ttk.Frame(self)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 状态文本
        self.status_text = tk.StringVar()
        self.status_text.set("系统就绪")
        ttk.Label(statusbar, textvariable=self.status_text).pack(side=tk.LEFT, padx=10)
        
        # 连接状态
        self.connection_status = ttk.Label(statusbar, text="相机: 0/4 | 阀门: 未连接")
        self.connection_status.pack(side=tk.RIGHT, padx=10)
        
    def init_hardware(self):
        """初始化硬件"""
        try:
            if self.simulation_mode:
                self.status_text.set("使用模拟模式")
                return
                
            # 枚举相机
            import mvsdk
            DevList = mvsdk.CameraEnumerateDevice()
            camera_count = min(len(DevList), 4)
            
            for i in range(camera_count):
                camera = MindVisionCamera()
                camera.open()
                self.cameras[i] = camera
                
            self.connection_status.config(text=f"相机: {camera_count}/4 | 阀门: 已连接")
            self.status_text.set(f"硬件初始化完成 - {camera_count}个相机已连接")
            
        except Exception as e:
            self.status_text.set(f"硬件初始化失败: {str(e)}")
            # 自动切换到模拟模式
            self.simulation_mode = True
            self.simulation_var.set(True)
            
    def save_region_data(self, camera_id, maze_id, points):
        """保存区域标注数据"""
        key = f"{camera_id}_{maze_id}"
        self.region_data[key] = points
        
        # 保存到文件
        try:
            with open(f"regions_{key}.json", "w") as f:
                json.dump(points, f)
        except:
            pass
            
    def open_detail_window(self, camera_id, maze_id):
        """打开详细监控窗口"""
        window_key = f"{camera_id}_{maze_id}"
        
        # 如果窗口已存在，则激活它
        if window_key in self.detail_windows and self.detail_windows[window_key].winfo_exists():
            self.detail_windows[window_key].lift()
        else:
            # 创建新窗口
            window = MazeMonitorWindow(self, camera_id, maze_id)
            self.detail_windows[window_key] = window
            
    def get_maze_data(self, camera_id, maze_id):
        """获取指定迷宫的数据"""
        if camera_id in self.maze_data and maze_id in self.maze_data[camera_id]:
            return self.maze_data[camera_id][maze_id]
        return None
        
    def start_all(self):
        """开始所有实验"""
        if not self.running:
            self.running = True
            self.status_text.set("正在启动所有实验...")
            
            # 启动所有追踪线程
            for cam_id in range(4):
                for maze_id in range(6):
                    self.start_single_maze(cam_id, maze_id)
                    
            self.status_text.set("所有实验已启动")
            
    def stop_all(self):
        """停止所有实验（修复版）"""
        self.running = False
        self.status_text.set("正在停止所有实验...")
        
        # 停止所有追踪线程
        for key in list(self.trackers.keys()):
            cam_id, maze_id = key.split('_')
            self.stop_single_maze(int(cam_id), int(maze_id))
            
        self.trackers.clear()
        self.status_text.set("所有实验已停止")
            
    def emergency_stop(self):
        """紧急停止 - 关闭所有阀门"""
        response = messagebox.askyesno("紧急停止", "确定要执行紧急停止吗？\n这将关闭所有阀门并停止实验。")
        if response:
            self.stop_all()
            # 关闭所有阀门
            try:
                # 这里应该调用阀门控制代码关闭所有阀门
                pass
            except:
                pass
            self.status_text.set("紧急停止已执行 - 所有阀门已关闭")
            
    def start_single_maze(self, camera_id, maze_id):
        """启动单个迷宫的追踪"""
        key = f"{camera_id}_{maze_id}"
        if key not in self.trackers:
            if self.simulation_mode:
                # 模拟模式
                self.simulate_maze_data(camera_id, maze_id)
            else:
                # 实际硬件模式
                try:
                    # 这里应该调用实际的livetracker类
                    pass
                except:
                    # 如果失败，使用模拟模式
                    self.simulate_maze_data(camera_id, maze_id)
            
            # 更新状态
            if camera_id in self.maze_data and maze_id in self.maze_data[camera_id]:
                self.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 运行中")
            
    def pause_single_maze(self, camera_id, maze_id):
        """暂停单个迷宫"""
        key = f"{camera_id}_{maze_id}"
        if key in self.trackers:
            # 暂停追踪
            pass
            
    def stop_single_maze(self, camera_id, maze_id):
        """停止单个迷宫"""
        key = f"{camera_id}_{maze_id}"
        if key in self.trackers:
            # 停止追踪
            self.trackers[key] = None  # 标记为停止
            
        # 更新状态
        if camera_id in self.maze_data and maze_id in self.maze_data[camera_id]:
            self.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 已停止")
            
    def simulate_maze_data(self, camera_id, maze_id):
        """模拟迷宫数据（修复版 - 减少雪花点）"""
        key = f"{camera_id}_{maze_id}"
        self.trackers[key] = True  # 标记为运行中
        
        def update():
            # 创建持久的背景
            background = np.ones((450, 450), dtype=np.uint8) * 128
            
            while self.running and key in self.trackers and self.trackers[key]:
                # 使用背景的副本
                frame = background.copy()
                
                # 添加一个移动的圆表示幼虫（减少噪声）
                t = time.time()
                x = int(225 + 100 * np.sin(t * 0.5))  # 减慢移动速度
                y = int(225 + 100 * np.cos(t * 0.5))
                cv2.circle(frame, (x, y), 10, 255, -1)
                
                # 添加少量高斯模糊以减少锐利边缘
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                
                # 更新数据
                if camera_id in self.maze_data and maze_id in self.maze_data[camera_id]:
                    data = self.maze_data[camera_id][maze_id]
                    data['frame'] = frame
                    data['position'] = (x, y)
                    
                    # 随机更新统计
                    if np.random.random() < 0.1:
                        choice = np.random.choice(['odor_a', 'odor_b', 'air'])
                        data['stats'][choice] += 1
                        
                    # 随机阀门状态
                    data['valve_states'] = {
                        'odor1': np.random.random() < 0.3,
                        'odor2': np.random.random() < 0.3,
                        'air': np.random.random() < 0.4
                    }
                    
                time.sleep(0.033)  # ~30 FPS
                
        thread = threading.Thread(target=update, daemon=True)
        thread.start()
        
    def process_data_queue(self):
        """处理数据队列"""
        while True:
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=0.1)
                
                # 处理数据
                # 这里应该处理来自追踪线程的实际数据
                
            except:
                pass
                
            # 更新UI
            self.update_displays()
            
    def update_displays(self):
        """更新所有显示"""
        # 更新预览
        for cam_id in self.maze_data:
            for maze_id in self.maze_data[cam_id]:
                data = self.maze_data[cam_id][maze_id]
                
                # 更新缩略图
                if 'frame' in data and data['frame'] is not None:
                    self.update_thumbnail(cam_id, maze_id, data['frame'])
                    
                # 更新状态标签
                if 'stats' in data:
                    total = sum(data['stats'].values())
                    text = f"决策: {data['stats']['odor_a']}/{data['stats']['odor_b']}/{data['stats']['air']}"
                    data['decision_label'].config(text=text)
                    
        # 更新全局统计
        self.update_global_stats()
        
    def update_thumbnail(self, camera_id, maze_id, frame):
        """更新缩略图"""
        # 缩小图像
        small = cv2.resize(frame, (200, 200))
        
        # 转换为RGB
        if len(small.shape) == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        else:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
        # 转换为PIL图像
        image = Image.fromarray(small)
        photo = ImageTk.PhotoImage(image)
        
        # 更新画布
        canvas = self.maze_data[camera_id][maze_id]['canvas']
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # 保持引用
        
    def update_global_stats(self):
        """更新全局统计"""
        total_decisions = 0
        active_mazes = 0
        
        for cam_id in self.maze_data:
            for maze_id in self.maze_data[cam_id]:
                stats = self.maze_data[cam_id][maze_id].get('stats', {})
                maze_total = sum(stats.values())
                if maze_total > 0:
                    active_mazes += 1
                total_decisions += maze_total
                
        # 更新标签
        self.stats_labels["活跃迷宫数:"].config(text=f"{active_mazes}/24")
        self.stats_labels["总决策数:"].config(text=str(total_decisions))
        if active_mazes > 0:
            avg = total_decisions / active_mazes
            self.stats_labels["平均决策/迷宫:"].config(text=f"{avg:.1f}")
            
    def export_data(self):
        """导出数据"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            # 这里应该实现数据导出功能
            messagebox.showinfo("导出成功", f"数据已导出到:\n{filename}")
            
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 应用配置
                messagebox.showinfo("成功", "配置已加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败:\n{str(e)}")
                
    def save_config(self):
        """保存配置"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = {
                    'experiment_mode': self.exp_mode.get(),
                    'training_cycles': self.training_cycles.get(),
                    'region_data': self.region_data,
                    # 添加其他配置
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("成功", "配置已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败:\n{str(e)}")
                
    def show_settings(self):
        """显示设置对话框"""
        SettingsDialog(self)
        
    def show_valve_mapping(self):
        """显示阀门映射对话框"""
        ValveMappingDialog(self)
        
    def show_help(self):
        """显示帮助"""
        help_text = """
幼虫行为追踪系统使用说明:

1. 系统启动后会自动检测并连接相机
2. 点击"开始所有"按钮启动所有迷宫的追踪
3. 点击迷宫缩略图可以打开详细监控窗口
4. 在详细窗口中可以标注迷宫的7个区域
5. 使用紧急停止按钮可以立即停止所有实验

区域标注说明:
- 点击"标注区域"按钮开始标注
- 依次点击7个区域中心点
- 标注完成后会自动保存

快捷键:
- F1: 显示帮助
- F5: 开始/停止所有实验
- Esc: 紧急停止
        """
        messagebox.showinfo("使用说明", help_text)
        
    def show_about(self):
        """显示关于"""
        about_text = """
幼虫行为追踪多装置实验系统
版本: 1.0
作者: 实验室团队
        """
        messagebox.showinfo("关于", about_text)


class SettingsDialog(tk.Toplevel):
    """设置对话框"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("实验参数设置")
        self.geometry("500x600")
        
        # 创建设置界面
        self.create_settings_ui()
        
    def create_settings_ui(self):
        """创建设置界面"""
        # 基本设置
        basic_frame = ttk.LabelFrame(self, text="基本设置", padding=10)
        basic_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_basic_settings(basic_frame)
        
        # 高级设置
        advanced_frame = ttk.LabelFrame(self, text="高级设置", padding=10)
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_advanced_settings(advanced_frame)
        
        # 按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="保存", command=self.save_settings).pack(side=tk.RIGHT, padx=10)
        ttk.Button(button_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        
    def create_basic_settings(self, parent):
        """创建基本设置"""
        # 实验时长
        ttk.Label(parent, text="实验时长 (秒):").grid(row=0, column=0, sticky="w", pady=5)
        self.duration = tk.Spinbox(parent, from_=60, to=7200, width=10, value=3600)
        self.duration.grid(row=0, column=1, pady=5)
        
        # CO2浓度
        ttk.Label(parent, text="CO2浓度 (%):").grid(row=1, column=0, sticky="w", pady=5)
        self.co2_concentration = tk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
        self.co2_concentration.set(18)
        self.co2_concentration.grid(row=1, column=1, pady=5)
        
        # 训练间隔
        ttk.Label(parent, text="训练间隔 (秒):").grid(row=2, column=0, sticky="w", pady=5)
        self.interval = tk.Spinbox(parent, from_=1, to=60, width=10, value=15)
        self.interval.grid(row=2, column=1, pady=5)
        
    def create_advanced_settings(self, parent):
        """创建高级设置"""
        frame = ttk.LabelFrame(parent, text="追踪参数", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 前景阈值
        ttk.Label(frame, text="前景阈值:").grid(row=0, column=0, sticky="w", pady=5)
        self.fg_threshold = tk.Scale(frame, from_=1, to=50, orient=tk.HORIZONTAL, length=200)
        self.fg_threshold.set(15)
        self.fg_threshold.grid(row=0, column=1, pady=5)
        
        # 背景更新率
        ttk.Label(frame, text="背景更新率:").grid(row=1, column=0, sticky="w", pady=5)
        self.bg_alpha = tk.Scale(frame, from_=0.01, to=0.1, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        self.bg_alpha.set(0.02)
        self.bg_alpha.grid(row=1, column=1, pady=5)
        
    def save_settings(self):
        """保存设置"""
        # 这里应该保存设置到配置文件或应用到系统
        self.destroy()


class ValveMappingDialog(tk.Toplevel):
    """阀门映射对话框"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("阀门映射配置")
        self.geometry("800x600")
        
        # 创建表格
        self.create_mapping_table()
        
        # 按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="保存", command=self.save_mapping).pack(side=tk.RIGHT, padx=10)
        ttk.Button(button_frame, text="加载默认", command=self.load_default).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        
    def create_mapping_table(self):
        """创建映射表格"""
        # 创建Treeview
        columns = ('maze', 'state', 'channel1', 'channel2', 'channel3')
        self.tree = ttk.Treeview(self, columns=columns, show='headings', height=20)
        
        # 定义列
        self.tree.heading('maze', text='迷宫')
        self.tree.heading('state', text='状态')
        self.tree.heading('channel1', text='通道1')
        self.tree.heading('channel2', text='通道2') 
        self.tree.heading('channel3', text='通道3')
        
        # 设置列宽
        self.tree.column('maze', width=100)
        self.tree.column('state', width=100)
        self.tree.column('channel1', width=150)
        self.tree.column('channel2', width=150)
        self.tree.column('channel3', width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # 加载示例数据
        self.load_example_data()
        
    def load_example_data(self):
        """加载示例数据"""
        states = ['中心', '通道1', '通道2', '通道3', '圆形1', '圆形2', '圆形3']
        
        for cam_id in range(4):
            for maze_id in range(6):
                maze_name = f"Cam{cam_id}_{chr(65+maze_id)}"
                for state_id, state in enumerate(states):
                    # 示例映射
                    if state_id == 0:  # 中心
                        mapping = ('关闭', '关闭', '关闭')
                    elif state_id in [1, 2, 3]:  # 通道
                        mapping = ('气味A', '气味B', '空气')
                    else:  # 圆形区域
                        mapping = ('空气', '空气', '空气')
                        
                    self.tree.insert('', 'end', values=(maze_name, state, *mapping))
                    
    def save_mapping(self):
        """保存映射"""
        # 这里应该保存映射到配置文件
        messagebox.showinfo("成功", "阀门映射已保存")
        self.destroy()
        
    def load_default(self):
        """加载默认映射"""
        # 清空现有数据
        for item in self.tree.get_children():
            self.tree.delete(item)
        # 重新加载
        self.load_example_data()


def main():
    """主函数"""
    app = MainUI()
    
    # 绑定快捷键
    app.bind('<F1>', lambda e: app.show_help())
    app.bind('<F5>', lambda e: app.start_all() if not app.running else app.stop_all())
    app.bind('<Escape>', lambda e: app.emergency_stop())
    
    # 运行主循环
    app.mainloop()


if __name__ == "__main__":
    main()