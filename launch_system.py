 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幼虫行为追踪系统启动器 - 修复版
解决相机重复初始化问题
"""

import sys
import os
import argparse
import logging
import json
import tkinter as tk
from tkinter import messagebox

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level='INFO'):
    """设置日志系统"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('larva_tracking.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_dependencies():
    """检查系统依赖"""
    dependencies = {
        'numpy': '数值计算',
        'cv2': '图像处理',
        'PIL': '图像显示',
        'matplotlib': '数据可视化',
        'tkinter': 'GUI界面'
    }
    
    missing = []
    for module, desc in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            else:
                __import__(module)
        except ImportError:
            missing.append(f"{module} ({desc})")
            
    if missing:
        print("缺少以下依赖:")
        for m in missing:
            print(f"  - {m}")
        return False
        
    # 检查可选依赖
    optional = {
        'mvsdk': '相机SDK',
        'nidaqmx': 'NI-DAQ控制'
    }
    
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"✓ {module} ({desc}) 已安装")
        except ImportError:
            print(f"! {module} ({desc}) 未安装 - 将使用模拟模式")
            
    return True


def create_config_file():
    """创建默认配置文件"""
    default_config = {
        "experiment": {
            "duration": 3600,
            "training_cycles": 20,
            "co2_concentration": 18,
            "interval": 15
        },
        "tracking": {
            "fg_threshold": 15,
            "bg_alpha": 0.02,
            "fps": 30
        },
        "data": {
            "base_path": "D:/Behaviour/data",
            "video_recording": True,
            "save_trajectories": True
        },
        "hardware": {
            "camera_count": 4,
            "maze_per_camera": 6,
            "valve_control": True
        },
        "ui": {
            "theme": "default",
            "auto_start": False,
            "show_trajectories": True
        }
    }
    
    import json
    config_file = "config.json"
    
    if not os.path.exists(config_file):
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        print(f"创建默认配置文件: {config_file}")
        
    return config_file


def load_config(config_file):
    """加载配置文件"""
    import json
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None


class IntegratedSystem:
    """集成系统 - 将UI和追踪器整合（修复相机初始化）"""
    
    def __init__(self, config, simulation_mode=False):
        self.config = config
        self.simulation_mode = simulation_mode
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 导入必要的模块
        self.import_modules()
        
        # 初始化组件
        self.ui = None
        self.controller = None
        self.cameras = {}  # 统一管理相机
        
    def import_modules(self):
        """导入模块"""
        try:
            # 导入UI模块
            from main_ui import MainUI
            self.MainUI = MainUI
            
            # 导入追踪器集成模块
            from tracker_integration import ExperimentController
            self.ExperimentController = ExperimentController
            
            self.logger.info("模块导入成功")
            
        except ImportError as e:
            self.logger.error(f"模块导入失败: {e}")
            raise
            
    def initialize(self):
        """初始化系统（修复版）"""
        try:
            # 1. 先初始化相机（只初始化一次）
            if not self.simulation_mode:
                self.init_cameras_once()
            
            # 2. 创建UI但禁用其硬件初始化
            self.ui = self.MainUI()
            
            # 3. 替换UI的init_hardware方法，避免重复初始化
            original_init_hardware = self.ui.init_hardware
            def dummy_init_hardware():
                # 只更新UI状态，不实际初始化相机
                if self.cameras:
                    self.ui.cameras = self.cameras  # 共享相机对象
                    self.ui.connection_status.config(text=f"相机: {len(self.cameras)}/4 | 阀门: 已连接")
                    self.ui.status_text.set(f"硬件初始化完成 - {len(self.cameras)}个相机已连接")
                else:
                    self.ui.status_text.set("使用模拟模式")
                    self.ui.simulation_mode = True
                    
            self.ui.init_hardware = dummy_init_hardware
            self.ui.init_hardware()  # 调用修改后的方法
            
            # 4. 创建实验控制器，也避免重复初始化
            self.controller = self.ExperimentController(
                ui_callback=self.handle_tracker_data
            )
            
            # 5. 确保控制器使用相同的相机对象
            if hasattr(self.controller, 'camera_manager') and self.cameras:
                self.controller.camera_manager.cameras = self.cameras
                
            # 6. 修改UI的方法，使其调用controller
            self.patch_ui_methods()
            
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
            
    def init_cameras_once(self):
        """只初始化一次相机"""
        try:
            import mvsdk
            from test import MindVisionCamera
            
            # 枚举设备
            DevList = mvsdk.CameraEnumerateDevice()
            self.logger.info(f"发现 {len(DevList)} 个相机")
            
            # 初始化每个相机（最多4个）
            for i in range(min(len(DevList), 4)):
                try:
                    camera = MindVisionCamera()
                    camera.open()
                    self.cameras[i] = camera
                    self.logger.info(f"相机 {i} 初始化成功")
                    
                except Exception as e:
                    self.logger.error(f"相机 {i} 初始化失败: {e}")
                    
            if not self.cameras:
                raise Exception("没有成功初始化任何相机")
                
        except Exception as e:
            self.logger.warning(f"相机初始化失败: {e}")
            self.simulation_mode = True
            
    def patch_ui_methods(self):
        """修改UI方法以集成控制器"""
        # 保存原始方法
        self.ui._original_start_single_maze = self.ui.start_single_maze
        self.ui._original_stop_single_maze = self.ui.stop_single_maze
        self.ui._original_stop_all = self.ui.stop_all
        
        # 替换为集成方法
        self.ui.start_single_maze = self.start_single_maze
        self.ui.stop_single_maze = self.stop_single_maze
        self.ui.stop_all = self.stop_all
        self.ui.get_maze_data = self.get_maze_data
        
    def start_single_maze(self, camera_id, maze_id):
        """启动单个迷宫（集成版本）"""
        if self.simulation_mode or not self.cameras:
            # 使用模拟方法
            self.ui._original_start_single_maze(camera_id, maze_id)
        else:
            # 使用实际的追踪器
            try:
                # 确保controller使用正确的相机
                if camera_id in self.cameras:
                    self.controller.start_maze(camera_id, maze_id)
                else:
                    self.logger.warning(f"相机 {camera_id} 不存在，使用模拟模式")
                    self.ui._original_start_single_maze(camera_id, maze_id)
            except Exception as e:
                self.logger.error(f"启动迷宫失败: {e}")
                # 降级到模拟模式
                self.ui._original_start_single_maze(camera_id, maze_id)
            
        # 更新UI状态
        if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
            self.ui.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 运行中")
                
    def stop_single_maze(self, camera_id, maze_id):
        """停止单个迷宫（集成版本）"""
        if not self.simulation_mode and self.controller:
            try:
                self.controller.stop_maze(camera_id, maze_id)
            except Exception as e:
                self.logger.error(f"停止迷宫失败: {e}")
                
        # 停止模拟
        key = f"{camera_id}_{maze_id}"
        if key in self.ui.trackers:
            self.ui.trackers[key] = None
            
        # 更新UI状态
        if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
            self.ui.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 已停止")
            
    def stop_all(self):
        """停止所有实验（集成版本）"""
        self.ui.running = False
        self.ui.status_text.set("正在停止所有实验...")
        
        # 停止控制器中的所有追踪器
        if not self.simulation_mode and self.controller:
            try:
                # 遍历所有可能的迷宫
                for cam_id in range(4):
                    for maze_id in range(6):
                        key = f"{cam_id}_{maze_id}"
                        if hasattr(self.controller, 'trackers') and key in self.controller.trackers:
                            self.controller.stop_maze(cam_id, maze_id)
            except Exception as e:
                self.logger.error(f"停止控制器失败: {e}")
                
        # 停止UI中的所有模拟
        for key in list(self.ui.trackers.keys()):
            cam_id, maze_id = key.split('_')
            self.stop_single_maze(int(cam_id), int(maze_id))
            
        self.ui.trackers.clear()
        self.ui.status_text.set("所有实验已停止")
            
    def get_maze_data(self, camera_id, maze_id):
        """获取迷宫数据（集成版本）"""
        if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
            return self.ui.maze_data[camera_id][maze_id]
        return None
  
  
        
    # 在 IntegratedSystem 类中修改以下方法：

    def handle_tracker_data(self, data):
        """处理来自追踪器的数据（修复版 - 确保数据正确转发）"""
        try:
            # 直接转发到UI的数据队列
            if hasattr(self.ui, 'data_queue') and self.ui.data_queue is not None:
                # 非阻塞发送
                try:
                    self.ui.data_queue.put_nowait(data)
                except:
                    # 队列满时跳过
                    pass
            else:
                # 如果队列不可用，直接更新UI数据（后备方案）
                self.update_ui_data_directly(data)
                
        except Exception as e:
            print(f"Error handling tracker data: {e}")

    def update_ui_data_directly(self, data):
        """直接更新UI数据的后备方案"""
        try:
            camera_id = data.get('camera_id')
            maze_id = data.get('maze_id')
            
            if camera_id is not None and maze_id is not None:
                if camera_id not in self.ui.maze_data:
                    self.ui.maze_data[camera_id] = {}
                if maze_id not in self.ui.maze_data[camera_id]:
                    self.ui.maze_data[camera_id][maze_id] = {}
                    
                # 更新数据
                maze_data = self.ui.maze_data[camera_id][maze_id]
                
                if 'frame' in data:
                    maze_data['frame'] = data['frame']
                if 'position' in data:
                    maze_data['position'] = data['position']
                if 'stats' in data:
                    maze_data['stats'] = data['stats']
                if 'valve_states' in data:
                    maze_data['valve_states'] = data['valve_states']
                    
        except Exception as e:
            print(f"Error updating UI data directly: {e}")

    def patch_ui_methods(self):
        """修改UI方法以集成控制器（修复版）"""
        # 保存原始方法
        self.ui._original_start_single_maze = self.ui.start_single_maze
        self.ui._original_stop_single_maze = self.ui.stop_single_maze
        self.ui._original_stop_all = self.ui.stop_all
        
        # 替换为集成方法
        self.ui.start_single_maze = self.start_single_maze
        self.ui.stop_single_maze = self.stop_single_maze
        self.ui.stop_all = self.stop_all
        self.ui.get_maze_data = self.get_maze_data
        
        # 确保控制器的数据队列指向UI的队列
        if hasattr(self.controller, 'data_queue') and hasattr(self.ui, 'data_queue'):
            self.controller.data_queue = self.ui.data_queue
    
    
               
    def run(self):
        """运行系统"""
        try:
            self.logger.info("启动UI主循环")
            self.ui.mainloop()
            
        except KeyboardInterrupt:
            self.logger.info("用户中断")
            
        except Exception as e:
            self.logger.error(f"运行时错误: {e}", exc_info=True)
            
        finally:
            self.shutdown()
            
    def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭系统...")
        
        # 停止所有实验
        if hasattr(self.ui, 'stop_all'):
            self.stop_all()
            
        # 关闭所有相机
        for camera in self.cameras.values():
            try:
                camera.close()
            except:
                pass
                
        # 关闭控制器
        if self.controller:
            self.controller.shutdown()
            
        self.logger.info("系统已关闭")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="幼虫行为追踪系统")
    parser.add_argument('--config', '-c', default='config.json', help='配置文件路径')
    parser.add_argument('--simulation', '-s', action='store_true', help='使用模拟模式')
    parser.add_argument('--log-level', '-l', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    logger.info("="*50)
    logger.info("幼虫行为追踪系统启动")
    logger.info("版本: 1.0.1 (修复版)")
    logger.info("="*50)
    
    # 检查依赖
    if not args.simulation:
        logger.info("检查系统依赖...")
        if not check_dependencies():
            logger.error("依赖检查失败")
            sys.exit(1)
    
    # 创建/加载配置文件
    if not os.path.exists(args.config):
        logger.info("配置文件不存在，创建默认配置")
        args.config = create_config_file()
        
    config = load_config(args.config)
    if config is None:
        logger.error("无法加载配置文件")
        sys.exit(1)
        
    logger.info(f"使用配置文件: {args.config}")
    
    # 创建并运行系统
    try:
        system = IntegratedSystem(config, simulation_mode=args.simulation)
        system.initialize()
        system.run()
        
    except Exception as e:
        logger.error(f"系统错误: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("系统正常退出")


if __name__ == "__main__":
    main()