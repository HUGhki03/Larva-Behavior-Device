#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幼虫行为追踪系统 - 启动脚本
整合UI和现有追踪代码
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
def setup_logging(log_level="INFO"):
    """设置日志配置"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"larva_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def check_dependencies():
    """检查必要的依赖"""
    required_modules = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'matplotlib': 'matplotlib',
        'nidaqmx': 'nidaqmx',
        'mvsdk': '迈德威视SDK (需要手动安装)'
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({package})")
            
    if missing:
        print("缺少以下依赖模块:")
        for m in missing:
            print(f"  - {m}")
        print("\n请使用pip安装缺少的模块，或参考文档进行安装。")
        return False
        
    return True


def create_config_file():
    """创建默认配置文件"""
    default_config = {
        "experiment": {
            "duration": 3600,
            "co2_concentration": 18,
            "training_cycles": 20,
            "training_interval": 15
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
    """集成系统 - 将UI和追踪器整合"""
    
    def __init__(self, config, simulation_mode=False):
        self.config = config
        self.simulation_mode = simulation_mode
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 导入必要的模块
        self.import_modules()
        
        # 初始化组件
        self.ui = None
        self.controller = None
        
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
        """初始化系统"""
        try:
            # 创建实验控制器
            self.controller = self.ExperimentController(
                ui_callback=self.handle_tracker_data
            )
            
            # 初始化硬件
            if not self.simulation_mode:
                num_cameras = self.controller.initialize()
                self.logger.info(f"初始化 {num_cameras} 个相机")
            else:
                self.logger.info("使用模拟模式")
                
            # 创建UI
            self.ui = self.MainUI()
            
            # 修改UI的方法，使其调用controller
            self.patch_ui_methods()
            
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
            
    def patch_ui_methods(self):
        """修改UI方法以集成控制器"""
        # 保存原始方法
        self.ui._original_start_single_maze = self.ui.start_single_maze
        self.ui._original_stop_single_maze = self.ui.stop_single_maze
        
        # 替换为集成方法
        self.ui.start_single_maze = self.start_single_maze
        self.ui.stop_single_maze = self.stop_single_maze
        self.ui.get_maze_data = self.get_maze_data
        
    def start_single_maze(self, camera_id, maze_id):
        """启动单个迷宫（集成版本）"""
        if self.simulation_mode:
            # 使用原始的模拟方法
            self.ui._original_start_single_maze(camera_id, maze_id)
        else:
            # 使用实际的追踪器
            self.controller.start_maze(camera_id, maze_id)
            
            # 更新UI状态
            if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
                self.ui.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 运行中")
                
    def stop_single_maze(self, camera_id, maze_id):
        """停止单个迷宫（集成版本）"""
        if not self.simulation_mode:
            self.controller.stop_maze(camera_id, maze_id)
            
        # 更新UI状态
        if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
            self.ui.maze_data[camera_id][maze_id]['status_label'].config(text="状态: 已停止")
            
    def get_maze_data(self, camera_id, maze_id):
        """获取迷宫数据（集成版本）"""
        if camera_id in self.ui.maze_data and maze_id in self.ui.maze_data[camera_id]:
            return self.ui.maze_data[camera_id][maze_id]
        return None
        
    def handle_tracker_data(self, data):
        """处理来自追踪器的数据"""
        # 将数据更新到UI
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
                
    def run(self):
        """运行系统"""
        try:
            self.logger.info("启动UI主循环")
            self.ui.mainloop()
            
        except KeyboardInterrupt:
            self.logger.info("用户中断")
            
        except Exception as e:
            self.logger.error(f"运行时错误: {e}")
            
        finally:
            self.shutdown()
            
    def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭系统...")
        
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