#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform

class Camera:
    def __init__(self, dev_info):
        self.hCamera = None
        self.pFrameBuffer = None
        self.cap = None
        self.monoCamera = False
        self.dev_info = dev_info

    def open(self):
        # 初始化并打开相机
        try:
            self.hCamera = mvsdk.CameraInit(self.dev_info, -1, -1)
            # 获取相机特性描述
            self.cap = mvsdk.CameraGetCapability(self.hCamera)
            # 判断是黑白相机还是彩色相机
            self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)
            # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
            mvsdk.CameraSetIspOutFormat(self.hCamera, 
                (mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self.monoCamera else mvsdk.CAMERA_MEDIA_TYPE_BGR8))
            # 相机模式切换成连续采集
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            mvsdk.CameraSetAeState(self.hCamera, 0)
            # 手动曝光，曝光时间200ms
            mvsdk.CameraSetExposureTime(self.hCamera, 50 * 1000)
            # 让SDK内部取图线程开始工作
            mvsdk.CameraPlay(self.hCamera)
            # 计算RGB buffer所需的大小，并分配
            self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * \
                self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)
            return True
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False

    def close(self):
        # 关闭相机并释放资源
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
        if self.pFrameBuffer:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = None

    def grab(self):
        # 从相机抓取一帧图像
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            # 左右翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 2)
            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 
                1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            return None

    def main_loop(self):
        # 主循环，用于捕获和显示图像
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            frame = self.grab()
            if frame is not None:
                frame = cv2.resize(frame, (1544, 1032), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Press q to end", frame)
            else:
                print('failed')

# 枚举相机并初始化

DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 1:
    print("No camera was found!")
else:
    for i, dev_info in enumerate(DevList):
        print("{}: {} {}".format(i, dev_info.GetFriendlyName(), dev_info.GetPortType()))
     # 选择一个相机进行初始化
    cam = Camera(DevList[0])
    if cam.open():
         cam.main_loop()
         cam.close() 
    else:
        print("Failed to open camera.")
cv2.destroyAllWindows()


		

