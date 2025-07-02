import time
import cv2
import numpy as np
import platform
import mvsdk
from statemachine import StateMachine
from statemachine import Initializer
from statemachine import Zero
from statemachine import One
from statemachine import Two
from statemachine import Three
from statemachine import Four
from statemachine import Five
from statemachine import Six
from multi_lightsvalvesobject_2odors_20240611 import LightsValves
from collections import Counter
from collections import deque
from BakCreator import BakCreator
from BakCreator import FIFO
from region_shihan import Region
import datetime
import os
import threading
import sys
import nidaqmx
import concurrent.futures


def ExponentialFilter(current, new, weight):
	if weight < 0:
		weight = 0
	if weight > 1:
		weight = 1
	if weight == 1:
		current = new.copy()
	if current.shape !=  new.shape:
		current = new
	else:
		current = cv2.addWeighted(new,weight,current,1-weight,0)
	return current


def displacement(new,prev):  # calculates the Euclidean distance between the two points
	xi = prev[0]
	xf = new[0]
	yi = prev[1]
	yf = new[1]
	return(np.sqrt((xi-xf)**2 + (yf-yi)**2))


def Centroid(cnt):
	M = cv2.moments(cnt)  # calculates the centroid of the contour
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	centroid = [cx,cy]
	return centroid


def FindLarva(img):
	contours,_ = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
	length =int(len(contours))
	areas = []
	for i in range(0,length):
		if length>0:
			area= cv2.contourArea(contours[i])
			areas.append(area)
		else:
			break
	if len(areas)>=1:
		largestCont = np.amax(areas)
		loc = areas.index(largestCont)
		larva = contours[loc]
		centroid = Centroid(larva)
		return centroid
	else:
		centroid = [-1,-1]
		return centroid


class MindVisionCamera:
    def __init__(self):
        self.hCamera = None
        self.pFrameBuffer = None
        self.cap = None
        self.monoCamera = False
        
    def open(self):
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) < 1:
            raise Exception("No camera found!")
        
        print(f"Found camera: {DevList[0].GetFriendlyName()}")
        self.hCamera = mvsdk.CameraInit(DevList[0], -1, -1)
        self.cap = mvsdk.CameraGetCapability(self.hCamera)
        
        # 判断是否为黑白相机
        self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        
        # 设置连续采集模式
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        
        # 手动曝光，50ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 50 * 1000)
        
        # 分配缓冲区
        FrameBufferSize = self.cap.sResolutionRange.iWidthMax * \
                         self.cap.sResolutionRange.iHeightMax * \
                         (1 if self.monoCamera else 3)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        # 开始采集
        mvsdk.CameraPlay(self.hCamera)
        print("Camera initialized successfully")
        
    def grab(self):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            
            # Windows系统需要翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
            
            # 转换为numpy数组
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            # 根据相机类型重塑数组
            if self.monoCamera:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth))
                # 转换为BGR格式（灰度图转3通道）
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))
            
            return frame
        except mvsdk.CameraException:
            return None
    
    def close(self):
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
        if self.pFrameBuffer:
            mvsdk.CameraAlignFree(self.pFrameBuffer)


class livetracker(threading.Thread):
    def __init__(self, name=None):
        threading.Thread.__init__(self, name=name, daemon=True)
        
    def run(self):
        # 阀门控制对象
        global lock
        v1,v2,v3,v4,v5,v6 = self.select_valves()
        lightsvalves = LightsValves(self.select_task(),v1,v2,v3,v4,v5,v6,lock,name=self.name)
        lightsvalves.offall()
        
        # 选择区域
        locs = []
        def selectRegions(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                region = [x,y]
                locs.append(region)
        number_of_regions = 7
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, selectRegions)
        while True:
            image = frame_data.copy()
            image = self.readImage(image)

            # 绘制点
            for region in locs:
                cv2.circle(image, region, 3, (0, 0, 0), -1)

            cv2.imshow(self.name, image)

            if len(locs) == number_of_regions:
                region_area =  Region(locs[0],locs)
                region_area.drawRegions(image)
                               
                cv2.imshow(self.name,image)
                cv2.waitKey(1)

                response = input("Are you satisfied with the regions you have selected? (y/n)")
                if response == "y":
                    break
                if response == "n":
                    locs = []

            key= cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow(self.name)
        initial = locs[0]
        statemachine = StateMachine(initial,locs)
        #建立背景
        fps = 10 #frame rate
        
        frame_0 = self.readImage(frame_data.copy())
        Ims = deque()
        Ims.append(frame_0)
        N = 1
        window = 60
        print(self.name + 'Building Background')
        while True:
            frame = frame_data.copy()
            im = self.readImage(frame)
            if N == fps:
                Ims.append(im)
                N = 1
            if len(Ims)==window:
                bgim = np.median(Ims, axis=0).astype(dtype = np.uint8)
                break
            N +=1
            cv2.imshow(self.name + 'background', im)
            key= cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow(self.name + 'background')
        print(self.name + "Background Built")
        # 数据文件
        testtime = 3301  #seconds
        now = datetime.datetime.now()
        now_date = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filepath = 'D:/Behaviour/data/'+ now_date +'/'
        filename = self.name + '_' + timestamp
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        decisionlist = []
        decisionlog = []
        valvelog = []
        positionframefile = []
        valvelogpath = filepath + filename + '_valvelog.txt'
        decisionlogpath = filepath + filename + '_decisionlog.txt'
        positionframepath = filepath + filename + '_positionframe.txt'
        videofile = filepath + filename + '_video.avi'
        trackvideofile = filepath + filename + '_trackvideo.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoframerate = 30
        out = cv2.VideoWriter(videofile, fourcc, videoframerate, (450, 450), False)
        out2 = cv2.VideoWriter(trackvideofile, fourcc, videoframerate, (450, 450), False)

        #开始追踪
        print(self.name + "test runs start")
        print(self.name + "test time = " + str(testtime)) 
        bgCreate = BakCreator(stacklen = 60, alpha = 0.02, bgim=bgim) 
        starttime = time.time()
        framecount = 0
        prev_pos = [0,0]
        prev_time = 0
        odorontime = [0]
        while True:
            if not flag:
                break
            new = frame_data.copy()
            new= self.readImage(new)
            out.write(new)
            fgim = cv2.subtract(new,bgim)
            _,fgthresh = cv2.threshold(fgim, fgthreshold, 255, cv2.THRESH_BINARY)
            fgthresh = cv2.morphologyEx(fgthresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            fgthresh = cv2.morphologyEx(fgthresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            bgim = bgCreate.run(new,fgthresh)
            maggot = FindLarva(fgthresh)
            elapsedtime = round(time.time()- starttime, 2)
            delta_x = displacement(prev_pos,maggot)
            delta_t = elapsedtime - prev_time
            velocity = round(delta_x/delta_t,2)
            prev_state, curr_state, nextclosest_state = statemachine.on_input(maggot)

            framecount += 1
            lightsvalves.run_test_noreward(prev_state, curr_state, nextclosest_state, decisionlist, decisionlog, elapsedtime, framecount,maggot,odorontime, valvelog)
            positionframefile.append([maggot,framecount,elapsedtime,velocity,region_area.isinRegions(maggot)])
            prev_pos = maggot
            prev_time = elapsedtime
            cv2.circle(new,tuple(maggot), 2, (0,0,255),-1)
            cv2.imshow(self.name + 'experiment', new)
            cv2.imshow(self.name + 'background image',bgim)
            out2.write(new)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            if key == ord('s'):
                input(self.name + '暂停：')
            if elapsedtime > testtime:
                break

        print("实验时长：" + str(elapsedtime))
        cv2.destroyWindow(self.name + 'experiment')
        cv2.destroyWindow(self.name + 'background image')
        lightsvalves.offall()

        # 数据写入
        with open(valvelogpath,'w') as filehandle: 
            filehandle.writelines("%s\n" % place for place in valvelog)
        with open(decisionlogpath,'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in decisionlog)

            count = Counter(decisionlist)
            countone = count[1]
            counttwo = count[2]
            countall = counttwo + countone
            ratioone = round(countone/(countall+0.0000001),2)
            ratiotwo = round(counttwo/(countall+0.0000001),2)
            filehandle.writelines(["\nTotal Choose: %s" % countall, "\nChoose1: %s" % ratioone, "\nChoose2: %s" % ratiotwo])

        with open(positionframepath,'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in positionframefile)
        print(self.name + "test runs done")
    
    def select_windows(self):
        if self.name == 'A':
            return 260, 440, 96, 276  
        elif self.name == 'B':
            return 260, 440, 672, 852    
        elif self.name == 'C':
            return 260, 440, 1248, 1428     
        elif self.name == 'D':
            return 800, 980, 92, 272            
        elif self.name == 'E':
            return 800, 980, 668, 848 
        elif self.name == 'F':
            return 780, 960, 1248, 1428
    
    def readImage(self, im):
        r_start, r_end, c_start, c_end = self.select_windows()
        im = im[r_start:r_end,c_start:c_end]
        im = cv2.resize(im, (resx, resy))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im
    
    def select_valves(self):
        if self.name == 'A':
            return [22,23,12,13,14,15]
        elif self.name == 'B':
            return list(range(16,22))
        elif self.name == 'C':
            return list(range(0,6))
        elif self.name == 'D':
            return [22,23,12,13,14,15]
        elif self.name == 'E':
            return list(range(16,22))
        elif self.name == 'F':
            return list(range(0,6))
        
    def select_task(self):
         if self.name in ['D','E','F']:
              return task1
         elif self.name in ['A','B','C']:
              return task2


# ========== 以下是主程序部分，只在直接运行时执行 ==========

def capture_frames():
    """读取相机帧的函数"""
    global frame_data, flag
    first_print = True
    while flag:
        frame = camera.grab()
        if frame is not None:
            frame_data = frame.copy()
            
            if first_print:
                print(f"Actual image size: {frame.shape}")
                first_print = False
                
            # 缩小显示尺寸
            display_frame = cv2.resize(frame, (800, 600))
            cv2.imshow('FRAME', display_frame)
        else:
            time.sleep(0.01)
            continue
            
        if cv2.waitKey(1) == 27:  # ESC键退出
            flag = False
            break


def main():
    """主函数 - 原始的多窗口追踪系统"""
    global camera, frame_data, flag, lock, task1, task2, resx, resy, fgthreshold
    
    # 全局参数
    resx = 450
    resy = 450
    fgthreshold = 15
    
    # 初始化相机
    camera = MindVisionCamera()
    camera.open()
    
    # 阀门端口设置
    task1 = nidaqmx.Task()
    task1.do_channels.add_do_chan("Dev1/port0/line0")
    task1.do_channels.add_do_chan("Dev1/port0/line1")
    task1.do_channels.add_do_chan("Dev1/port0/line2")
    task1.do_channels.add_do_chan("Dev1/port0/line3")
    task1.do_channels.add_do_chan("Dev1/port0/line4")
    task1.do_channels.add_do_chan("Dev1/port0/line5")
    task1.do_channels.add_do_chan("Dev1/port0/line6")
    task1.do_channels.add_do_chan("Dev1/port0/line7")
    task1.do_channels.add_do_chan("Dev1/port1/line0")
    task1.do_channels.add_do_chan("Dev1/port1/line1")
    task1.do_channels.add_do_chan("Dev1/port1/line2")
    task1.do_channels.add_do_chan("Dev1/port1/line3")
    task1.do_channels.add_do_chan("Dev1/port1/line4")
    task1.do_channels.add_do_chan("Dev1/port1/line5")
    task1.do_channels.add_do_chan("Dev1/port1/line6")
    task1.do_channels.add_do_chan("Dev1/port1/line7")
    task1.do_channels.add_do_chan("Dev1/port2/line0")
    task1.do_channels.add_do_chan("Dev1/port2/line1")
    task1.do_channels.add_do_chan("Dev1/port2/line2")
    task1.do_channels.add_do_chan("Dev1/port2/line3")
    task1.do_channels.add_do_chan("Dev1/port2/line4")
    task1.do_channels.add_do_chan("Dev1/port2/line5")
    task1.do_channels.add_do_chan("Dev1/port2/line6")
    task1.do_channels.add_do_chan("Dev1/port2/line7")
    
    task2 = nidaqmx.Task()
    task2.do_channels.add_do_chan("Dev2/port0/line0")
    task2.do_channels.add_do_chan("Dev2/port0/line1")
    task2.do_channels.add_do_chan("Dev2/port0/line2")
    task2.do_channels.add_do_chan("Dev2/port0/line3")
    task2.do_channels.add_do_chan("Dev2/port0/line4")
    task2.do_channels.add_do_chan("Dev2/port0/line5")
    task2.do_channels.add_do_chan("Dev2/port0/line6")
    task2.do_channels.add_do_chan("Dev2/port0/line7")
    task2.do_channels.add_do_chan("Dev2/port1/line0")
    task2.do_channels.add_do_chan("Dev2/port1/line1")
    task2.do_channels.add_do_chan("Dev2/port1/line2")
    task2.do_channels.add_do_chan("Dev2/port1/line3")
    task2.do_channels.add_do_chan("Dev2/port1/line4")
    task2.do_channels.add_do_chan("Dev2/port1/line5")
    task2.do_channels.add_do_chan("Dev2/port1/line6")
    task2.do_channels.add_do_chan("Dev2/port1/line7")
    task2.do_channels.add_do_chan("Dev2/port2/line0")
    task2.do_channels.add_do_chan("Dev2/port2/line1")
    task2.do_channels.add_do_chan("Dev2/port2/line2")
    task2.do_channels.add_do_chan("Dev2/port2/line3")
    task2.do_channels.add_do_chan("Dev2/port2/line4")
    task2.do_channels.add_do_chan("Dev2/port2/line5")
    task2.do_channels.add_do_chan("Dev2/port2/line6")
    task2.do_channels.add_do_chan("Dev2/port2/line7")
    
    flag = True
    frame_data = None
    lock = threading.Lock()
    
    # 等待相机初始化
    print("Waiting for camera initialization...")
    while frame_data is None:
        frame = camera.grab()
        if frame is not None:
            frame_data = frame.copy()
        time.sleep(0.1)
    print('Camera ready, starting threads...')
    
    # 创建线程
    t0 = threading.Thread(target=capture_frames)
    t1 = livetracker(name='A')
    t2 = livetracker(name='B')
    t3 = livetracker(name='C')
    t4 = livetracker(name='D')
    t5 = livetracker(name='E')
    t6 = livetracker(name='F')
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    executor.submit(t0.start)
    time.sleep(5)
    executor.submit(t1.start)
    executor.submit(t2.start)
    executor.submit(t3.start)
    executor.submit(t4.start)
    executor.submit(t5.start)
    executor.submit(t6.start)
    time.sleep(1)
    
    # 监控线程状态并重启
    while flag:
        if not t1.is_alive():
            t1 = livetracker(name='A')
            executor.submit(t1.start)
            time.sleep(1)
        if not t2.is_alive():
            t2 = livetracker(name='B')
            executor.submit(t2.start)
            time.sleep(1)
        if not t3.is_alive():
            t3 = livetracker(name='C')
            executor.submit(t3.start)
            time.sleep(1)
        if not t4.is_alive():
            t4 = livetracker(name='D')
            executor.submit(t4.start)
            time.sleep(1)
        if not t5.is_alive():
            t5 = livetracker(name='E')
            executor.submit(t5.start)
            time.sleep(1)
        if not t6.is_alive():
            t6 = livetracker(name='F')
            executor.submit(t6.start)
            time.sleep(1)
    
    # 等待所有线程结束
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    
    # 清理资源
    cv2.destroyAllWindows()
    camera.close()
    print("程序结束")
    executor.shutdown(wait=False)
    sys.exit()


# 只有直接运行这个文件时才执行主程序
if __name__ == "__main__":
    main()