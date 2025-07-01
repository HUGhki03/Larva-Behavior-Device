import time
import cv2
import numpy as np
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
from region import Region
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
#Neil/Rey: on Pi's with Buster, it's 2 arguments not 3 (so contours,_ not _,contours,_)
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

############################################################################################
##################################CAMERA INITIALIZATION####################################

resx = 450
resy = 450

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FPS, 20)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,5000)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,5000)

fgthreshold = 15

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
# 读帧的主程序
def capture_frames():
    while True:
        ret, frame = camera.read()
        if ret:
            global frame_data
            frame_data = frame
        else:
            break
        cv2.imshow('FRAME', frame)
        if cv2.waitKey(1) == 27:
            global flag
            flag = False
            break

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
            if event == cv2.EVENT_LBUTTONDOWN:  # 	indicates that the left mouse button is pressed.
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
        ret, frame = camera.read()
        frame_0 = self.readImage(frame)
        Ims = deque()                   #set up FIFO data structure for video frames
        Ims.append(frame_0)
        N = 1                           #N keeps track of how many frames have gone by
        window = 60                     #sets the length of the window over which mean is calculated
        print(self.name + 'Building Background')
        while True:
            frame = frame_data.copy()
            im = self.readImage(frame)
            if N == fps:            #add a new frame to kernel each second
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
        # naivefilenamepath = filepath + filename + '_testfile.txt'
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
        with open(valvelogpath,'w') as filehandle: #valvelog 
            filehandle.writelines("%s\n" % place for place in valvelog)
        with open(decisionlogpath,'w') as filehandle: #decision list descriptive
            filehandle.writelines("%s\n" % place for place in decisionlog)

            count = Counter(decisionlist)
            countone = count[1]
            counttwo = count[2]
            countall = counttwo + countone
            ratioone = round(countone/(countall+0.0000001),2)
            ratiotwo = round(counttwo/(countall+0.0000001),2)
            filehandle.writelines(["\nTotal Choose: %s" % countall, "\nChoose1: %s" % ratioone, "\nChoose2: %s" % ratiotwo])

        with open(positionframepath,'w') as filehandle: #larva position at each timestamp
            filehandle.writelines("%s\n" % place for place in positionframefile)
        print(self.name + "test runs done")

    def select_windows(self):  #设置相机读取的几个区域
        if self.name == 'A':
            return 650, 1100, 240, 690
        elif self.name == 'B':
            return 650, 1100, 1680, 2130
        elif self.name == 'C':
            return 650, 1100, 3120, 3570
        elif self.name == 'D':
            return 2000, 2450, 230, 680
        elif self.name == 'E':
            return 2000, 2450, 1670, 2120
        elif self.name == 'F':
            return 1950, 2400, 3120, 3570 
    
    def readImage(self, im):  # captures an image of the video capture object
        r_start, r_end, c_start, c_end = self.select_windows()
        im = im[r_start:r_end,c_start:c_end]
        im = cv2.resize(im, (resx, resy))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # converts the image to grayscale
        return im
    
    def select_valves(self):  # 读取的几个阀门端口
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
        
    def select_task(self):   #选择用哪个设备的端口
         if self.name in ['D','E','F']:
              return task1
         elif self.name in ['A','B','C']:
              return task2
        
        

# 多线程
frame_data = None
t0 = threading.Thread(target=capture_frames)
t1 = livetracker(name='A') #创建线程，需要几个加入几个，注意线程名字
t2 = livetracker(name='B')
t3 = livetracker(name='C')
t4 = livetracker(name='D')
t5 = livetracker(name='E')
t6 = livetracker(name='F')
lock = threading.Lock()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
executor.submit(t0.start) #开始读帧
time.sleep(5)
executor.submit(t1.start) #执行线程
executor.submit(t2.start)
executor.submit(t3.start)
executor.submit(t4.start)
executor.submit(t5.start)
executor.submit(t6.start)
time.sleep(1)
while flag:  #多加线程后也要改的地方
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
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
cv2.destroyAllWindows()
camera.release()
print("程序结束")
executor.shutdown(wait=False)
sys.exit()
