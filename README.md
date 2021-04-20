# video_stabilization
对一段视频防抖，对实时视频防抖（usb相机、rtsp相机）

英文链接：
https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

在OpenCV中通过匹配点特征实现视频防抖

在本文中，我们会介绍一种OpenCV库中点特征匹配的技术来实现一个简单的视频稳定器。在这里会讨论算法并分享一个用OpenCV中方法设计的简单稳定器的python代码。
本文是受到Nghia Ho （http://nghiaho.com/?p=2093） 和另一篇文章的启发 （https://abhitronix.github.io/2018/11/30/humanoid-AEAM-3/） 。

视频防抖是指用来降低相机动作在最终视频上影响的一系列方法。相机动作包括平移（x、y、z）和旋转（yaw、pitch、roll）。


视频防抖的应用
对视频防抖的需求跨越多个领域。

视频防抖在消费级和专业级摄像中极其重要。所以有了很多不同的机械、光学和算法方案存在。即使在静止的图片摄影中，防抖也可以在长时间曝光的手持图片中产生效果。

在内窥镜检查和结肠镜检查这样的医疗诊断中，视频防抖对确定病灶的确切位置和大小也很有帮助。

类似地，在军事应用中，飞行侦察中航空工具捕获的视频也需要防抖来帮助定位、导航、目标追踪等等。在机器人应用中也有类似的应用。

视频防抖的不同方法
视频防抖包括机械、光学、数字防抖方法。下面简要介绍：
机械视频防抖：机械的图片防抖系统使用像陀螺仪和加速计这样的特殊传感器所探测到的动作来移动图片传感器来抵消相机的运动。
光学视频防抖：在这个方法中，与移动整个相机不同的是通过移动部分镜头来防抖。这个方法使用了可移动镜头来变动的调整光线在相机镜头系统中的路径长度。
数字视频防抖：这个方法不需要特殊的传感器来估算相机的运动。该方法主要有三步：1、估算运动 2、运动平滑 3、图片合成。在第一阶段得到连续两帧之间的转换参数，第二阶段过滤掉不想要的运动，第三阶段重建防抖视频。

本文中我们会学习一个数字视频防抖算法的快速和健壮的实现。这个算法基于二维运动模型，该模型结合了欧氏转换和移动、旋转、缩放。
 


从上图可以看到，在欧式运动模型中，一个正方形可以通过不同的位置、大小、旋转转换到其它正方形。这样比仿射变换和单应性变换要严格很多，但是对运动的稳定是足够的，因为在连续两帧之间的相机移动经常很小。

使用点特征匹配实现视频防抖
这个方法用到了追踪连续两帧之间的特征点。通过追踪到的特征可以估计两帧之间的运动并进行弥补。

下面的流程图显示了基本步骤。
 

第1步：设置读取输入视频和保存输出视频。
Python
#Import numpy and OpenCV

import numpy as np
import cv2
 
#Read input video

cap = cv2.VideoCapture('video.mp4')
 
#Get frame count

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
#Get width and height of video stream

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
#Define the codec for output video

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
 
#Set up output video

out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

C++

// Read input video

VideoCapture cap("video.mp4");
 
// Get frame count

int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));
 
// Get width and height of video stream

int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
 
// Get frames per second (fps)

double fps = cap.get(CV_CAP_PROP_FPS);
 
// Set up output video

VideoWriter out("video_out.avi", CV_FOURCC('M','J','P','G'), fps, Size(2 * w, h));

第2步：读取第一帧并转成灰度图。对于视频防抖，需要捕获视频中的两帧，估算两帧之间的运动，改正运动。
Python

#Read first frame

_, prev = cap.read()
 
#Convert frame to grayscale

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

C++

// Define variable for storing frames

Mat curr, curr_gray;

Mat prev, prev_gray;
 
// Read first frame

cap >> prev;
 
// Convert frame to grayscale

cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

第3步：发现两帧之间的运动。这是算法中最重要的部分。我们会重复所有的帧，发现当前帧与前一帧之间的运动。没必要知道每个像素点的运动。欧式运动模型要求我们知道两帧上的两点就够了。实际上，发现50-100个点的运动会更好，再用他们自信的估算运动模型。


3.1 更易追踪的特征
现在的问题是需要选择哪些点来追踪。需要记住，追踪算法使用某个点周围的小片区域来追踪。这样的追踪算法遇到了光圈问题，下面的视频会解释。
所以光滑的区域不利于追踪，而有很多角落的纹理有利于追踪。幸运的是，OpenCV有一个快速的特征探测器用来探测纹理，这个非常适合追踪。这称作易于追踪的特征。

3.2 Lucas-Kanade 光流算法
一旦在前一帧中发现好的特征，就可以用Lucas-Kanade 光流算法在下一帧中追踪。Lucas-Kanade  Optical Flow 以发明者的名字命名。OpenCV中的calcOpticalFlowPyrLK函数实现了该功能。LK代表Lucas-Kanade，Pyr代表pyramid，一个图片pyramid在计算机视觉中用来在不同大小（分辨率）下处理图片。

由于多种原因，calcOpticalFlowPyrLK可能无法计算所有点的运动。例如，当前帧的特征点可能在下一帧中被遮盖。幸运的是，在下面的代码中你会看到，calcOpticalFlowPyrLK中的status标签可以用来过滤掉这些值。

3.3 运动估算
重述一下，在3.1 ，在前一帧中发现易于追踪的特征；在3.2，用光流追踪特征。换句话说，在当前帧中发现纹理的位置，已经知道前一帧中纹理的位置。就可以利用两个位置的集来计算从前一帧到当前帧的欧式转换。用estimateRigidTransform函数来实现这个转换。
一旦对运动估算完成，就可以解析出平移的x、y和旋转的角度。把这些值存储到数组中用来平滑图像。
下面的代码描述了步骤3.1到3.3。阅读的时候一定记得看看代码中的注释。


Python


#Pre-define transformation-store array

transforms = np.zeros((n_frames-1, 3), np.float32)
 
for i in range(n_frames-2):

  #Detect feature points in previous frame
  
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
 
  #Read next frame
  
  success, curr = cap.read()
  
  if not success:
  
    break
 
  #Convert to grayscale
  
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
 
  #Calculate optical flow (i.e. track feature points)
  
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
 
  #Sanity check
  
  assert prev_pts.shape == curr_pts.shape
 
  #Filter only valid points
  
  idx = np.where(status==1)[0]
  
  prev_pts = prev_pts[idx]
  
  curr_pts = curr_pts[idx]
 
  #Find transformation matrix
  
  m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
 
  #Extract traslation
  
  dx = m[0,2]
  
  dy = m[1,2]
 
  #Extract rotation angle
  
  da = np.arctan2(m[1,0], m[0,0])
 
  #Store transformation
  
  transforms[i] = [dx,dy,da]
 
  #Move to next frame
  
  prev_gray = curr_gray
 
  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))


在C++的实现中，先定义了几个用来存储运动估算向量的类。TransformParam类存储运动信息（dx---x方向的运动，dy---y方向的运动，da---角度的变动），并提供了一个getTransform方法把对应的运动转换成矩阵。


C++

struct TransformParam

{

  TransformParam() {}
  
  TransformParam(double _dx, double _dy, double _da)
  
  {
  
      dx = _dx;
      
      dy = _dy;
      
      da = _da;
      
  }
 
  double dx;
  
  double dy;
  
  double da; // angle
 
  void getTransform(Mat &T)
  
  {
  
    // Reconstruct transformation matrix accordingly to new values
    
    T.at<double>(0,0) = cos(da);
    
    T.at<double>(0,1) = -sin(da);
    
    T.at<double>(1,0) = sin(da);
    
    T.at<double>(1,1) = cos(da);
 
    T.at<double>(0,2) = dx;
    
    T.at<double>(1,2) = dy;
    
  }
  
};

下面的代码是在帧之间循环的执行步骤3.1到3.3


// Pre-define transformation-store array

  vector <TransformParam> transforms;
 
  //
  
  Mat last_T;
 
  for(int i = 1; i < n_frames-1; i++)
  
  {
  
    // Vector from previous and current feature points
    
    vector <Point2f> prev_pts, curr_pts;
 
    // Detect features in previous frame
    
    goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
 
    // Read next frame
    
    bool success = cap.read(curr);
    
    if(!success) break;
     
    // Convert to grayscale
    
    cvtColor(curr, curr_gray, COLOR_BGR2GRAY);
 
    // Calculate optical flow (i.e. track feature points)
    
    vector <uchar> status;
    
    vector <float> err;
    
    calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);
 
    // Filter only valid points
    
    auto prev_it = prev_pts.begin();
    
    auto curr_it = curr_pts.begin();
    
    for(size_t k = 0; k < status.size(); k++)
    
    {
    
        if(status[k])
        
        {
        
          prev_it++;
          
          curr_it++;
          
        }
        
        else
        
        {
        
          prev_it = prev_pts.erase(prev_it);
          
          curr_it = curr_pts.erase(curr_it);
          
        }
        
    }
 
     
    // Find transformation matrix
    
    Mat T = estimateRigidTransform(prev_pts, curr_pts, false);
 
    // In rare cases no transform is found.
    
    // We'll just use the last known good transform.
    
    if(T.data == NULL) last_T.copyTo(T);
    
    T.copyTo(last_T);
 
    // Extract traslation
    
    double dx = T.at<double>(0,2);
    
    double dy = T.at<double>(1,2);
     
    // Extract rotation angle
    
    double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
 
    // Store transformation
    
    transforms.push_back(TransformParam(dx, dy, da));
 
    // Move to next frame
    
    curr_gray.copyTo(prev_gray);
 
    cout << "Frame: " << i << "/" << n_frames << " -  Tracked points : " << prev_pts.size() << endl;
    
  }

第4步：计算帧间的平滑运动
在前面的步骤中，已经估算了帧间的运动并存储到数组中。现在需要通过累积加上前面步骤中不同运动的估算来得到运动轨迹。

4.1 计算轨迹
本步中，通过加上帧间的运动来计算轨迹。终极目标是把这个轨迹平滑掉。

Python

在python中通过numpy中的cumsum方法很容易实现。

#Compute trajectory using cumulative sum of transformations

trajectory = np.cumsum(transforms, axis=0)

C++

在C++中，定义了一个Trajectory类来存储转换参数的和。

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
 
    double x;
    double y;
    double a; // angle
};

还定义了一个函数cumsum，输入参数TransformParams的向量，通过计算不同运动的dx、dy、da的和来返回轨迹。
vector<Trajectory> cumsum(vector<TransformParam> &transforms)
{
  vector <Trajectory> trajectory; // trajectory at all frames
 
  // Accumulated frame to frame transform
 
  double a = 0;
  double x = 0;
  double y = 0;
 
  for(size_t i=0; i < transforms.size(); i++)
  {
      x += transforms[i].dx;
      y += transforms[i].dy;
      a += transforms[i].da;
 
      trajectory.push_back(Trajectory(x,y,a));
 
  }
 
  return trajectory;
}

4.2 计算平滑轨迹
在前面的步骤中已经计算了运动轨迹。所以有三个曲线（x、y、角度）来展示随着时间变化的运动。这里将展示如何平滑这三个曲线。

平滑曲线的最简单的方式是用移动平均滤波器。正如其名字的意思，一个移动平均滤波器用一个窗口中某点临近点的平均值来取代该点函数的值。可以看一个例子。

假设把曲线存储在数组c中，所以曲线上的点是c[0]…c[n-1].假设函数f做为平滑曲线，f用跨度为5的平均移动滤波器。
这个曲线的第k个元素的计算方式为：
 
可以看到，平滑曲线的值是小范围内噪声曲线的平均值。下图中左侧的是包含噪声的图表，右侧的是跨度为5的滤波器处理后的图表。

 
Python
在python的实现中，定义了一个移动平均滤波器把任何曲线做为输入，返回平滑的曲线。
def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  
  #Define the filter
  
  f = np.ones(window_size)/window_size
  
  #Add padding to the boundaries
  
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  
  #Apply convolution
  
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  
  #Remove padding
  
  curve_smoothed = curve_smoothed[radius:-radius]
  
  #return smoothed curve
  
  return curve_smoothed
还定义了一个函数，以轨迹做为输入，在三个曲线上做平滑。

def smooth(trajectory):

  smoothed_trajectory = np.copy(trajectory)
  
  #Filter the x, y and angle curves
  
  for i in range(3):
  
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory
  
  
下面是最后的使用。

#Compute trajectory using cumulative sum of transformations

trajectory = np.cumsum(transforms, axis=0)

C++
定义函数smooth用来计算平均的移动平滑轨迹。

vector <Trajectory> smooth(vector <Trajectory>& trajectory, int radius)
{
  vector <Trajectory> smoothed_trajectory;
  for(size_t i=0; i < trajectory.size(); i++) {
      double sum_x = 0;
      double sum_y = 0;
      double sum_a = 0;
      int count = 0;
 
      for(int j=-radius; j <= radius; j++) {
          if(i+j >= 0 && i+j < trajectory.size()) {
              sum_x += trajectory[i+j].x;
              sum_y += trajectory[i+j].y;
              sum_a += trajectory[i+j].a;
 
              count++;
          }
      }
 
      double avg_a = sum_a / count;
      double avg_x = sum_x / count;
      double avg_y = sum_y / count;
 
      smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
  }
 
  return smoothed_trajectory;
}

在main函数中的使用。
// Smooth trajectory using moving average filter

vector <Trajectory> smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS);

4.3 计算平滑转换
现在已经获得了平滑轨迹。在这一步，用平滑轨迹获得平滑转换，这个平滑转换可以作用于视频帧上稳定视频。通过对比平滑轨迹与原始轨迹，把差值作用到原始转换上。

Python

#Calculate difference in smoothed_trajectory and trajectory

difference = smoothed_trajectory - trajectory
 
#Calculate newer transformation array

transforms_smooth = transforms + difference

C++


vector <TransformParam> transforms_smooth;
   
  for(size_t i=0; i < transforms.size(); i++)
  {
    // Calculate difference in smoothed_trajectory and trajectory
    
    double diff_x = smoothed_trajectory[i].x - trajectory[i].x;
    double diff_y = smoothed_trajectory[i].y - trajectory[i].y;
    double diff_a = smoothed_trajectory[i].a - trajectory[i].a;
 
    // Calculate newer transformation array
    
    double dx = transforms[i].dx + diff_x;
    double dy = transforms[i].dy + diff_y;
    double da = transforms[i].da + diff_a;
 
    transforms_smooth.push_back(TransformParam(dx, dy, da));
  }
第5步：将平滑的相机运动作用到视频帧上
剩下的只需要遍历所有的帧，并把刚计算得到的转换应用到这些帧上。如果有一个公式 ，对应 的转换矩阵为：

 

Python

#Reset stream to first frame

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
#Write n_frames-1 transformed frames

for i in range(n_frames-2):
  #Read next frame
  
  success, frame = cap.read()
  
  if not success:
    break
 
  #Extract transformations from the new transformation array
  
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]
 
  #Reconstruct transformation matrix accordingly to new values
  
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
 
  #Apply affine wrapping to the given frame
  
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
 
  #Fix border artifacts
  
  frame_stabilized = fixBorder(frame_stabilized)
 
  #Write the frame to the file
  
  frame_out = cv2.hconcat([frame, frame_stabilized])
 
  #If the image is too big, resize it.
  
  if(frame_out.shape[1] &gt; 1920):
    frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
 
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(10)
  out.write(frame_out)

C++

cap.set(CV_CAP_PROP_POS_FRAMES, 1);
Mat T(2,3,CV_64F);
Mat frame, frame_stabilized, frame_out;
 
for( int i = 0; i < n_frames-1; i++)
  {
    bool success = cap.read(frame);
    if(!success) break;
     
    // Extract transform from translation and rotation angle.
    
    transforms_smooth[i].getTransform(T);
 
    // Apply affine wrapping to the given frame
    
    warpAffine(frame, frame_stabilized, T, frame.size());
 
    // Scale image to remove black border artifact
    
    fixBorder(frame_stabilized);
 
    // Now draw the original and stablised side by side for coolness
    
    hconcat(frame, frame_stabilized, frame_out);
 
    // If the image is too big, resize it.
    
    if(frame_out.cols > 1920)
    {
        resize(frame_out, frame_out, Size(frame_out.cols/2, frame_out.rows/2));
    }
 
    imshow("Before and After", frame_out);
    out.write(frame_out);
    waitKey(10);
  }

5.1 修改边界效果

当稳定一个视频的时候，有时能看到黑色的边界效果。帧的尺寸可能会缩小，所以黑色边界是可预料的。可以通过轻微的参照中心缩放视频来缓和黑色边界。

下面的函数fixBorder就是实现该功能的。用到了getRotationMatrix2D函数，因为这个函数在不移动图片中心的情况下可以旋转和缩放图片。这里不需要旋转，只要把图片缩放1.04就可以了（最大是放大4%）。

Python

def fixBorder(frame):
  s = frame.shape
  #Scale the image 4% without moving the center
  
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

C++

void fixBorder(Mat &frame_stabilized)
{
  Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols/2, frame_stabilized.rows/2), 0, 1.04);
  warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

结果：
上面展示了防抖的代码。这里的目标是显著的降低运动的影响，而不上完全消除。完全消除运动的影响这个问题留给读者去思考如何修改代码。你如果完全消除相机的运动影响会有什么副作用呢？

这里的代码只是能处理一段固定长度的视频，而不是实时的视频。如果要进行实时输出视频，需要修改多出代码，这个不在本文的范围内。更多资料请参考：https://abhitronix.github.io/2018/11/30/humanoid-AEAM-3/


优点：
1、对于低频率的运动，该方法可以起到很好的稳定效果
2、该方法消耗内存不多，所以适合嵌入式设备（像树莓派）
3、该方法对视频中的突然抖动有稳定效果

缺点：
1、	该方法对高频率扰动表现不好
2、	如果有严重的运动模糊，会导致特征追踪失败，效果也不会好
3、	该方法对卷帘快门扭曲的效果也不好

参考资料：
1、	代码：videostab.cpp
2、	数据、图片来自：https://abhitronix.github.io/
