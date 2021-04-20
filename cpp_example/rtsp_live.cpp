// 使用：
// 1. 编译原代码rtsp_live.cpp，生成可执行文件rtsp_live.out:  g++ rtsp_live.cpp -o rtsp_live.out `pkg-config --cflags --libs opencv`
// 2. 调用rtsp相机，运行rtsp_live.out:   ./rtsp_live.out 'rtsp://192.168.31.10:554/h264/ch1/main/av_stream'
//    注：'rtsp://192.168.31.10:554/h264/ch1/main/av_stream'是访问rtsp相机，不同的相机ip不一样，有的有用户名和密码，不同厂家的访问都有差别
// 3. 调用usb相机，运行rtsp_live.out:    ./rtsp_live.out /dev/video0
//    注：如果系统中有可用相机，在终端用命令ls /dev/video* 可以看到/dev/video0  /dev/video1等，想调用哪个就在参数中填写哪个
//    注：如果是mipi相机，也能看到/dev/video0  /dev/video1等，但可能调不通，需要单独修改
// 建议：
// 1、如果传过来的是图片，需要修改原代码rtsp_live.cpp
// 2、如果想进一步加快速度，把对应的计算放到gpu上应该是可以的

/*
Thanks Nghia Ho for his excellent code.
And,I modified the smooth step using a simple kalman filter .
So,It can processes live video streaming.
modified by shihailong.
email:lengkujiaai@126.com
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sys/time.h>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

//const int SMOOTHING_RADIUS = 15; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};
//
int main(int argc, char **argv)
{
	if(argc < 2) {
		cout << "./VideoStab [video.avi]" << endl;
		return 0;
	}
	// For further analysis
	//ofstream out_transform("prev_to_cur_transformation.txt");
	//ofstream out_trajectory("trajectory.txt");
	//ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	//ofstream out_new_transform("new_prev_to_cur_transformation.txt");

	VideoCapture cap(argv[1]);
	assert(cap.isOpened());

	Mat cur, cur_grey;
	Mat prev, prev_grey;

	cap >> prev;//get the first frame.ch
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
	
	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransformParam> prev_to_cur_transform; // previous to current
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	// Step 2 - Accumulate the transformations to get the image trajectory
	vector <Trajectory> trajectory; // trajectory at all frames
	//
	// Step 3 - Smooth out the trajectory using an averaging window
	vector <Trajectory> smoothed_trajectory; // trajectory at all frames
	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance 
	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <TransformParam> new_prev_to_cur_transform;
	//
	// Step 5 - Apply the new transformation to the video
	//cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2,3,CV_64F);

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
        //VideoWriter outputVideo; 
	//outputVideo.open("compare.avi" , CV_FOURCC('X','V','I','D'), 24,cvSize(cur.rows, cur.cols*2+10), true);  
	//
	int k=1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	Mat prev_grey_,cur_grey_;
	 
	while(true) {
                long int sum_time;
                int num;
                struct timeval time;
                gettimeofday(&time,NULL);
                sum_time = (time.tv_sec*1000 + time.tv_usec/1000);
                printf("begin time  s: %ld, ms: %ld\n", time.tv_sec, (time.tv_sec*1000 + time.tv_usec/1000));


		cap >> cur;
		if(cur.data == NULL) {
			break;
		}

		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <uchar> status;
		vector <float> err;

		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

		// weed out bad matches
		for(size_t i=0; i < status.size(); i++) {
			if(status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}

		// translation + rotation only
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
		//
		//prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

		//out_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		// Accumulated frame to frame transform
		x += dx;
		y += dy;
		a += da;
		//trajectory.push_back(Trajectory(x,y,a));
		//
		//out_trajectory << k << " " << x << " " << y << " " << a << endl;
		//
		z = Trajectory(x,y,a);
		//
		if(k==1){
			// intial guesses
			X = Trajectory(0,0,0); //Initial estimate,  set 0
			P =Trajectory(1,1,1); //set error variance,set 1
		}
		else
		{
			//time update（prediction）
			X_ = X; //X_(k) = X(k-1);
			P_ = P+Q; //P_(k) = P(k-1)+Q;
			// measurement update（correction）
			K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
			P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
		}
		//smoothed_trajectory.push_back(X);
		//out_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
		//-
		// target - current
		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;

		//new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
		//
		//out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		T.at<double>(0,0) = cos(da);
		T.at<double>(0,1) = -sin(da);
		T.at<double>(1,0) = sin(da);
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;

		Mat cur2;
		
		warpAffine(prev, cur2, T, cur.size());

		cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

		// Resize cur2 back to cur size, for better side by side comparison
		resize(cur2, cur2, cur.size());

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
                printf("image  rows: %d, cols: %d\n", cur.rows, cur.cols);
		prev.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		if(canvas.cols > 1920) {
			resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
		}
		//outputVideo<<canvas;
		imshow("before and after", canvas);

		waitKey(10);
		//if(waitKey(1) & 0xFF == 'q')
	        //		return 0;
		//
		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

                //cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
		k++;

	}
	return 0;
}
