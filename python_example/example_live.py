import numpy as np
import cv2

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(1):
    #smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=30)#30
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

#cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture('rtsp://192.168.31.10:554/h264/ch1/main/av_stream')
while True:
    # Get frame count
    n_frames = 3 #int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print('w: ',w)
    #print('h: ',h)
    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-2, 3), np.float32)
    #for i in range(n_frames-2):
    i = 0
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    success, curr = cap.read()
    if not success:
        break
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    # Sanity check
    assert prev_pts.shape == curr_pts.shape
    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    #Find transformation matrix
    #m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
    m,n = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    # Extract traslation
    dx = m[0,2]
    dy = m[1,2]
    # Extract rotation angle
    da = np.arctan2(m[1,0], m[0,0])
    # Store transformation
    transforms[i] = [dx,dy,da]
    # Move to next frame
    prev_gray = curr_gray
    print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)  #print('trajectory: ',trajectory)
    #Smooth trajectory using moving average filter
    smoothed_trajectory = smooth(trajectory)  #print('smoothed_trajectory: ',smoothed_trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory #print('difference: ',difference)
    # Calculate newer transformation array
    transforms_smooth = transforms + difference  #print('transforms_smooth: ',transforms_smooth)

    #for i in range(n_frames-2):
    #success, frame = cap.read()
    frame = curr
    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]
    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w,h))
    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)
    frame = cv2.resize(frame,(800,600))
    frame_stabilized = cv2.resize(frame_stabilized,(800,600))
    frame_out = cv2.hconcat([frame, frame_stabilized])
    cv2.imshow("Before and After", frame_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

