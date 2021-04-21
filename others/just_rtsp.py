#rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov

# use: python2 rtsp.py

import cv2
def get_img_from_camera_net():
    #cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
    #rtsp://192.168.31.10:554/h264/ch1/main/av_stream
    cap = cv2.VideoCapture("rtsp://192.168.31.10:554/h264/ch1/main/av_stream")
    
    i = 1
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        print (str(i))
        #cv2.imwrite(folder_path + str(i) + '.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #i += 1
    cap.release()
    cv2.destroyAllWindows()
 
# 
if __name__ == '__main__':
    get_img_from_camera_net()
