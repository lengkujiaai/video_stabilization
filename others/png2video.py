import cv2
import os

img_root = '/home/nvidia/Desktop/wending/png2/'

fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
fps = 38
out = cv2.VideoWriter('video_out.avi', fourcc, fps, (1200, 500))

nameList = ['0038','0039','0040','0041','0042','0043','0044','0045','0046','0047','0048','0049','0050','0051','0052','0053','0054','0055','0056','0057','0058','0059','0060','0061','0062','0063','0064','0065','0066','0067','0068','0069','0070','0071','0072','0073','0074','0075']
for item in nameList:
    print(item)
    frame = cv2.imread(img_root + item + '.png')
    out.write(frame)
out.release()
