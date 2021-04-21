from PIL import Image
def pil_read(path,fname):
    filename = fname + '.tif'
    path = path + filename
    type = path.split('.')[-1]
    print('image type:', type)
    if type == 'tif':
        img = Image.open(path)
        pil_path = r'png/' + fname + '.png'
        img.save(pil_path,quality=95, subsampling=0)
        return pil_path

import cv2
#import cv2 as cv
def cv_read(path):
    img= cv2.imread(path)
    print(img)
    cv2.imshow('',img)
    cv2.waitKey(0)

def tif2png():
    nameList = ['0038','0039','0040','0041','0042','0043','0044','0045','0046','0047','0048','0049','0050','0051','0052','0053','0054','0055','0056','0057','0058','0059','0060','0061','0062','0063','0064','0065','0066','0067','0068','0069','0070','0071','0072','0073','0074','0075']
    #path = '/home/nvidia/Desktop/wending/tif/0038.tif'
    path = '/home/nvidia/Desktop/wending/tif/'
    for item in nameList:
        a = pil_read(path,item)
        print(a)


def pngresize():
    nameList = ['0038','0039','0040','0041','0042','0043','0044','0045','0046','0047','0048','0049','0050','0051','0052','0053','0054','0055','0056','0057','0058','0059','0060','0061','0062','0063','0064','0065','0066','0067','0068','0069','0070','0071','0072','0073','0074','0075']
    path = '/home/nvidia/Desktop/wending/'
    for item in nameList:
        print(item)
        img_path = path + 'png/' + item + '.png'
        print(img_path)
        image = cv2.imread(img_path)
        image2 = cv2.resize(image,(1200,500))
        img2_path = path + 'png2/' + item + '.png'
        cv2.imwrite(img2_path,image2)
pngresize()
