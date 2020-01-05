import argparse
import numpy as np
import cv2
from imutils import resize
import time
import os
import requests
try:
    from imutils.video.pivideostream import PiVideoStream
except:
    from imutils.video import VideoStream

parser=argparse.ArgumentParser()
parser.add_argument('--url','-u',required=True,help = 'Server url')
parser.add_argument('--width','-w', default=1080, type=int,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
parser.add_argument('--full-screen','-fs', default = 1, type = int,
                    help = 'Display the output in full screen mode.')
parser.add_argument('--prep-time','-pt', default = 10, type = float,   
                    help = 'Time (in seconds) before taking a photo (count-down time).')
parser.add_argument('--view-time','-vt', default = 10, type = float,
                    help = 'Time (in seconds) the result will be viewed.')
parser.add_argument('--rotate',default = 0, type = int, 
                    help = 'if "1" will rotate the image 90 degrees CW, if "-1" will rotate the image 90 degrees CCW')
parser.add_argument('--raspi','-pi',default = 0, type = int,
                    help = 'Use value "1" if using on Raspberry Pi.')
def photo_booth(url, path, models, width = 1080,prep_time = 10, view_time = 10,rotate = 0,raspi=False):
    
    if raspi:
        vs = PiVideoStream().start()
    else:
        vs = VideoStream(src=0).start()
    print('Warming up')
    time.sleep(2.0)
    print('Program started')
    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    while True:
        preparation(vs,prep_time,rotate)
        img = vs.read()
        img = cv2.flip(img, 1)
        img = resize(img,width)

        # rotate
        if rotate>0:
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        elif rotate<0:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Choosing and loading the model
        model_name = np.random.choice(models)
        print('Using {}'.format(model_name))
        # model_path = os.path.join(path,model_name)

        # Show the captured frame
        cv2.imshow('Output',img)
        key = cv2.waitKey(1) & 0xFF
        cv2.imwrite('./images/content-images/snapshot.jpg',img)
        # Inference
        start = time.time()
        # url=" http://0.0.0.0:8080/"
        req = {'style': os.path.splitext(model_name)[0]}
        with open('./images/content-images/snapshot.jpg','rb') as fp:
            filename = 'snapshot.jpg'
            resp = requests.post(url,files = [('file', (filename, fp, 'application/octet'))]
                                ,data=req)
        end = time.time()
        print('response received in {} seconds'.format(end-start))
        with open('./images/output-images/output.jpg','wb') as fp:
            fp.write(resp.content)
        

        # Postprocessing
        output = cv2.imread('./images/output-images/output.jpg')
        cv2.imshow('Output',output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        view_result(output, view_time)
                

def preparation(streamer, length = 10, rotate = 0):
    start = time.time()
    while (time.time()-start)<length:
        x = (time.time()-start)/length
        frame = streamer.read()
        frame = cv2.flip(frame, 1)
        frame_show = countdown_style(frame,x)
        # rotate
        if rotate>0:
            frame_show = cv2.rotate(frame_show,cv2.ROTATE_90_CLOCKWISE)
        elif rotate<0:
            frame_show = cv2.rotate(frame_show,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('Output', frame_show)
        key = cv2.waitKey(1) & 0xFF
            # time.sleep(.1)
        if key == ord("q"):
            quit()

    return frame

def countdown_style(img,x):
    h, w, _ = img.shape
    ci = np.array([0,255,0])
    cf = np.array([0,0,255])
    c = ci*(1-x)+cf*x
    c.astype(np.uint8)
    yi= np.int32(h*x)
    xf = w//10
    output = img
    for i in range(3):
        output[yi:,:xf,i]=c[i]
    return output

def view_result(image, length = 10):
    cv2.imshow('Output',image)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(length)
    
    if key == ord("q"):
        quit()

    

if __name__=='__main__':
    args=parser.parse_args()
    models_path = './models'
    width = args.width
    prep_time = args.prep_time
    view_time = args.view_time
    url=args.url
    rotate = args.rotate
    raspi=args.raspi
    if os.path.isdir(models_path):
        models = os.listdir(models_path)
    else:
        models = [os.path.basename(models_path)]
        models_path = os.path.dirname(models_path)
    photo_booth(url,models_path, models, width,prep_time,view_time,rotate,raspi>0)
