import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet
from style_cam import read_state_dict, style_frame
from imutils.video import VideoStream
from imutils import resize
import time
import os
from gstreamer import gstreamer_pipeline
from concurrent import futures

parser=argparse.ArgumentParser()
parser.add_argument('--model','-m', default='./models', help='Path to style model.')
parser.add_argument('--width','-w', default=1080, type=int,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
parser.add_argument('--gpu','-g',default=0,type=int,
                    help='If it is non-zero gpu will be used for inference.')
parser.add_argument('--full-screen','-fs', default = 1, type = int,
                    help = 'Display the output in full screen mode.')
parser.add_argument('--prep-time','-pt', default = 10, type = float,   
                    help = 'Time (in seconds) before taking a photo (count-down time).')
parser.add_argument('--view-time','-vt', default = 10, type = float,
                    help = 'Time (in seconds) the result will be viewed.')
parser.add_argument('--rotate',default = 0, type = int, 
                    help = 'if "1" will rotate the image 90 degrees CW, if "-1" will rotate the image 90 degrees CCW')
parser.add_argument('--camera', '-c', default=0,type=int,help = 'Index of USB camera. "-1" to use CSI camera.')
parser.add_argument('--save', '-s', default = None, type = str, 
                    help = 'path to where the snapshots will be saved. Default is None which means no snapshot will be saved.')
def photo_booth(path, models, width = 1080, device = torch.device('cpu'),prep_time = 10, view_time = 10,rotate = 0,camera = 0,output_path = None):
    
    # attempts to load jetcam for Jetson Nano, if fails uses normal camera.
    height = int(width*.75)
    if camera<0:
        print('Using CSI camera')
        vs = cv2.VideoCapture(gstreamer_pipeline(capture_width=width,
                                                capture_height=height,
                                                display_width=width ,
                                                display_height=height), cv2.CAP_GSTREAMER)
        time.sleep(2.0)
        img = vs.read()
        assert img[1] is not None
        
    else:
        print('Using USB camera')
        vs = VideoStream(src=camera,resolution=(width,height)).start()
    print('Warming up')
    time.sleep(2.0)

    print('Program started')
    model = TransformerNet()
    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    while True:
        preparation(vs,prep_time,rotate)
        img = vs.read()
        if type(img) == type(()):
            img=img[1]
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
        model_path = os.path.join(path,model_name)
        state_dict = read_state_dict(model_path)
        model.load_state_dict(state_dict)
        model.to(device)
        if output_path is not None:
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            filename = time.strftime('%Y_%m_%d_%H_%M_%S') + os.path.splitext(model_name)[0] + '.jpg'
            filepath = os.path.join(output_path,filename)
            cv2.imwrite(filepath,img)
        # Inference
        cv2.imshow('Output',img)
        key = cv2.waitKey(1) & 0xFF
        busy = BusyShow(img, style_frame, **{'img': img,'style_model': model,'device': device})
        output = busy.execute()
        output = output.numpy()
        # output = style_frame(img,model,device).numpy()

        # Postprocessing
        output = post_process(output)
        cv2.imshow('Output',output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        view_result(img, view_time)
                

def preparation(streamer, length = 10, rotate = 0):
    start = time.time()
    while (time.time()-start)<length:
        x = (time.time()-start)/length
        frame = streamer.read()
        if type(frame)==type(()):
            frame=frame[1]
        frame = cv2.flip(frame, 1)
        frame_show = countdown_style(frame,x)
        # rotate
        if rotate>0:
            frame_show = cv2.rotate(frame_show,cv2.ROTATE_90_CLOCKWISE)
        elif rotate<0:
            frame_show = cv2.rotate(frame_show,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('Output', frame_show)
        key = cv2.waitKey(1) & 0xFF
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
    cv2.imshow('Output',cv2.flip(image, 1))
    time.sleep(length)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        quit()

def post_process(image):
    img=np.clip(image,0,255)
    img=img.astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img[:,:,::-1]


class BusyShow():
    def __init__(self,image,func,**kwargs):
        self.func = func
        self.kwargs = kwargs
        self.image = image
    
    def busy_frame(self,t):
        img = self.image
        h,w,_ = img.shape
        xc , yc = w//2 , h//2
        rc = h//6
        ri = rc//4
        f = 2
        n = 6
        out = img.copy()
        for i in range(n):
            c = int(i/n*200)
            color = (100, 0,c)
            xi = int(xc + rc*np.cos(f*t+ i*2*np.pi/n))
            yi = int(yc + rc*np.sin(f*t+ i*2*np.pi/n))
            out = cv2.circle(out, (xi,yi), ri, color, -1)
        return out
    
    def execute(self,window_name = 'Output'):
        t0 = 0
        with futures.ThreadPoolExecutor() as executor:
            f = executor.submit(self.func,**self.kwargs)
            while f.running():
                t = time.perf_counter()-t0
                temp_img = self.busy_frame(t)
                cv2.imshow(window_name, temp_img)
                key = cv2.waitKey(1) & 0xFF
                time.sleep(0.05)
        return f.result()


if __name__=='__main__':
    args=parser.parse_args()
    models_path = args.model
    width = args.width
    prep_time = args.prep_time
    view_time = args.view_time
    rotate = args.rotate
    camera = args.camera
    path_out = args.save
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if os.path.isdir(models_path):
        models = os.listdir(models_path)
    else:
        models = [os.path.basename(models_path)]
        models_path = os.path.dirname(models_path)
    photo_booth(models_path, models, width, device,prep_time,view_time,rotate,camera,path_out)
