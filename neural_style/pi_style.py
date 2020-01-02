import argparse
import time
import cv2
import torch
from torchvision import transforms
import utils
import re
from transformer_net import TransformerNet
import pathlib
# from imutils.video import VideoStream
from imutils.video.pivideostream import PiVideoStream
from imutils import resize
import numpy as np
import os
import itertools

device=torch.device('cpu')
parser=argparse.ArgumentParser()

parser.add_argument('--model', default='./models', help='Path to style model.')
parser.add_argument('--width', default=320, type=float,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
parser.add_argument('--gpu',default=0,type=int,
                    help='If it is non-zero gpu will be used for inference.')

class Timer():
    def __init__(self):
        self.end = time.time()
        self.start=self.end

    def __call__(self,show=True):
        self.start = self.end
        self.end = time.time()
        if show:
            print(f'dt = {self.time():.3},\t FPS = {self.fps():.3}')

    def time(self):
        return self.end-self.start

    def fps(self):
        return 1/self.time()
    

def style_frame(img,style_model,device=device):
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(img)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = style_model(content_image).cpu()
        
    return output[0]



def style_cam(style_model,width=320):
    print("[INFO] starting video stream...")
    vs = PiVideoStream().start()
    time.sleep(2.0)
    timer=Timer()
    while(True):
        frame=vs.read()
        if frame is None:
            frame=np.random.randint(0,255,(480,640,3),dtype=np.uint8)
        frame=cv2.flip(frame, 1)
        frame = resize(frame, width=width)

        # Style the frame
        img=style_frame(frame,style_model,device).numpy()
        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        
        img = img.transpose(1, 2, 0)
        img=cv2.resize(img[:,:,::-1],(640,480))

        # print(img.shape)
        cv2.imshow("Output", img)
        timer()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

def multi_style(path,width=320,device=device):
    model_iter=itertools.cycle(os.listdir(path))
    model_file=next(model_iter)
    print(f'Using {model_file} ')
    model_path=os.path.join(path,model_file)
    model = TransformerNet()
    model.load_state_dict(read_state_dict(model_path))
    model.to(device)
    
    vs = PiVideoStream().start()
    time.sleep(2.0)
    timer=Timer()
    while(True):
        frame=vs.read()
        if frame is None:
            frame=np.random.randint(0,255,(int(width/1.5),width,3),dtype=np.uint8)
        frame=cv2.flip(frame, 1)
        frame = resize(frame, width=width)

        # Style the frame
        img=style_frame(frame,model,device).numpy()
        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        
        img = img.transpose(1, 2, 0)
        img=cv2.resize(img[:,:,::-1],(640,480))

        # print(img.shape)
        cv2.imshow("Output", img)
        timer()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            model_file=next(model_iter)
            print(f'Using {model_file} ')
            model_path=os.path.join(path,model_file)
            model.load_state_dict(read_state_dict(model_path))
            model.to(device)
        elif key == ord("q"):
            break



def read_state_dict(path):
    state_dict=torch.load(path)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    return state_dict



if __name__=='__main__':
    # Parse input arguments
    args=parser.parse_args()
    gpu=args.gpu
    if gpu!=0 and torch.cuda.is_available():
        device=torch.device('cuda')
    print('Using {}'.format(device))
    path= pathlib.Path(args.model)
    width=np.int(args.width)
    if path.is_file():
        # load model
        model = TransformerNet()
        # state_dict = torch.load(args.model)
        # for k in list(state_dict.keys()):
        #     if re.search(r'in\d+\.running_(mean|var)$', k):
        #         del state_dict[k]
        state_dict=read_state_dict(args.model)
        model.load_state_dict(state_dict)
        model.to(device)
        style_cam(model,width)
    else:
        multi_style( path = path,
                    width = width,
                    device = device)

