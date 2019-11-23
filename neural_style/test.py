import argparse
import time
import cv2
import torch
from torchvision import transforms
import utils
import re
from transformer_net import TransformerNet
import pathlib
from imutils.video import VideoStream
from imutils import resize
import numpy as np
import os
from style_cam import read_state_dict, style_frame


parser=argparse.ArgumentParser()
parser.add_argument('--model', default='./models/udnie.pth', help='Path to style model.')
parser.add_argument('--width', default=320, type=int,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
parser.add_argument('--image',default=None,help = 'Path to an image for testing. If empty random noise will be used.')
parser.add_argument('--trials',default=32,type=int,help='Number of times the image will be passed through the network.')
parser.add_argument('--gpu',default=0,type=int,help='If it is non-zero gpu will be used for inference.')
if __name__=='__main__':
    args=parser.parse_args()
    gpu=args.gpu
    if gpu==0:
        device=torch.device('cpu')
    else:
        device=torch.device('cuda')
    width=args.width
    if args.image is None:
        image = np.random.randint(0,255,(width,width,3),dtype=np.uint8)
    else:
        image = cv2.imread(args.image)
        image = image[:width,:width,:]
    n_try= args.trials
    model_path = args.model
    state_dict=read_state_dict(model_path)
    model=TransformerNet()
    model.load_state_dict(state_dict)
    model.to(device)
    start=time.time()
    for i in range(n_try):
        output=style_frame(image,model,device=device)
        output=output.numpy().astype(np.uint8)
        output = output.transpose(1, 2, 0)
    end=time.time()

    # cv2.imshow('output',output)
    # key = cv2.waitKey(1) & 0xFF
    # print(output.numpy().astype(np.uint8))
    # print(image)
    print(f'Size: {width} x {width}')
    print(f'Trials: {n_try}')
    print(f'average time = {(end-start)/n_try:.3},\t {n_try/(end-start):.3} FPS')