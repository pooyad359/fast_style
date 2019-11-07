import argparse
from time import time
import cv2
import torch
from torchvision import transforms
import utils
import re
from transformer_net import TransformerNet
import pathlib

device=torch.device('cpu')
parser=argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to style model.')
parser.add_argument('--image', required=True, help='Path to image.')
parser.add_argument('--output', default=None, help='Path to where the output should be saved.')
parser.add_argument('--scale', default=1, type=float,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
def style_frame(img,style_model):
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(img)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = style_model(content_image).cpu()
        
    return output[0]
        

if __name__=='__main__':
    # Parse input arguments
    args=parser.parse_args()
    # load image
    image = utils.load_image(args.image, scale=args.scale)

    # load model
    model = TransformerNet()
    state_dict = torch.load(args.model)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    model.load_state_dict(state_dict)
    model.to(device)

    # style the image
    output=style_frame(image,model)
    if output is None:
        output= pathlib.Path(args.image).parent.joinpath(f'output{int(time.time()*100)}.jpg')
    # save the output
    output_path=args.output
    utils.save_image(output_path, output)
