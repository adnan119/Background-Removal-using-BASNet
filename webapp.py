import os
import argparse
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import numpy as np
import flask
import io
import cv2
from model import BASNet

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
# initialize our Flask application and the model


app = flask.Flask(__name__, instance_relative_config=True)

model_dir = './saved_models/basnet_bsi/basnet.pth'

net = None

def load_model():
    global net

    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

def prepare_image(image, target=256):
    image = SalObjDataset(img_name_list = image, lbl_name_list = [],transform=transforms.Compose([RescaleT(target),ToTensorLab(flag=0)]))

    # return the processed image
    return image

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

@app.route("/predict", methods=["POST"])
def predict():

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            img = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            #img = img.type(torch.FloatTensor)

            if torch.cuda.is_available():
                img = Variable(image.cuda())
            else:
                img = Variable(img)

            d1,d2,d3,d4,d5,d6,d7,d8 = net(img)
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            

            predict = pred
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            im = Image.fromarray(predict_np*255).convert('RGB')
            image = io.imread(image)
            imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

            pb_np = np.array(imo)

            cv_img = cv2.imread(image)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            result_img = cv2.imread(imo)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            new_img = np.where(result_img<[250,250,250],[254,254,254],cv_img)
            new_img = new_img.astype('uint8')
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

            byte_io = io.BytesIO()
            new_img.save(byte_io, 'PNG')
            byte_io.seek(0)

    return flask.send_file(byte_io, mimetype='image/png')



if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0')