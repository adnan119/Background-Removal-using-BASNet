
import os
import argparse
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet


ap = argparse.ArgumentParser()
ap.add_argument('-i','--image_directory',required=True,
    help = "insert path of the directory that contains input images to remove background from")
ap.add_argument('-p',"--predict_directory", required=True,
    help = "insert the path of the directory that will be used to save the images predicted after runnnig the model")
ap.add_argument('-s',"--save_directory", required=True,
    help = "insert the path of the directory that will be used to save the images after the background has been removed")	
args = vars(ap.parse_args())


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')



image_dir = args['image_directory']
prediction_dir = args['predict_directory']
model_dir = './saved_models/basnet_bsi/basnet.pth'

img_name_list = glob.glob(image_dir + '*.jpg')




test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)




net = BASNet(3,1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
	net.cuda()
net.eval()




for i_test, data_test in enumerate(test_salobj_dataloader):

	print("inferencing:",img_name_list[i_test].split("/")[-1])

	inputs_test = data_test['image']
	inputs_test = inputs_test.type(torch.FloatTensor)

	if torch.cuda.is_available():
		inputs_test = Variable(inputs_test.cuda())
	else:
		inputs_test = Variable(inputs_test)

	d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

	# normalization
	pred = d1[:,0,:,:]
	pred = normPRED(pred)

	# save results to test_results folder
	save_output(img_name_list[i_test],pred,prediction_dir)

	del d1,d2,d3,d4,d5,d6,d7,d8


img_name_list = sorted(img_name_list)

result_img_list = glob.glob(prediction_dir + '*.png')
result_img_list = sorted(result_img_list)

for img_num in range(len(img_name_list)):
    
    image = cv2.imread(img_name_list[img_num])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result_image = cv2.imread(result_img_list[img_num])
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    new_img = np.where(result_image<[250,250,250],[254,254,254],image)
    new_img = new_img.astype('uint8')
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(arg['save_directory'] + str(img_num) + '.jpg',new_img)