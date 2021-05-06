
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import os
import torch
from skimage import io
from skimage.measure.simple_metrics import compare_psnr, compare_mse
import sys  
from utils import *
import time 

import torch.cuda

# Paths for data, pretrained models, and precomputed performance measures
pretrained_base = './pretrained/'
precomputed_base = './precomputed/'
data_base = 'data/'

# Datasets available in the data folder
train_folder_path = os.path.join(data_base, 'Train400/')
test_folder_path = os.path.join(data_base, 'Test/Set68/')
set12_path = os.path.join(data_base, 'Test/Set12/')
kodak_path = os.path.join(data_base, 'Test/Kodak23/') 

# Choose a model (pre-trained options: 'dncnn', 'unet', 'rcnn', 'sdensenet')
modelname = 'dncnn' 

# Select the range of noise levels (stdev, relative to intensities in range [0,255]) 
# used during training (options are 0-10, 0-30, 0-55, 0-100).
l = 0   # lower bound of training range 
h = 100 # upper bound of training range

# BF_CNN = load_model(os.path.join(pretrained_base, model, 'bias_free', str(l)+'-'+str(h)+'.pt'))

def calc_hessian_mpf( im,gpui,ji,ri):
    '''@im: a noisy image - 2-dimensional
    '''
    
    with torch.cuda.device(int(gpui)):
        model = load_model(os.path.join(pretrained_base, modelname, 'bias_free', str(l)+'-'+str(h)+'.pt'))

        os.getpid()
        print('init')
    #     inp = im

    #     model.to(gpui)
        inp=torch.tensor(im.astype('float32'),requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()


    #     inp_test = torch.tensor(noisy_im.astype('float32'),requires_grad=True).unsqueeze(1).cuda()
    #     input_imgs.append(inp_test)
    #     residual= BF_CNN(inp_test)


        ############## prepare the static model
        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        ############## find Jacobian
        out = model(inp)
        jacob = []
        hessian=[]
        for i in [ri]:#[inp.size()[2]//2]:#range(inp.size()[2]):
            for j in [ji]:#[inp.size()[3]//2]:#range(inp.size()[3]):
                part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True, create_graph=True)
                jacob.append( part_der[0][0,0].data.view(-1).cpu().numpy())
    #             print(part_der)

    #             print(np.shape(out))
    #             print(np.shape(part_der))

    #             part_der_var = torch.tensor(part_der[0][0,0,40,40],requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
    #             part_der_der = torch.autograd.grad(, inp, retain_graph=True)

        print('jacobian done')
        for ii in range(inp.size()[2]):#[20:22]:
    #             print(ii)
            for jj in range(inp.size()[3]):#[20:22]:
                part_der_der = torch.autograd.grad(part_der[0][0,0,ii,jj], inp, retain_graph=True)
    #             print(np.shape(part_der_der[0]))

                hessian.append(part_der_der[0][0,0].data.view(-1).cpu().numpy())

        part_der_der = torch.autograd.grad(part_der[0][0,0,ii,jj], inp, retain_graph=False)
        del part_der_der
        del part_der
        del model
        del out
        del inp

        print('hessian done')
    #     return [torch.stack(jacob), torch.stack(hessian)] 
    return [jacob, hessian]  

def calc_hessian_im_mpf(inp, part_der, ji):
    hessian=[]
#     inp=torch.tensor(im.astype('float32'),requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
    for ii in [20]:#range(inp.size()[2]):#[20:22]:
#             print(ii)
        for jj in [ji]:#[20:22]:
            part_der_der = torch.autograd.grad(part_der[ii,jj], inp.cuda(), retain_graph=True)
#             print(np.shape(part_der_der[0]))

            hessian.append(part_der_der[0][0,0].data.view(-1))
        
    print('hessian done')
    return [torch.stack(hessian)]

# def calc_hessian_mpf( im, inp=h):#,model):
#     '''@im: a noisy image - 2-dimensional
#     '''

#     os.getpid()
#     print('init')
# #    inp = im
    
# #    inp=torch.tensor(im.astype('float32'),requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
    
    
# #     inp_test = torch.tensor(noisy_im.astype('float32'),requires_grad=True).unsqueeze(1).cuda()
# #     input_imgs.append(inp_test)
# #     residual= BF_CNN(inp_test)
    
    
# #     ############## prepare the static model
# #     for param in model.parameters():
# #         param.requires_grad = False

# #     model.eval()

# #     ############## find Jacobian
# #     out = model(inp)
#     out = im
#     jacob = []
#     hessian=[]
#     for i in [20]:#[inp.size()[2]//2]:#range(inp.size()[2]):
#         for j in [20]:#[inp.size()[3]//2]:#range(inp.size()[3]):
#             part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True, create_graph=True)
#             jacob.append( part_der[0][0,0].data.view(-1))
# #             print(part_der)

# #             print(np.shape(out))
# #             print(np.shape(part_der))
            
# #             part_der_var = torch.tensor(part_der[0][0,0,40,40],requires_grad=True).unsqueeze(0).unsqueeze(0).cuda()
# #             part_der_der = torch.autograd.grad(, inp, retain_graph=True)

#     print('jacobian done')
#     for ii in range(inp.size()[2]):#[20:22]:
# #             print(ii)
#         for jj in range(inp.size()[3]):#[20:22]:
#             part_der_der = torch.autograd.grad(part_der[0][0,0,ii,jj], inp, retain_graph=True)
# #             print(np.shape(part_der_der[0]))

#             hessian.append(part_der_der[0][0,0].data.view(-1))
        
#     print('hessian done')
#     return [torch.stack(jacob), torch.stack(hessian)]  