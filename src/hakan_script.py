from typing import List, Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from matplotlib import pyplot as plt

from models.vsc import VariationalSparseCoding
from models.conv_vsc import ConvolutionalVariationalSparseCoding

dataset = 'fashion'
width = 28
height = 28
channels = 1
hidden_size = '400'
latent_size = 8
lr = 0.01
alpha = 0.5
device = 'cpu'
log_interval = 4
normalize = False
flatten=False
kernels =  '32'

convvsc = VariationalSparseCoding(dataset, width, height, channels, 
                                  hidden_size, latent_size, lr, 
                                  alpha, device, log_interval,
                                  normalize)

print(convvsc.latent_sz)
print(convvsc.hidden_sz)
print(convvsc.model)

#convvsc = ConvolutionalVariationalSparseCoding(dataset, width, height, channels, kernels,
#                                  hidden_size, latent_size, lr, 
#                                  alpha, device, log_interval,
#                                 normalize, flatten)
model_path = '/home/raphaela/Documents/UU/classes/EODL/project_01/Variational-Sparse-Coding/results/checkpoints/VSC_fashion_1_11_8_0-001_11.pth'
convvsc.model.load_state_dict(torch.load(model_path, map_location='cpu'))

values = np.arange(-2.5,3,0.5)
for i in range(latent_size):
    z = torch.from_numpy(np.asarray([0 for i in range(latent_size)])).view(1,-1)
    for j in range(len(values)):
        z[0][i] = values[j]
        z = z.type(torch.FloatTensor)
        if j == 0:
            out = convvsc.model.decode( z ).view(1,1,28,28)
        else:
            out=torch.cat((out,convvsc.model.decode( z ).view(1,1,28,28)), dim=0)

    if i == 0:
        img = torchvision.utils.make_grid(out, nrow=11).unsqueeze(0)
    else:
        img = torch.cat((img, torchvision.utils.make_grid(out, nrow=11).unsqueeze(0)), dim=0)
    print(img.shape)
img_out =  torchvision.utils.make_grid(img, nrow=1)
plt.imshow(img_out.detach().numpy().transpose(1,2,0))
plt.show()
plt.save('test.png')
#img_out = cv2.cvtColor((img_out.detach().numpy().transpose(1,2,0)*255).astype('uint8'), cv2.COLOR_RGB2BGR)
#cv2.imwrite('test.png', img_out)
