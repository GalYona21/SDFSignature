'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os

import torch.cuda

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, training, loss
from model import SIREN
from torch.utils.data import DataLoader
import configargparse
from plyfile import PlyData
import numpy as np

from utils import rotate_point_cloud

# Load the PLY file

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')


# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='./data/bunny_curvs.ply',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default="./checkpoint.pth", help='Checkpoint to trained model.')
opt = p.parse_args()

torch.manual_seed(0)  # Set the random seed for CPU operations
torch.cuda.manual_seed(0)  # Set the random seed for GPU operations
np.random.seed(0)


plydata = PlyData.read(opt.point_cloud_path)
x = plydata.elements[0]['x']
y = plydata.elements[0]['y']
z = plydata.elements[0]['z']
nx = plydata.elements[0]['nx']
ny = plydata.elements[0]['ny']
nz = plydata.elements[0]['nz']
vertices = np.stack([x,y,z], axis=1)
normals = np.stack([nx,ny,nz], axis=1)
# vertices, normals = rotate_point_cloud(vertices, normals, angle_x=90, angle_y=30, angle_z=30)

sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size, coords=vertices, normals=normals)
dataloader = DataLoader(sdf_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
# if opt.model_type == 'nerf':
#     model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
# else:
#     model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
hidden_dim = 256
length_nn = 5
config_nn = [hidden_dim]*length_nn
model = SIREN(n_in_features=3, n_out_features=1, hidden_layer_config=config_nn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.load_state_dict(torch.load("/home/gal.yona/SDFSignatures/SDFSignature/logs/bunny_sdf/model_bunny_sanity", map_location=device))

if device == 'cuda':
    model.cuda()

# Define the loss
loss_fn = loss.sdf

root_path = os.path.join(opt.logging_root, "sdf_bunny_same_init2")

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, double_precision=False,
               clip_grad=True, device=device)