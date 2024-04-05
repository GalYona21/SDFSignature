'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sdf_meshing
import configargparse
from model import SIREN

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')


# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default='./logs/sdf_bunny/checkpoints/model_current.pth', help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=256)

opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        # if opt.mode == 'mlp':
        #     self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
        # elif opt.mode == 'nerf':
        #     self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        hidden_dim = 256
        length_nn = 5
        config_nn = [hidden_dim] * length_nn
        self.model = SIREN(n_in_features=3, n_out_features=1, hidden_layer_config=config_nn)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device(self.device)))
        if self.device == ' cuda':
            self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(coords)['model_out']


sdf_decoder = SDFDecoder()

root_path = os.path.join(opt.logging_root, "sdf_meshing")
if not os.path.exists(root_path):
    os.makedirs(root_path)

sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution, device=sdf_decoder.device)