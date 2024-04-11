import torch
import json
from utils import rotate_point_cloud

# Load the JSON file
with open(f"./results/sanity_bunny_signature.json", "r") as file:
    loaded_bunny_signature = json.load(file)
with open(f"./results/sanity_rotated_bunny_signature.json", "r") as file:
    loaded_rotated_bunny_signature = json.load(file)


print("first sig: ")
print("H: "+str(loaded_signature_bunny[3,13780].item()))
print("H1: "+str(loaded_signature_bunny[4,13780].item()))
print("H2: "+str(loaded_signature_bunny[5,13780].item()))
print("H11: "+str(loaded_signature_bunny[6,13780].item()))
# print(loaded_signature_bunny2)
print("second sig: ")
print("H: "+str(loaded_signature_rotated_bunny[3,37687].item()))
print("H1: "+str(loaded_signature_rotated_bunny[4,37687].item()))
print("H2: "+str(loaded_signature_rotated_bunny[5,37687].item()))
print("H11: "+str(loaded_signature_rotated_bunny[6,37687].item()))

torch.norm(torch.abs(loaded_signature_bunny[:1,:,:]) - torch.abs(loaded_signature_rotated_bunny[:1,:,:]))/loaded_signature_bunny2.size(1)
torch.norm(loaded_signature_bunny[:1,:,:] - loaded_signature_rotated_bunny[:1,:,:])/loaded_signature_bunny2.size(1)