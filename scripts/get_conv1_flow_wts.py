import torch
import glob
import os
import numpy as np
from pathlib import Path
def main():
    checkpoints = glob.glob("../rcgrad_flow/models/*ckpt")
    checkpoints.sort(key=os.path.getmtime)

    for i, ckpt in enumerate(checkpoints):
        fname = Path(ckpt).stem
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
        conv_flow_wts = state_dict['image_encoder.conv1_flow.weight'].numpy()
        np.save(f"../rcgrad_flow/flow_wts/{i}-{fname}.npy", conv_flow_wts[:,3,:,:].flatten())

if __name__ == "__main__":
    os.makedirs("../rcgrad_flow/flow_wts/", exist_ok=True)
    main()