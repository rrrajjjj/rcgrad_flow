import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch

from rethinking_visual_sound_localization.data import UrbansasDataset
from rethinking_visual_sound_localization.eval_utils import compute_metrics, cal_CIOU
from rethinking_visual_sound_localization.models import CLIPTran
from rethinking_visual_sound_localization.models import RCGrad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOW_MODEL_PATH = "../rcgrad_flow/models/epoch=82-val_loss=3.4746.ckpt"
checkpoint = torch.load(FLOW_MODEL_PATH, map_location=device)["state_dict"]

def main():
    urbansas_dataset = UrbansasDataset(data_root = data_root)

    # get predictions
    if model == "rc_grad":
        if flow_channel:
            rc_grad = RCGrad(modal="flow", checkpoint = checkpoint)
        else:
            rc_grad = RCGrad()
    
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            flow_norm = None
            if flow_channel:
                try:
                    flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
                    flow_norm = (flow+5)/200
                except:
                    continue
            
            preds.append((ft, rc_grad.pred_audio(img, audio, flow_norm), gt_map))
    
    
    elif model == "flow":
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
            except:
                continue
            preds.append((ft, flow, gt_map))

    
    elif model == "rc_grad_flow":
        if flow_channel:
            rc_grad = RCGrad(modal="flow", checkpoint = checkpoint)
        else:
            rc_grad = RCGrad()

        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            flow_norm = None
            if flow_channel:
                try:
                    flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
                    flow_norm = (flow+5)/200
                except:
                    continue
            pred = rc_grad.pred_audio(img, audio, flow_norm)
            pred*=flow
            preds.append((ft, pred, gt_map))

    elif model == "clip_tran":
        clip_tran = CLIPTran()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            preds.append((ft, clip_tran.pred_audio(img, audio), gt_map))

    # save image-wise cIoU
    ious = [cal_CIOU(pred, gt_map)[0] for _, pred, gt_map in preds]
    filenames = [ft for ft, _, _ in preds]
    iou_df = pd.DataFrame()
    iou_df["filename"] = filenames
    iou_df["iou"] = ious
    if_flow = "with_flow" if flow_channel else "without_flow"
    os.makedirs(f"evaluation/{if_flow}", exist_ok=True)
    iou_df.to_csv(f"evaluation/{if_flow}/{model}.csv", index=None)

    # compute metrics
    metrics = compute_metrics(preds)
    print(metrics)  



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='evaluates ssl on urbansas')
    parser.add_argument('-filtered', '--f', action='store_true',
                        help='The filtered version of the dataset will be used if the argument is passed')
    parser.add_argument("-model", action = "store", default="rc_grad")
    parser.add_argument("-flow_channel", action="store_true")

    filtered = parser.parse_args().f
    model = parser.parse_args().model
    flow_channel = parser.parse_args().flow_channel

    dataset = "urbansas"
    if filtered:
        dataset = "urbansas_filtered"
    data_root = f"../{dataset}/"

    print(f"Using model - {model}")
    print(f"Dataset - {dataset}")
    print(f"Flow Channel: {flow_channel}")

    # setup evaluation directory
    if not os.path.isdir("evaluation/"):
        os.mkdir("evaluation/")

    main()

