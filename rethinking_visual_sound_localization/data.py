import glob
import xml.etree.ElementTree as ET
from pathlib import Path


import librosa
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import IterableDataset
from rethinking_visual_sound_localization.eval_utils import parse_annot



class FlickrSoundNetDataset(IterableDataset):
    def __init__(self, data_root):
        super(FlickrSoundNetDataset).__init__()
        self.data_root = data_root
        self.flickr_test = list(
            zip(
                *pd.read_csv(
                    "https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/main/metadata/flickr_test.csv",
                    header=None,
                ).values
            )
        )[0]
        self.files = glob.glob("{}/Data/*/*.wav".format(self.data_root))

    def __iter__(self):
        for ft in self.flickr_test:
            img = Image.open(
                [f for f in self.files if str(ft) in f][0].replace("wav", "jpg")
            ).convert("RGB")
            audio, _ = librosa.load(
                [f for f in self.files if str(ft) in f][0], sr=16000
            )
            gt = ET.parse("{}/Annotations/{}.xml".format(self.data_root, ft)).getroot()

            gt_map = np.zeros([224, 224])
            bboxs = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == "bbox":
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text) / 256))
                    bboxs.append(bbox)

            for item in bboxs:
                temp = np.zeros([224, 224])
                temp[item[1] : item[3], item[0] : item[2]] = 1
                gt_map += temp
            gt_map /= 2
            gt_map[gt_map > 1] = 1
            yield ft, img, audio, gt_map


class UrbansasDataset(IterableDataset):
    def __init__(self, data_root, modal = "vision"):
        super(UrbansasDataset).__init__()
        self.data_root = data_root
        self.files = glob.glob("{}/Data/*.wav".format(self.data_root))
        self.urbansas_test = [Path(f).stem for f in self.files]
            
    def __iter__(self):
        for ft in self.urbansas_test:
            img = Image.open(
                [f for f in self.files if str(ft) in f][0].replace("wav", "jpg")
            ).convert("RGB")
            w, h = img.size

            audio, _ = librosa.load(
                [f for f in self.files if str(ft) in f][0]
            )

            bboxs = parse_annot("{}/Annotations/{}.txt".format(self.data_root,ft))
            gt_map = np.zeros([224, 224])
            
            for item in bboxs:
                x1, y1, bbox_w, bbox_h = int(item[0]/w*224), int(item[1]/h*224), int(item[2]/w*224), int(item[3]/h*224)
                x2, y2 = x1+bbox_w, y1+bbox_h
                temp = np.zeros([224, 224])
                temp[y1:y2, x1:x2] = 1
                gt_map += temp
            #gt_map /= 2
            gt_map[gt_map > 1] = 1
            yield ft, img, audio, gt_map




