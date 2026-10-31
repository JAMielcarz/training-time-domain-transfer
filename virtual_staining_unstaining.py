"""Train a Pix2Pix model (U-Net## generator, PatchGAN discriminator) on paired tiles of the same tissue
before and after staining.

--direction stained2unstained (default): H&E -> unstained (virtual unstaining, training-time transfer)
--direction unstained2stained:           unstained -> H&E (virtual staining)

Input: CSV of registered tile pairs (slide_id, unstained, stained). Slides are split 80/20 (at slide level).
--train_csv and --val_csv (existing splits are reused).
Loss: adversarial + L1 + Pearson correlation. Checkpoints are saved every epoch.
"""
import argparse
import csv
import os
import random
import time
from collections import defaultdict
import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
torch.set_float32_matmul_precision('medium')


CSV_COLUMNS = ["slide_id", "unstained", "stained"]
DIRECTIONS = ("stained2unstained", "unstained2stained")


############################################
#  DATA AUGMENTATION & DATASET DEFINITIONS #
############################################

class BasicAugmentations:
    def __init__(self):
        self.augmentations = [
            A.Compose([]),
            A.Compose([A.VerticalFlip(p=1)]),
            A.Compose([A.HorizontalFlip(p=1)]),
            A.Compose([A.VerticalFlip(p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(90, 90), p=1)]),
            A.Compose([A.Rotate(limit=(90, 90), p=1), A.VerticalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(90, 90), p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(90, 90), p=1), A.VerticalFlip(p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(180, 180), p=1)]),
            A.Compose([A.Rotate(limit=(180, 180), p=1), A.VerticalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(180, 180), p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(180, 180), p=1), A.VerticalFlip(p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(270, 270), p=1)]),
            A.Compose([A.Rotate(limit=(270, 270), p=1), A.VerticalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(270, 270), p=1), A.HorizontalFlip(p=1)]),
            A.Compose([A.Rotate(limit=(270, 270), p=1), A.VerticalFlip(p=1), A.HorizontalFlip(p=1)]),
        ]

    def apply(self, image, aug_idx):
        augmentation = self.augmentations[aug_idx]
        transformed = augmentation(image=image)
        return transformed["image"]

class NoiseAugmentations:
    """Optional noise/compression augmentations.
    Not used in the original study and disabled in VirtualStainDataModule.setup()."""
    def __init__(self):
        self.transform = A.Compose([
            A.GaussNoise(var_limit=(1,50)),
            A.MultiplicativeNoise(multiplier=(0.95,1.05), elementwise=True),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.ImageCompression(quality_lower=70)
        ])
    def apply(self, image: np.ndarray) -> np.ndarray:
        transformed = self.transform(image=image)
        return transformed["image"]

class AdvancedAugmentations:
    """Optional blur/color/tone augmentations.
    Not used in the original study and disabled in VirtualStainDataModule.setup()."""
    def __init__(self):
        self.transform = A.ReplayCompose([
            A.UnsharpMask(blur_limit=(1,51), alpha=(0.5,1.0), p=0.2),
            A.GaussianBlur(blur_limit=(1,9), p=0.2),
            A.ColorJitter(brightness=(0.8,1.2), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.05,0.05)),
            A.RandomGamma(gamma_limit=(80,120)),
            A.RandomToneCurve(scale=0.2)
        ])
        self.replaydata = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        transformed = self.transform(image=image)
        self.replaydata = transformed['replay']
        return transformed["image"]

    def replay(self, image: np.ndarray) -> np.ndarray:
        transformed = A.ReplayCompose.replay(self.replaydata, image=image)
        return transformed["image"]

class PairedDataset(Dataset):
    def __init__(self, csv_file, basic_augmentations=None, noise_augmentations=None,
             advanced_augmentations=None, transform=None, tiles_folder=None,
             direction="stained2unstained"):
        self.direction = direction
        self.image_pairs = []
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                self.image_pairs.append((row["unstained"], row["stained"]))
        self.basic_augmentations = basic_augmentations
        self.noise_augmentations = noise_augmentations
        self.advanced_augmentations = advanced_augmentations
        self.transform = transform
        self.tiles_folder = tiles_folder

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        unstained_path, stained_path = self.image_pairs[idx]
        stained_image = np.array(Image.open(os.path.join(self.tiles_folder, stained_path)).convert("RGB"))
        unstained_image = np.array(Image.open(os.path.join(self.tiles_folder, unstained_path)).convert("RGB"))
        # source = model input, target = ground truth
        if self.direction == "stained2unstained":
            source_image, target_image = stained_image, unstained_image
        else:  # unstained2stained
            source_image, target_image = unstained_image, stained_image

        # Same random flip/rotation for both images of the pair
        if self.basic_augmentations:
            aug_idx = random.randint(0, len(self.basic_augmentations.augmentations) - 1)
            source_image = self.basic_augmentations.apply(source_image, aug_idx)
            target_image = self.basic_augmentations.apply(target_image, aug_idx)

        # Optional noise/color augmentations, applied to the model input only
        if self.noise_augmentations:
            source_image = self.noise_augmentations.apply(source_image)
        if self.advanced_augmentations:
            source_image = self.advanced_augmentations.apply(source_image)

        # Normalize to [-1, 1] and convert to tensors
        if self.transform:
            transformed = self.transform(image=source_image, target=target_image)
            source_image = transformed["image"]
            target_image = transformed["target"]

        return source_image, target_image

def load_pairs(csv_file):
    """Read a CSV with columns slide_id, unstained, stained.
    Returns {slide_id: [(slide_id, unstained, stained), ...]}."""
    slide_dict = defaultdict(list)
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_file} is missing required columns: {sorted(missing)}")
        for row in reader:
            slide_dict[row["slide_id"]].append(
                (row["slide_id"], row["unstained"], row["stained"])
            )
    return slide_dict


def split_by_slide(slide_dict, train_ratio=0.8, seed=0):
    """Randomly assign whole slides to train or val (avoiding tile-level leakage)."""
    slide_ids = sorted(slide_dict.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(slide_ids))
    n_train = int(len(perm) * train_ratio)
    train_slides = [slide_ids[i] for i in perm[:n_train]]
    val_slides = [slide_ids[i] for i in perm[n_train:]]
    return train_slides, val_slides

def create_tile_splits(slide_dict, train_slides, val_slides):
    train_tiles = [pair for slide in train_slides for pair in slide_dict[slide]]
    val_tiles = [pair for slide in val_slides for pair in slide_dict[slide]]
    return train_tiles, val_tiles

def save_to_csv(data, csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(data)

class VirtualStainDataModule(pl.LightningDataModule):
    def __init__(self, pairs_csv: str, tiles_folder: str, train_csv: str, val_csv: str,
                direction: str = "stained2unstained", train_ratio=0.8,
                batch_size=1, num_workers=4, seed: int = 1):
        super().__init__()
        self.pairs_csv = pairs_csv
        self.direction = direction
        self.tiles_folder = tiles_folder
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_pairs = []
        self.val_pairs = []
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.basic_augmentations = BasicAugmentations()
        self.noise_augmentations = NoiseAugmentations()
        self.advanced_augmentations = AdvancedAugmentations()
        self.transform = Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={"target": "image"})

    def prepare_data(self):
        if not os.path.exists(self.train_csv) or not os.path.exists(self.val_csv):
            slide_dict = load_pairs(self.pairs_csv)
            train_slides, val_slides = split_by_slide(slide_dict, train_ratio=self.train_ratio, seed=self.seed)
            self.train_pairs, self.val_pairs = create_tile_splits(slide_dict, train_slides, val_slides)
            print("Split stats:")
            print(f"Train slides: {len(train_slides)} -> {len(self.train_pairs)} pairs")
            print(f"Val slides: {len(val_slides)} -> {len(self.val_pairs)} pairs")
            save_to_csv(self.train_pairs, self.train_csv)
            save_to_csv(self.val_pairs, self.val_csv)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = PairedDataset(
                csv_file=self.train_csv,
                basic_augmentations=self.basic_augmentations,
                # NOTE: The noise and advanced augmentations below were NOT used in the original study
                # but kept optional
                #noise_augmentations=self.noise_augmentations,
                #advanced_augmentations=self.advanced_augmentations,
                transform=self.transform,
                tiles_folder=self.tiles_folder,
                direction=self.direction,
            )
            self.val_dataset = PairedDataset(
                csv_file=self.val_csv,
                transform=self.transform,
                tiles_folder=self.tiles_folder,
                direction=self.direction,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

#############################################
#                Generator                   #
#############################################

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv_block = nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.InstanceNorm2d(out_ch, affine=True),
                          nn.ReLU(inplace=True)
                          )

    def forward(self, x):

        output = self.conv_block(x)
        return output


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()

        self.down_conv = nn.Sequential(
                         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.InstanceNorm2d(out_ch, affine=True),
                         nn.ReLU(inplace=True)
                         )

    def forward(self, x):

        output = self.down_conv(x)
        return output


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()

        self.up_conv = nn.Sequential(
                       nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                       nn.InstanceNorm2d(out_ch, affine=True),
                       nn.ReLU(inplace=True),
                       )

    def forward(self, x):

        output = self.up_conv(x)
        return output


class inter_conv(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(inter_conv, self).__init__()

        self.inter_conv = nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                          nn.InstanceNorm2d(out_ch, affine=True),
                          nn.ReLU(inplace=True),
                          )

    def forward(self, x):

        output = self.inter_conv(x)
        return output


class UNetSharpSharp2D(nn.Module):
    def __init__(self, in_ch, out_ch, deep_supervision=True):
        super(UNetSharpSharp2D, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 512]

        # Step-1: Encode-0 Blocks of the U-Nets
        self.Input_to_B01234_0   = conv_block(in_ch, filters[0])

        # Step-2: Encode-1 Blocks of the U-Nets
        self.B01234_0_to_B0123_1 = conv_block(filters[0], filters[0])
        self.B01234_0_to_B4_1    = down_conv (filters[0], filters[1])

        # Step-3: Auxiliary Convolution Blocks for the Encode-1 Feature Maps
        self.B4_1_to_B0123_1     = inter_conv(filters[1], filters[0], 2)

        # Step-4: Encode-2 Blocks of the U-Nets
        self.B0123_1_to_B012_2   = conv_block(filters[0], filters[0])
        self.B0123_1_to_B3_2     = down_conv (filters[0], filters[1])
        self.B4_1_to_B4_2        = down_conv (filters[1], filters[2])

        # Step-5: Auxiliary Convolution Blocks for the Encode-2 Feature Maps
        self.B4_2_to_B3_2        = inter_conv(filters[2], filters[1], 2)
        self.B3_2_to_B012_2      = inter_conv(filters[1], filters[0], 2)

        # Step-6: Encode-3 Blocks of the U-Nets
        self.B012_2_to_B01_3     = conv_block(filters[0], filters[0])
        self.B012_2_to_B2_3      = down_conv (filters[0], filters[1])
        self.B3_2_to_B3_3        = down_conv (filters[1], filters[2])
        self.B4_2_to_B4_3        = down_conv (filters[2], filters[3])

        # Step-7: Auxiliary Convolution Blocks for the Encode-3 Feature Maps
        self.B4_3_to_B3_3        = inter_conv(filters[3], filters[2], 2)
        self.B3_3_to_B2_3        = inter_conv(filters[2], filters[1], 2)
        self.B2_3_to_B01_3       = inter_conv(filters[1], filters[0], 2)

        # Step-8: Encode-4 Blocks of the U-Nets
        self.B01_3_to_B0_4       = conv_block(filters[0], filters[0])
        self.B01_3_to_B1_4       = down_conv (filters[0], filters[1])
        self.B2_3_to_B2_4        = down_conv (filters[1], filters[2])
        self.B3_3_to_B3_4        = down_conv (filters[2], filters[3])
        self.B4_3_to_B4_4        = down_conv (filters[3], filters[4])

        # Step-9: Auxiliary Convolution Blocks for the Encode-4 Feature Maps
        self.B4_4_to_B3_4        = inter_conv(filters[4], filters[3], 2)
        self.B3_4_to_B2_4        = inter_conv(filters[3], filters[2], 2)
        self.B2_4_to_B1_4        = inter_conv(filters[2], filters[1], 2)
        self.B1_4_to_B0_4        = inter_conv(filters[1], filters[0], 2)

        # Step-10: Decode-3 Blocks of the U-Nets
        self.B0_4_to_B01_5       = conv_block(filters[0], filters[0])
        self.B1_4_to_B01_5       = up_conv   (filters[1], filters[0])
        self.B01_5_Cat_Blocks    = conv_block(2 * filters[0], filters[0])
        self.B2_4_to_B2_5        = up_conv   (filters[2], filters[1])
        self.B3_4_to_B3_5        = up_conv   (filters[3], filters[2])
        self.B4_4_to_B4_5        = up_conv   (filters[4], filters[3])

        # Step-11: Auxiliary Convolution Blocks for the Decode-3 Feature Maps
        self.B4_5_to_B3_5        = inter_conv(filters[3], filters[2], 2)
        self.B3_5_to_B2_5        = inter_conv(filters[2], filters[1], 2)
        self.B2_5_to_B01_5       = inter_conv(filters[1], filters[0], 2)

        # Step-12: Convolution Blocks for the Skip Connection Concatenations on the Decode-3 Blocks
        self.B2_5_Cat_Skips      = conv_block(2 * filters[1], filters[1])
        self.B3_5_Cat_Skips      = conv_block(2 * filters[2], filters[2])
        self.B4_5_Cat_Skips      = conv_block(2 * filters[3], filters[3])

        # Step-13: Decode-2 Blocks of the U-Nets
        self.B01_5_to_B012_6     = conv_block(filters[0], filters[0])
        self.B2_5_to_B012_6      = up_conv   (filters[1], filters[0])
        self.B012_6_Cat_Blocks   = conv_block(2 * filters[0], filters[0])
        self.B3_5_to_B3_6        = up_conv   (filters[2], filters[1])
        self.B4_5_to_B4_6        = up_conv   (filters[3], filters[2])

        # Step-14: Auxiliary Convolution Blocks for the Decode-2 Feature Maps
        self.B4_6_to_B3_6        = inter_conv(filters[2], filters[1], 2)
        self.B3_6_to_B012_6      = inter_conv(filters[1], filters[0], 2)

        # Step-15: Convolution Blocks for the Skip Connection Concatenations on the Decode-2 Blocks
        self.B3_6_Cat_Skips      = conv_block(2 * filters[1], filters[1])
        self.B4_6_Cat_Skips      = conv_block(2 * filters[2], filters[2])

        # Step-16: Decode-1 Blocks of the U-Nets
        self.B012_6_to_B0123_7   = conv_block(filters[0], filters[0])
        self.B3_6_to_B0123_7     = up_conv   (filters[1], filters[0])
        self.B0123_7_Cat_Blocks  = conv_block(2 * filters[0], filters[0])
        self.B4_6_to_B4_7        = up_conv   (filters[2], filters[1])

        # Step-17: Auxiliary Convolution Blocks for the Decode-1 Feature Maps
        self.B4_7_to_B0123_7     = inter_conv(filters[1], filters[0], 2)

        # Step-18: Convolution Blocks for the Skip Connection Concatenations on the Decode-1 Blocks
        self.B4_7_Cat_Skips      = conv_block(2 * filters[1], filters[1])

        # Step:19: Decode-0 Blocks of the U-Nets
        self.B0123_7_to_B01234_8 = conv_block(filters[0], filters[0])
        self.B4_7_to_B01234_8    = up_conv   (filters[1], filters[0])
        self.B01234_8_Cat_Blocks = conv_block(2 * filters[0], filters[0])

        # Step-20: Output Blocks
        self.final_d0 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        if deep_supervision:
            self.final_d1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            self.final_d2 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            self.final_d3 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            self.final_d4 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x):

        # Step-1: Encode-0 Blocks of the U-Nets
        x01234_0 = self.Input_to_B01234_0(x)

        # Step-2: Encode-1 Blocks of the U-Nets
        x0123_1  = self.B01234_0_to_B0123_1(x01234_0)
        x4_1     = self.B01234_0_to_B4_1   (x01234_0)

        # Step-3: Addition of the Encode-1 Feature Maps
        x0123_1  = ( x0123_1 + self.B4_1_to_B0123_1(x4_1) )

        # Step-4: Encode-2 Blocks of the U-Nets
        x012_2   = self.B0123_1_to_B012_2(x0123_1)
        x3_2     = self.B0123_1_to_B3_2  (x0123_1)
        x4_2     = self.B4_1_to_B4_2     (x4_1)

        # Step-5: Addition of the Encode-2 Feature Maps
        x3_2     = ( x3_2 + self.B4_2_to_B3_2(x4_2) )
        x012_2   = ( x012_2 + self.B3_2_to_B012_2(x3_2) )

        # Step-6: Encode-3 Blocks of the U-Nets
        x01_3    = self.B012_2_to_B01_3(x012_2)
        x2_3     = self.B012_2_to_B2_3 (x012_2)
        x3_3     = self.B3_2_to_B3_3   (x3_2)
        x4_3     = self.B4_2_to_B4_3   (x4_2)

        # Step-7: Addition of the Encode-3 Feature Maps
        x3_3     = ( x3_3  + self.B4_3_to_B3_3(x4_3) )
        x2_3     = ( x2_3  + self.B3_3_to_B2_3(x3_3) )
        x01_3    = ( x01_3  + self.B2_3_to_B01_3(x2_3) )

        # Step-8: Encode-4 Blocks of the U-Nets
        x0_4     = self.B01_3_to_B0_4(x01_3)
        x1_4     = self.B01_3_to_B1_4(x01_3)
        x2_4     = self.B2_3_to_B2_4 (x2_3)
        x3_4     = self.B3_3_to_B3_4 (x3_3)
        x4_4     = self.B4_3_to_B4_4 (x4_3)

        # Step-9: Addition of the Encode-4 Feature Maps
        x3_4     = ( x3_4 + self.B4_4_to_B3_4(x4_4) )
        x2_4     = ( x2_4 + self.B3_4_to_B2_4(x3_4) )
        x1_4     = ( x1_4 + self.B2_4_to_B1_4(x2_4) )
        x0_4     = ( x0_4 + self.B1_4_to_B0_4(x1_4) )

        # Step-10: Decode-3 Blocks of the U-Nets
        x01_5    = torch.cat((self.B0_4_to_B01_5(x0_4), self.B1_4_to_B01_5(x1_4)), dim=1)
        x01_5    = self.B01_5_Cat_Blocks(x01_5)
        x2_5     = self.B2_4_to_B2_5    (x2_4)
        x3_5     = self.B3_4_to_B3_5    (x3_4)
        x4_5     = self.B4_4_to_B4_5    (x4_4)

        # Step-11: Addition of the Decode-3 Feature Maps
        x3_5     = ( x3_5 + self.B4_5_to_B3_5(x4_5) )
        x2_5     = ( x2_5 + self.B3_5_to_B2_5(x3_5) )
        x01_5    = ( x01_5 + self.B2_5_to_B01_5(x2_5) )

        # Step-12: Concatenating Skip Connections to the Decode-3 Blocks
        x2_5     = torch.cat((x2_5, x2_3), dim=1)
        x2_5     = self.B2_5_Cat_Skips(x2_5)

        x3_5     = torch.cat((x3_5, x3_3), dim=1)
        x3_5     = self.B3_5_Cat_Skips(x3_5)

        x4_5     = torch.cat((x4_5, x4_3), dim=1)
        x4_5     = self.B4_5_Cat_Skips(x4_5)

        # Step-13: Decode-2 Blocks of the U-Nets
        x012_6   = torch.cat((self.B01_5_to_B012_6(x01_5), self.B2_5_to_B012_6(x2_5)), dim=1)
        x012_6   = self.B012_6_Cat_Blocks(x012_6)
        x3_6     = self.B3_5_to_B3_6     (x3_5)
        x4_6     = self.B4_5_to_B4_6     (x4_5)

        # Step-14: Addition of the Decode-2 Feature Maps
        x3_6     = ( x3_6 + self.B4_6_to_B3_6(x4_6) )
        x012_6   = ( x012_6 + self.B3_6_to_B012_6(x3_6) )

        # Step-15: Concatenating Skip Connections to the Decode-2 Blocks
        x3_6     = torch.cat((x3_6, x3_2), dim=1)
        x3_6     = self.B3_6_Cat_Skips(x3_6)

        x4_6     = torch.cat((x4_6, x4_2), dim=1)
        x4_6     = self.B4_6_Cat_Skips(x4_6)

        # Step-16: Decode-1 Blocks of the U-Nets
        x0123_7  = torch.cat((self.B012_6_to_B0123_7(x012_6), self.B3_6_to_B0123_7(x3_6)), dim=1)
        x0123_7  = self.B0123_7_Cat_Blocks(x0123_7)
        x4_7     = self.B4_6_to_B4_7(x4_6)

        # Step-17: Addition of the Decode-1 Feature Maps
        x0123_7  = ( x0123_7 + self.B4_7_to_B0123_7(x4_7) )

        # Step-18: Concatenating Skip Connections to the Decode-1 Blocks
        x4_7     = torch.cat((x4_7, x4_1), dim=1)
        x4_7     = self.B4_7_Cat_Skips(x4_7)

        # Step-19: Decode-0 Blocks of the U-Nets
        x01234_8 = torch.cat((self.B0123_7_to_B01234_8(x0123_7), self.B4_7_to_B01234_8(x4_7)), dim=1)
        x01234_8 = self.B01234_8_Cat_Blocks(x01234_8)

        # Step-20: Output Blocks
        d0 = self.final_d0(x01234_8)
        if self.deep_supervision:
            d1 = self.final_d1(x0123_7)
            d2 = self.final_d2(x012_6)
            d3 = self.final_d3(x01_5)
            d4 = self.final_d4(x0_4)

            ds_outputs = [d4, d3, d2, d1, d0]
            #output = torch.stack(ds_outputs).mean(dim=0)
            ds_weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3])
            output = sum(w * o for w, o in zip(ds_weights, ds_outputs))

        else:
            output = d0

        #output = nn.Softmax(dim=1)(output)
        #output = nn.Sigmoid()(output)
        #############
        output = torch.tanh(output)
        return output


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, norm_type='instancenorm'):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels*2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True) if norm_type.lower() == 'instancenorm' else nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True) if norm_type.lower() == 'instancenorm' else nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(512, affine=True) if norm_type.lower() == 'instancenorm' else nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, source, other):
        x = torch.cat([source, other], dim=1)
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        x = self.last(x)
        return x

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def pcc_loss(generated, target):
    """1 - mean Pearson correlation between generated and target images."""
    # Flatten the images: shape (B, C*H*W)
    generated_flat = generated.view(generated.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Compute mean and centered vectors
    gen_mean = torch.mean(generated_flat, dim=1, keepdim=True)
    target_mean = torch.mean(target_flat, dim=1, keepdim=True)
    generated_centered = generated_flat - gen_mean
    target_centered = target_flat - target_mean

    # Compute covariance and standard deviations
    covariance = torch.sum(generated_centered * target_centered, dim=1)
    generated_std = torch.sqrt(torch.sum(generated_centered ** 2, dim=1) + 1e-8)
    target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=1) + 1e-8)

    pcc = covariance / (generated_std * target_std)
    # Loss: 1 - average PCC over batch (want PCC close to 1)
    return 1 - torch.mean(pcc)

#############################################
#       LIGHTNING MODULE: PIX2PIX           #
#############################################

class Pix2Pix(pl.LightningModule):
    def __init__(self, input_channels, out_channels, output_dir, norm_type="instancenorm",
                direction="stained2unstained", lr=0.0002, lambda_l1=100, lambda_pcc=10):
        super().__init__()
        self.save_hyperparameters()
        self.generator = UNetSharpSharp2D(input_channels, out_channels, deep_supervision=True).to(self.device)
        self.discriminator = PatchGANDiscriminator(input_channels, norm_type=norm_type)
        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.output_dir = output_dir
        self.automatic_optimization = False
        self.ssim_metric = StructuralSimilarityIndexMeasure()
        self.lambda_l1 = lambda_l1
        self.lambda_pcc = lambda_pcc

    def forward(self, x):
        return self.generator(x)

    def discriminator_accuracy(self, real_output, fake_output):
        """Overall discriminator accuracy: real pairs correct when output > 0, fake pairs when < 0."""
        real_pred = (real_output > 0).float()
        fake_pred = (fake_output < 0).float()
        return (real_pred.sum() + fake_pred.sum()) / (real_pred.numel() + fake_pred.numel())

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        opt_G = optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(b1, b2))
        opt_D = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(b1, b2))
        return [opt_G, opt_D]

    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()
        source, target = batch

        # -- Discriminator training --
        # Discriminator output for the real pair: (source, target)
        discriminator_real_output = self.discriminator(source, target)
        valid = torch.ones_like(discriminator_real_output)
        real_loss = self.criterion_GAN(discriminator_real_output, valid)

        # Generate fake image from the source input.
        generated_image = self.generator(source)
        # Get discriminator output for fake pair: (source, generated_image)
        discriminator_fake_output = self.discriminator(source, generated_image.detach())
        fake = torch.zeros_like(discriminator_fake_output)
        fake_loss = self.criterion_GAN(discriminator_fake_output, fake)


        total_acc = self.discriminator_accuracy(discriminator_real_output, discriminator_fake_output)

        d_loss = 0.5 * (real_loss + fake_loss)
        opt_D.zero_grad()
        self.manual_backward(d_loss)
        opt_D.step()

        # -- Generator training --
        # Recompute discriminator output on fake pair for generator update.
        discriminator_fake_output_for_gen = self.discriminator(source, generated_image)
        gan_loss = self.criterion_GAN(discriminator_fake_output_for_gen, valid)

        l1_loss = self.lambda_l1 * self.criterion_L1(generated_image, target)
        pcc_loss_value = self.lambda_pcc * pcc_loss(generated_image, target)

        g_loss = gan_loss + l1_loss + pcc_loss_value
        opt_G.zero_grad()
        self.manual_backward(g_loss)
        opt_G.step()

        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('real_loss', real_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('fake_loss', fake_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('l1_loss', l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('pcc_loss_value', pcc_loss_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('d_total_acc', total_acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        # Save an image each epoch for visual inspection
        if self.global_rank == 0 and batch_idx == 1:
            train_output_dir = os.path.join(self.output_dir, "train_outputs")
            os.makedirs(train_output_dir, exist_ok=True)
            source_denorm = (source + 1) / 2.0
            generated_denorm = (generated_image + 1) / 2.0
            target_denorm = (target + 1) / 2.0
            save_image(source_denorm, os.path.join(train_output_dir, f"source_epoch_{self.current_epoch}_{batch_idx}.png"))
            save_image(generated_denorm, os.path.join(train_output_dir, f"generated_epoch_{self.current_epoch}_{batch_idx}.png"))
            save_image(target_denorm, os.path.join(train_output_dir, f"target_epoch_{self.current_epoch}_{batch_idx}.png"))

    def validation_step(self, batch, batch_idx):
        source, target = batch
        generated_image = self(source)

        val_l1_loss = self.criterion_L1(generated_image, target)
        val_pcc_loss = pcc_loss(generated_image, target)
        val_ssim = self.ssim_metric(generated_image, target)

        # Discriminator losses
        disc_real_output = self.discriminator(source, target)
        valid = torch.ones_like(disc_real_output)
        val_real_loss = self.criterion_GAN(disc_real_output, valid)

        disc_fake_output = self.discriminator(source, generated_image.detach())
        fake = torch.zeros_like(disc_fake_output)
        val_fake_loss = self.criterion_GAN(disc_fake_output, fake)

        val_d_loss = 0.5 * (val_real_loss + val_fake_loss)

        # Generator loss
        disc_fake_for_gen = self.discriminator(source, generated_image)
        val_gan_loss = self.criterion_GAN(disc_fake_for_gen, valid)
        val_g_loss = val_gan_loss + self.lambda_l1 * val_l1_loss + self.lambda_pcc * val_pcc_loss


        self.log("val_g_loss", val_g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_d_loss", val_d_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_l1_loss", val_l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_pcc_loss", val_pcc_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_ssim", val_ssim, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_real_loss", val_real_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_fake_loss", val_fake_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        if self.global_rank == 0:
            val_output_dir = os.path.join(self.output_dir, "val_outputs")
            os.makedirs(val_output_dir, exist_ok=True)

            if batch_idx == 1:
                # Only save source and target images in the first epoch
                if self.current_epoch == 0:
                    source_denorm = (source + 1) / 2.0
                    target_denorm = (target + 1) / 2.0
                    save_image(source_denorm, os.path.join(val_output_dir, f"source_epoch_{self.current_epoch}_{batch_idx}.png"))
                    save_image(target_denorm, os.path.join(val_output_dir, f"target_epoch_{self.current_epoch}_{batch_idx}.png"))

                # Save the generated image every epoch
                generated_denorm = (generated_image + 1) / 2.0
                save_image(generated_denorm, os.path.join(val_output_dir, f"generated_epoch_{self.current_epoch}_{batch_idx}.png"))

#############################################
#             ARGUMENT PARSING            #
#############################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Virtual staining/unstaining with Pix2Pix")

    # Required: data and output locations
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run (used for W&B and checkpoint file names)")
    parser.add_argument("--pairs_csv", type=str, required=True, help="CSV with columns slide_id,unstained,stained")
    parser.add_argument("--train_csv", type=str, required=True, help="Train split CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Validation split CSV")
    parser.add_argument("--tiles_folder", type=str, required=True,  help="Root folder; paths in the CSV are relative to it")
    parser.add_argument("--model_save_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for example images during training")

    # Optional: defaults reproduce the original setup
    parser.add_argument("--direction", choices=DIRECTIONS, default="stained2unstained",
                    help="Translation direction (input -> output): "
                         "stained2unstained = virtual unstaining (default), "
                         "unstained2stained = virtual staining")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lambda_l1", type=float, default=100)
    parser.add_argument("--lambda_pcc", type=float, default=10)
    parser.add_argument("--wandb_project", type=str, default=None)
    return parser.parse_args()

#############################################
#                MAIN                     #
#############################################

if __name__ == "__main__":
    args = parse_arguments()
    pl.seed_everything(args.seed, workers=True)

    datamodule = VirtualStainDataModule(
        pairs_csv=args.pairs_csv,
        tiles_folder=args.tiles_folder,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        direction=args.direction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    logger = (WandbLogger(project=args.wandb_project, name=args.run_name)
              if args.wandb_project else False)

    model = Pix2Pix(input_channels=3, out_channels=3, output_dir=args.output_dir,
                    norm_type="instancenorm", direction=args.direction,
                    lr=args.lr, lambda_l1=args.lambda_l1, lambda_pcc=args.lambda_pcc)

    checkpoint_callback = ModelCheckpoint(
        filename=f"{args.run_name}-{{epoch}}", dirpath=args.model_save_dir,
        save_weights_only=False, verbose=True, save_top_k=-1, every_n_epochs=1)


    trainer = Trainer(
        num_sanity_val_steps=0,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        num_nodes=1,
        max_epochs=args.epochs,
        logger=logger,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )

    start = time.time()
    trainer.fit(model, datamodule)
    elapsed = time.time() - start
    print(f"Total training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
