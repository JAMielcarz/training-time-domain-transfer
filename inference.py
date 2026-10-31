"""Generate synthetic tiles with a trained virtual unstaining or virtual staining model.

--direction stained2unstained (default): H&E tiles -> synthetic unstained tiles
--direction unstained2stained:           unstained tiles -> synthetic H&E tiles

Per tile: 1024x1024 input -> (optional HED stain augmentation) -> generator
-> 2x2 grid of 512x512 crops -> each resized to 256x256 -> JPEG.
Input and output: one TFRecord per WSI, <dir>/<wsi_id>/<wsi_id>.tfrecord.
"""
import argparse
import io
import os
import dareblopy as db
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from stainlib.augmentation.augmenter import HedColorAugmenter

DIRECTIONS = ("stained2unstained", "unstained2stained")
OUTPUT_IMAGE_KEY = {"stained2unstained": "unstained_image", "unstained2stained": "stained_image"}

CROP_SIZE = 512    # model output is split into a 2x2 grid of 512x512 crops
OUTPUT_SIZE = 256  # each crop is resized to 256x256
GRID = 2


#############################################
#   Generator (U-Net##, Firat Korkmaz, MIT)  #
#   https://github.com/kaijiu1013/UNetSharpSharp/blob/c6c1304e1eb331c2076fde7af8db436d1275ab2d/model/UNetSharpSharp2D.py
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
            ds_weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3])
            output = sum(w * o for w, o in zip(ds_weights, ds_outputs))

        else:
            output = d0

        return output


def load_generator(checkpoint_path, device, direction):
    """Load the generator from a checkpoint of virtual_staining_unstaining.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    ckpt_direction = ckpt.get("hyper_parameters", {}).get("direction")
    if ckpt_direction is not None and ckpt_direction != direction:
        print(f"WARNING: checkpoint was trained with direction={ckpt_direction}, but --direction is {direction}")

    prefix = "generator."
    state = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    generator = UNetSharpSharp2D(3, 3, deep_supervision=True)
    generator.load_state_dict(state, strict=True)
    return generator.to(device).eval()


#############################################
#                 TFRecord I/O               #
#############################################

def read_tiles(tfrecord_path):
    """Yield (tile_id, encoded image bytes) for every record."""
    parser = db.RecordParser({
        "tile_id": db.FixedLenFeature([], db.string),
        "image": db.FixedLenFeature([], db.string),
    })
    for record in db.RecordReader(tfrecord_path):
        tile_id, image_bytes = parser.parse_single_example(record)
        yield tile_id[0].decode("utf-8"), image_bytes[0]


def crop_names(tile_id):
    """Names of the 2x2 crops of a tile named <wsi_id>_<y1>_<y2>_<x1>_<x2><ext>,
    using the crop coordinates in the WSI. Returns [(grid x, grid y, name)]."""
    base, ext = os.path.splitext(tile_id)
    parts = base.split("_")
    if len(parts) >= 5 and all(p.isdigit() for p in parts[-4:]):
        wsi_id, y1, x1 = "_".join(parts[:-4]), int(parts[-4]), int(parts[-2])
    else:
        wsi_id, y1, x1 = base, 0, 0

    names = []
    for gy in range(GRID):
        for gx in range(GRID):
            cx1, cy1 = x1 + gx * CROP_SIZE, y1 + gy * CROP_SIZE
            names.append((gx, gy, f"{wsi_id}_{cy1}_{cy1 + CROP_SIZE - 1}_{cx1}_{cx1 + CROP_SIZE - 1}{ext}"))
    return names


#############################################
#                 Inference                  #
#############################################

def to_tensor(image):
    """RGB PIL image -> 1x3xHxW tensor in [-1, 1]."""
    return (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)


def process_wsi(input_tfrecord, output_tfrecord, model, device, direction,
                batch_size, reference_tiles=None, stain_augmenter=None):
    """Translate all tiles of one WSI and write the 256x256 crops to output_tfrecord.
    Returns (n_input, n_written). Raises, leaving no output file, if anything fails or
    not all expected crops were written."""
    tmp_path = output_tfrecord + ".tmp"
    n_input, n_written = 0, 0

    def write_batch(tile_ids, tensors):
        nonlocal n_written
        with torch.no_grad():
            preds = torch.tanh(model(torch.cat(tensors).to(device)))
        for tile_id, pred in zip(tile_ids, preds):
            # model output [-1, 1] -> [0, 1]
            pred_img = TF.to_pil_image(((pred.cpu() + 1.0) / 2.0).clamp(0, 1))
            for gx, gy, name in crop_names(tile_id):
                if reference_tiles is not None and name not in reference_tiles:
                    continue
                crop = pred_img.crop((gx * CROP_SIZE, gy * CROP_SIZE, (gx + 1) * CROP_SIZE, (gy + 1) * CROP_SIZE))
                crop = crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS, reducing_gap=3.0)
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=90)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "tile_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode("utf-8")])),
                    OUTPUT_IMAGE_KEY[direction]: tf.train.Feature(bytes_list=tf.train.BytesList(value=[buf.getvalue()])),
                }))
                writer.write(example.SerializeToString())
                n_written += 1

    try:
        with tf.io.TFRecordWriter(tmp_path) as writer:
            tile_ids, tensors = [], []
            for tile_id, image_bytes in read_tiles(input_tfrecord):
                n_input += 1
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if stain_augmenter is not None:
                    image = Image.fromarray(np.clip(stain_augmenter.transform(np.array(image)), 0, 255).astype(np.uint8))
                tile_ids.append(tile_id)
                tensors.append(to_tensor(image))
                if len(tensors) >= batch_size:
                    write_batch(tile_ids, tensors)
                    tile_ids, tensors = [], []
            if tensors:
                write_batch(tile_ids, tensors)

        expected = len(reference_tiles) if reference_tiles is not None else n_input * GRID * GRID
        if n_written != expected:
            raise RuntimeError(f"{n_written}/{expected} crops written")
        os.rename(tmp_path, output_tfrecord)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return n_input, n_written


def make_stain_augmenter():
    """HED colour augmentation of haematoxylin and eosin."""
    return HedColorAugmenter(haematoxylin_sigma_range=(-0.9, 0.9),
                             haematoxylin_bias_range=(0, 0),
                             eosin_sigma_range=(-0.9, 0.9),
                             eosin_bias_range=(0, 0),
                             dab_sigma_range=(0, 0),
                             dab_bias_range=(0, 0),
                             cutoff_range=(0, 1))


#############################################
#                    Main                    #
#############################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic unstained or H&E tiles with a trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint from virtual_staining_unstaining.py trained with the same --direction")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with <wsi_id>/<wsi_id>.tfrecord")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder, same layout as --input_dir")
    parser.add_argument("--direction", choices=DIRECTIONS, default="stained2unstained",
                        help="stained2unstained = H&E -> synthetic unstained (default), "
                            "unstained2stained = unstained -> synthetic H&E")
    parser.add_argument("--reference_dir", type=str, default=None,
                        help="Folder with the same layout as --input_dir; keeps the tile set identical to its "
                            "TFRecords (only crops whose tile_id is in the reference are written, "
                            "WSIs without a reference are skipped)")
    parser.add_argument("--stain_augmentation", action="store_true",
                        help="HED stain augmentation of 50%% of WSIs, one random augmentation per WSI "
                            "(only with --direction stained2unstained)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for --stain_augmentation")
    parser.add_argument("--chunk_start", type=int, default=0, help="Index of the first WSI folder to process")
    parser.add_argument("--chunk_end", type=int, default=None, help="Index after the last WSI folder to process")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.stain_augmentation and args.direction != "stained2unstained":
        raise SystemExit("--stain_augmentation requires --direction stained2unstained (H&E input)")
    if args.seed is not None:
        np.random.seed(args.seed)

    device = torch.device(args.device)
    model = load_generator(args.checkpoint, device, args.direction)

    wsi_ids = sorted(f for f in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, f)))
    wsi_ids = wsi_ids[args.chunk_start:args.chunk_end]
    print(f"Processing {len(wsi_ids)} WSIs ({args.direction})")

    failed = []
    n_done = 0
    for wsi_id in wsi_ids:
        input_tfrecord = os.path.join(args.input_dir, wsi_id, f"{wsi_id}.tfrecord")
        output_tfrecord = os.path.join(args.output_dir, wsi_id, f"{wsi_id}.tfrecord")
        if not os.path.exists(input_tfrecord):
            print(f"{wsi_id}: input TFRecord missing, skipping")
            continue
        if os.path.exists(output_tfrecord):
            print(f"{wsi_id}: output exists, skipping")
            continue

        try:
            reference_tiles = None
            if args.reference_dir:
                reference_tfrecord = os.path.join(args.reference_dir, wsi_id, f"{wsi_id}.tfrecord")
                if not os.path.exists(reference_tfrecord):
                    print(f"{wsi_id}: reference TFRecord missing, skipping")
                    continue
                reference_tiles = {tile_id for tile_id, _ in read_tiles(reference_tfrecord)}

            # Stain augmentation: decided per WSI; the same random augmentation for all its tiles
            stain_augmenter = None
            if args.stain_augmentation:
                augmenter = make_stain_augmenter()
                if np.random.random() < 0.5:
                    augmenter.randomize()
                    stain_augmenter = augmenter

            os.makedirs(os.path.dirname(output_tfrecord), exist_ok=True)
            n_input, n_written = process_wsi(
                input_tfrecord, output_tfrecord, model, device, args.direction,
                args.batch_size, reference_tiles, stain_augmenter)
        except Exception as e:
            print(f"{wsi_id}: FAILED ({e})")
            failed.append((wsi_id, str(e)))
            continue
        print(f"{wsi_id}: {n_input} tiles -> {n_written} crops")
        n_done += 1

    print(f"\nWritten: {n_done} WSIs, failed: {len(failed)}")
    for wsi_id, error in failed:
        print(f"  {wsi_id}: {error}")


if __name__ == "__main__":
    main()
