#!/usr/bin/env python3
"""
generate_brain_masks_simple.py

Create a brain mask for each image in the CSV using mask = (image > 0).

Usage:
    python generate_brain_masks_simple.py --csv /path/to/vs_dataset.csv --out_dir /path/to/brain_masks

This writes:
  - <input_csv_basename>_with_brainmask.csv  (same as input + column 'brain_mask_nifti')
  - brain mask files named <image_basename>_brainmask.nii.gz in --out_dir
"""
import os
import argparse
import pandas as pd
import nibabel as nib
import numpy as np

def make_brain_mask_from_image(img_path, out_path, overwrite=False):
    if os.path.exists(out_path) and not overwrite:
        return out_path
    img = nib.load(img_path)
    data = img.get_fdata(dtype=np.float32)
    # mask = voxels strictly greater than 0 (user requested)
    mask = (data > 0).astype(np.uint8)
    # Save with same affine + header (header will be adjusted to mask dtype)
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)
    # ensure dtype uint8
    mask_img.set_data_dtype(np.uint8)
    nib.save(mask_img, out_path)
    return out_path

def main(args):
    df = pd.read_csv(args.csv)
    if args.image_col not in df.columns:
        raise ValueError(f"CSV missing required column '{args.image_col}'")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    new_col = args.brain_col

    # prepare output column
    df_out = df.copy()
    df_out[new_col] = pd.NA

    for i, row in df.iterrows():
        img_path = row[args.image_col]
        if pd.isna(img_path) or not os.path.exists(img_path):
            print(f"[WARN] missing image path at row {i}: {img_path}")
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        # handle .nii.gz
        if base.endswith('.nii'):
            base = base[:-4]
        out_name = base + args.suffix
        out_path = os.path.join(out_dir, out_name)
        try:
            saved = make_brain_mask_from_image(img_path, out_path, overwrite=args.overwrite)
            df_out.at[i, new_col] = saved
            print(f"[OK] wrote brain mask for {img_path} -> {saved}")
        except Exception as e:
            print(f"[ERR] failed to create mask for {img_path}: {e}")

    out_csv = os.path.splitext(args.csv)[0] + "_with_brainmask.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved new CSV with brain masks: {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, default='/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg_VS.csv', help="Input CSV with at least column 'image_nifti'")
    p.add_argument("--image_col", default="image_nifti")
    p.add_argument("--out_dir", default="/home/ubuntu/Documents/Nafisha/VS_data_nifti_Genseg/brain_masks")
    p.add_argument("--brain_col", default="brain_mask_nifti")
    p.add_argument("--suffix", default="_brainmask.nii.gz")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
