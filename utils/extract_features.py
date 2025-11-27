import os
import json
import time
import argparse
import numpy as np
import cv2
import openslide
import torch
import torch.nn as nn
from torchvision import transforms
import timm 
from tqdm import tqdm 

# --- 1. Custom ViT Model Loading Module (Unchanged) ---

def init_custom_vit(ckpt_path, device, model_name='vit_small_patch16_224'):
    """
    Load custom trained ViT model and remove the classification head for feature extraction.
    """
    print(f"🔄 Building model architecture: {model_name} ...")
    
    try:
        model = timm.create_model(model_name, pretrained=False)
    except Exception as e:
        print(f"❌ Unable to create model {model_name}: {e}")
        return None

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found: {ckpt_path}")
        return None

    print(f"🔄 Loading checkpoint: {ckpt_path} ...")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("module.", "")
            clean_state_dict[clean_key] = v
            
        msg = model.load_state_dict(clean_state_dict, strict=False)
        # print(f"   Checkpoint loading msg: {msg}") 

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

    model.reset_classifier(0) 
    model.to(device)
    model.eval()
    return model

# --- 2. WSI Preprocessing Tools (Unchanged) ---

def get_tissue_mask(slide, level=2):
    if level >= len(slide.level_dimensions):
        level = len(slide.level_dimensions) - 1
        
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    img_np = np.array(img)
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, slide.level_downsamples[level]

# --- 3. Core Processing Pipeline (Unchanged) ---

def process_slide(wsi_path, save_path, model, device, patch_size=256, batch_size=64):
    
    # 1. Check output file
    if os.path.exists(save_path):
        tqdm.write(f"⚠️ Skipping existing file: {os.path.basename(save_path)}") 
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        tqdm.write(f"❌ Cannot open WSI: {wsi_path}")
        return

    # A. Generate Mask
    try:
        tissue_mask, downsample = get_tissue_mask(slide)
    except Exception as e:
        tqdm.write(f"❌ Mask generation failed: {wsi_path}")
        return
    
    # B. Generate Coordinates
    w_wsi, h_wsi = slide.dimensions
    patches_coords = []
    stride = patch_size 
    mask_patch_size = int(patch_size / downsample)
    
    for y in range(0, h_wsi, stride):
        for x in range(0, w_wsi, stride):
            mx = int(x / downsample)
            my = int(y / downsample)
            if my + mask_patch_size >= tissue_mask.shape[0] or mx + mask_patch_size >= tissue_mask.shape[1]:
                continue
            mask_roi = tissue_mask[my:my+mask_patch_size, mx:mx+mask_patch_size]
            if np.count_nonzero(mask_roi) / mask_roi.size > 0.3:
                patches_coords.append((x, y))
    
    if len(patches_coords) == 0:
        tqdm.write(f"❌ {os.path.basename(wsi_path)} No valid tissue detected, skipping.")
        return

    # C. Preprocessing
    eval_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # D. Extract Features
    all_features = []
    all_coords = []
    
    loader = tqdm(range(0, len(patches_coords), batch_size), 
                  desc=f"   Extracting {os.path.basename(wsi_path)[:15]}...", 
                  unit="batch", 
                  leave=False) 
    
    for i in loader:
        batch_coords = patches_coords[i : i + batch_size]
        batch_tensors = []
        
        for (x, y) in batch_coords:
            try:
                img = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                img = eval_transforms(img)
                batch_tensors.append(img)
            except Exception:
                pass
                
        if not batch_tensors:
            continue
            
        batch_stack = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            features = model(batch_stack)
            all_features.append(features.cpu())
            
        all_coords.extend(batch_coords)

    # E. Save
    if len(all_features) > 0:
        features_tensor = torch.cat(all_features, dim=0)
        coords_tensor = torch.tensor(all_coords)
        
        torch.save({
            'features': features_tensor, 
            'coords': coords_tensor
        }, save_path)
    
    slide.close()


# --- 4. Main Entry Point (Modified for argparse) ---

if __name__ == '__main__':
    # === Argument Parser Definition ===
    parser = argparse.ArgumentParser(description='WSI Feature Extraction with Custom ViT')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to JSON config file containing "input_wsi: output_pt" mapping')
    parser.add_argument('--ckpt', type=str, required=True, 
                        help='Path to model checkpoint (.pth)')
    
    # Optional arguments (with default values)
    parser.add_argument('--arch', type=str, default='vit_small_patch16_224', 
                        help='Model architecture name (default: vit_small_patch16_224)')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size (default: 64)')
    parser.add_argument('--patch_size', type=int, default=256, 
                        help='WSI patch size (default: 256, will be resized to 224)')

    args = parser.parse_args()

    # === Argument Verification and Printing ===
    print("="*40)
    print(f"Config File:  {args.config}")
    print(f"Checkpoint:   {args.ckpt}")
    print(f"Model Arch:   {args.arch}")
    print(f"GPU ID:       {args.gpu}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Patch Size:   {args.patch_size}")
    print("="*40)

    if not os.path.exists(args.config):
        print(f"❌ Error: JSON config file not found: {args.config}")
        exit()

    # Read JSON
    with open(args.config, 'r') as f:
        path_mapping = json.load(f)
    
    print(f"📋 Total files to process in task list: {len(path_mapping)}")

    # Initialize Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = init_custom_vit(args.ckpt, device, model_name=args.arch)
    
    if model is not None:
        print("\n🚀 Starting batch processing...")
        
        pbar = tqdm(path_mapping.items(), desc="Total Progress", unit="slide")
        
        for input_wsi_path, output_pt_path in pbar:
            
            pbar.set_postfix(current_file=os.path.basename(input_wsi_path)[:10])
            
            if not os.path.exists(input_wsi_path):
                tqdm.write(f"❌ Input file not found: {input_wsi_path}, skipping.")
                continue
            
            process_slide(
                wsi_path=input_wsi_path, 
                save_path=output_pt_path, 
                model=model, 
                device=device, 
                patch_size=args.patch_size, 
                batch_size=args.batch_size
            )
            
    print("\n🎉 All tasks completed!")