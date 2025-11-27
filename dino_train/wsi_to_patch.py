import os
import glob
import time
import argparse
import json
import openslide
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def get_tissue_mask_from_thumbnail(slide, downsample_factor=32):
    """
    Level 1 Filtering: Generate Low-Resolution Tissue Mask
    Purpose: Quickly determine where tissue is likely located to skip disk reads for large blank areas.
    """
    # 1. Get thumbnail (Utilizes OpenSlide's internal pyramid structure, very fast)
    w, h = slide.dimensions
    target_w = w // downsample_factor
    target_h = h // downsample_factor
    
    # Protection against extremely small dimensions
    target_w = max(target_w, 1)
    target_h = max(target_h, 1)
    
    thumbnail = slide.get_thumbnail((target_w, target_h))
    thumbnail = thumbnail.convert("RGB")
    thumb_arr = np.array(thumbnail)

    # 2. Color threshold filtering
    img_gray = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)
    s_channel = img_hsv[:, :, 1] # Saturation

    # Logic: Saturation > 15 AND Brightness < 240 (Exclude pure white/extremely bright backgrounds)
    mask = np.logical_and(s_channel > 15, img_gray < 240)
    mask = mask.astype(np.uint8)

    # 3. Morphological operations (Key Step)
    
    # A. Closing: Fill small holes inside the tissue to make the mask more coherent
    close_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    # B. Dilation: Expand the mask slightly outwards
    # Purpose: Resolve alignment errors between low-res and high-res, preventing tissue edges from being missed
    dilate_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    
    # Return Mask and scaling ratio
    return mask, (w / thumb_arr.shape[1], h / thumb_arr.shape[0])

def is_tissue_high_res(patch_arr, tissue_ratio_threshold=0.5):
    """
    Level 2 Filtering: Patch-level Fine Detection
    Purpose: After reading the High-Res Patch, verify if it is actually tissue (exclude background edges, stains).
    """
    try:
        # To accelerate calculation, perform 4x downsampling (256x256 -> 64x64), reducing computation by 16x
        # This is sufficient for determining "is it blank".
        small_patch = patch_arr[::4, ::4, :] 
        
        img_gray = cv2.cvtColor(small_patch, cv2.COLOR_RGB2GRAY)
        
        # === Core: Texture Detection ===
        # Background is usually smooth (low std dev), tissue usually has texture (high std dev)
        if np.std(img_gray) < 10: 
            return False

        # === Color Detection ===
        img_hsv = cv2.cvtColor(small_patch, cv2.COLOR_RGB2HSV)
        s_channel = img_hsv[:, :, 1]
        
        # Saturation > 20 AND Brightness < 235
        tissue_mask = np.logical_and(s_channel > 20, img_gray < 235)
        ratio = np.sum(tissue_mask) / (small_patch.shape[0] * small_patch.shape[1])
        
        return ratio > tissue_ratio_threshold

    except Exception as e:
        return False

def process_one_slide(args):
    """
    Logic for processing a single slide
    """
    slide_path, output_root, patch_size, step_size = args
    
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    save_dir = os.path.join(output_root, slide_name)
    
    # Simple resume capability: If the folder already has many images, skip it
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 10:
        return f"[Skipped] {slide_name}"

    try:
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.dimensions
        
        # === Step 1: Get Low-Res Mask (High-speed pre-screening) ===
        # downsample_factor=32 means operations happen at 1/1000th pixel magnitude
        tissue_mask, (scale_x, scale_y) = get_tissue_mask_from_thumbnail(slide, downsample_factor=32)
        mask_h, mask_w = tissue_mask.shape
        
    except Exception as e:
        return f"[Error Opening] {slide_name}: {e}"

    os.makedirs(save_dir, exist_ok=True)
    count = 0
    
    # === Step 2: Iterate High-Res Coordinates ===
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            # Boundary check
            if x + patch_size > w or y + patch_size > h:
                continue
            
            # --- Lookup (O(1) complexity) ---
            # Map High-Res coordinates back to Mask coordinates
            mx = int(x / scale_x)
            my = int(y / scale_y)
            
            # Mask boundary protection
            mx = min(mx, mask_w - 1)
            my = min(my, mask_h - 1)
            
            # If Mask indicates background (0), skip directly! No disk I/O!
            if not tissue_mask[my, mx]:
                continue
            
            # --- Step 3: Read from Disk (I/O) ---
            try:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = patch.convert("RGB")
                patch_arr = np.array(patch)

                # --- Step 4: Secondary Fine Check ---
                if is_tissue_high_res(patch_arr):
                    patch_name = f"{slide_name}_x{x}_y{y}.jpg"
                    # Save as JPG Quality 95
                    patch.save(os.path.join(save_dir, patch_name), format='JPEG', quality=95)
                    count += 1
            except Exception as e:
                print(f"Error reading region {slide_name} at {x},{y}: {e}")
                continue
    
    slide.close()
    return f"[Done] {slide_name}: {count} patches"

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch WSI Patch Extraction")
    parser.add_argument('--config_json', type=str, required=True, help="JSON file mapping src dirs to dst dirs")
    parser.add_argument('--patch_size', type=int, default=256, help="Patch size (default 256)")
    parser.add_argument('--overlap', type=int, default=0, help="Overlap size (default 0)")
    parser.add_argument('--workers', type=int, default=8, help="Number of CPU workers")
    args = parser.parse_args()

    # 1. Check configuration
    if not os.path.exists(args.config_json):
        print(f"Config file {args.config_json} not found.")
        return

    with open(args.config_json, 'r') as f:
        dir_mapping = json.load(f)

    # 2. Collect tasks
    tasks = []
    # Support common pathology slide formats
    exts = ['*.svs', '*.tif', '*.ndpi', '*.mrxs', '*.tiff', '*.bif']
    
    print("Scanning directories...")
    for src_dir, dst_dir in dir_mapping.items():
        if not os.path.exists(src_dir):
            print(f"Warning: Source dir not found: {src_dir}")
            continue
            
        found_slides = []
        for ext in exts:
            found_slides.extend(glob.glob(os.path.join(src_dir, ext)))
        
        step_size = args.patch_size - args.overlap
        for slide_path in found_slides:
            tasks.append((slide_path, dst_dir, args.patch_size, step_size))

    total_files = len(tasks)
    if total_files == 0:
        print("No slides found. Please check your JSON config and paths.")
        return

    print(f"Found {total_files} slides. Starting processing with {args.workers} workers...")
    
    # 3. Multiprocessing + Progress Bar
    start_time = time.time()
    
    # Use imap_unordered with tqdm for real-time progress tracking
    # Setting chunksize slightly larger (e.g., 1) can reduce IPC overhead, but 1 is sufficient for large tasks
    with Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_one_slide, tasks), total=total_files, unit="slide"):
            pass

    end_time = time.time()
    print(f"\nAll done! Total time: {(end_time - start_time)/60:.2f} mins")

if __name__ == '__main__':
    main()