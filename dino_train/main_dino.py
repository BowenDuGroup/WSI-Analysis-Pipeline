import os
import argparse
import math
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model_dino import get_backbone, DINOHead, MultiCropWrapper
from data_dino import DINOAugmentation
from loss_dino import DINOLoss

def parse_args():
    parser = argparse.ArgumentParser('DINO Training Script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory (contains class folders)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay')
    parser.add_argument('--momentum_teacher', type=float, default=0.996, help='Base EMA parameter for teacher update')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=5, help='Save checkpoint every X epochs')
    return parser.parse_args()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Creates a cosine learning rate/weight decay schedule with warm-up.
    Returns a tensor of size (epochs * niter_per_ep).
    """
    # 1. Linear Warmup
    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    
    # 2. Cosine Decay
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    
    # 3. Concatenate
    schedule = torch.cat((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train(args):
    # === 1. Setup Device & Directories ===
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Running on device: {device}")

    # === 2. Data Loading ===
    print("Loading data...")
    # DINOAugmentation handles the multi-crop generation (2 global + N local crops)
    transform = DINOAugmentation()

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: {len(dataset)} images.")

    # === 3. Model Initialization (Student & Teacher) ===
    # We use the exact same architecture for both networks
    student_backbone = get_backbone('vit_small_patch16_224')
    teacher_backbone = get_backbone('vit_small_patch16_224')
    
    embed_dim = student_backbone.embed_dim 
    
    # Wrap backbone and DINO head together
    student = MultiCropWrapper(
        student_backbone,
        DINOHead(embed_dim, 65536, norm_last_layer=True)
    )
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(embed_dim, 65536, norm_last_layer=True)
    )

    student, teacher = student.to(device), teacher.to(device)

    # === 4. Teacher Configuration ===
    # Teacher starts with the same weights as the Student
    teacher.load_state_dict(student.state_dict())
    
    # Teacher is NOT updated via backprop, so we disable gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # === 5. Loss & Optimizer ===
    dino_loss = DINOLoss(65536).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # === 6. Schedulers ===
    # Calculate schedules for every iteration
    lr_schedule = cosine_scheduler(
        args.lr, 1e-6, args.epochs, len(data_loader), warmup_epochs=10
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay, 0.4, args.epochs, len(data_loader)
    )
    # Momentum usually starts high (0.996) and goes to 1.0
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher, 1.0, args.epochs, len(data_loader)
    )

    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        
        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
            
            for i, (images, _) in enumerate(tepoch):
                # Current global iteration step
                it = len(data_loader) * epoch + i 
                
                # Update learning rate & weight decay for this step
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_schedule[it]
                    if param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule[it]

                # Move list of images (crops) to GPU
                images = [im.to(device, non_blocking=True) for im in images]

                # --- Student Forward ---
                # Student processes ALL crops (Global + Local)
                student_output = student(images)
                
                # --- Teacher Forward ---
                # Teacher ONLY processes Global views (the first two images in the list)
                with torch.no_grad():
                    teacher_output = teacher(images[:2])

                # --- Calculate Loss ---
                # Ensure the chunking logic in loss.py matches the number of crops
                loss = dino_loss(student_output, teacher_output)

                # --- Backprop ---
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping (Prevents gradient explosion, common in ViTs)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                optimizer.step()

                # --- Update Teacher (EMA) ---
                # Exponential Moving Average update of Teacher weights based on Student weights
                with torch.no_grad():
                    m = momentum_schedule[it] 
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        # Formula: param_k = m * param_k + (1-m) * param_q
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # =========================
        # === Save Logic (Key Modification) ===
        # =========================
        
        # 1. Always save the latest Teacher Backbone 
        # (This is the primary high-quality model you need for downstream tasks)
        teacher_state = teacher.backbone.state_dict()
        torch.save(teacher_state, os.path.join(args.output_dir, "checkpoint_teacher_latest.pth"))
        
        # 2. Periodic Checkpointing (Default: every 5 Epochs)
        if (epoch + 1) % args.save_freq == 0:
            # A. Save clean Teacher Backbone (Convenient for direct loading/testing later)
            torch.save(teacher_state, os.path.join(args.output_dir, f"checkpoint_teacher_{epoch+1}.pth"))
            
            # B. Save full training state (Convenient for resuming training if crashed)
            torch.save({
                'epoch': epoch + 1,
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dino_loss': dino_loss.state_dict(),
            }, os.path.join(args.output_dir, f"checkpoint_full_{epoch+1}.pth"))
            
            print(f"--> Saved checkpoint at epoch {epoch+1}")

    print(f"Training finished. Model saved to {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    train(args)