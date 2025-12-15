import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from improved_model import DenoiseUNet
from dataset import DenoiseDataset
from losses import CharbonnierLoss, SSIM
from utils import save_checkpoint
import random

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoiseUNet().to(device)
    
    # Data
    train_ds = DenoiseDataset(args.train_noise, args.train_clean, augment=True)
    val_ds = DenoiseDataset(args.val_noise, args.val_clean, augment=False) if args.val_noise else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)
    
    # ===== FIX 1: 降低学习率 =====
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # ===== FIX 2: 添加学习率调度 =====
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    charbonnier = CharbonnierLoss()
    ssim_loss = SSIM()

    best_val = -1
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_char = 0.0
        running_ssim = 0.0
        
        for i, (noisy, clean, _) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # ===== FIX 3: 移除训练时的额外噪声增强 =====
            # 保持原始数据集的噪声,不再额外添加
            
            pred = model(noisy)
            
            # ===== FIX 4: 修正损失函数权重 =====
            loss_char = charbonnier(pred, clean)
            loss_ssim = 1 - ssim_loss(pred, clean)
            
            # Charbonnier主导,SSIM辅助
            loss = loss_char + 0.1 * loss_ssim
            
            optimizer.zero_grad()
            loss.backward()
            
            # ===== FIX 5: 梯度裁剪防止爆炸 =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            running_char += loss_char.item()
            running_ssim += loss_ssim.item()
            
            # 打印更详细的训练信息
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} "
                      f"(Char: {loss_char.item():.4f}, SSIM: {loss_ssim.item():.4f})")
        
        avg_loss = running_loss / len(train_loader)
        avg_char = running_char / len(train_loader)
        avg_ssim = running_ssim / len(train_loader)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_loss:.6f} (Char: {avg_char:.6f}, SSIM: {avg_ssim:.6f})")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Validation
        if val_ds:
            model.eval()
            with torch.no_grad():
                tot_ssim = 0.0
                tot_psnr = 0.0
                for noisy, clean, _ in val_loader:
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    pred = model(noisy)
                    
                    # SSIM
                    val_ssim = ssim_loss(pred, clean).item()
                    tot_ssim += val_ssim
                    
                    # PSNR
                    mse = torch.mean((pred - clean) ** 2).item()
                    if mse > 0:
                        psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
                        tot_psnr += min(psnr.item(), 30.0)
                
                mean_ssim = tot_ssim / len(val_loader)
                mean_psnr = tot_psnr / len(val_loader)
                
                print(f"Validation - SSIM: {mean_ssim:.6f}, PSNR: {mean_psnr:.4f}")
                
                # 综合评分
                score = 0.6 * mean_ssim + 0.4 * (mean_psnr / 30.0)
                print(f"Combined Score: {score:.6f}")
                
                if mean_ssim > best_val:
                    best_val = mean_ssim
                    save_checkpoint({
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'epoch': epoch,
                        'ssim': mean_ssim,
                        'psnr': mean_psnr
                    }, args.ckpt_dir, name=f'best_epoch{epoch+1}_ssim{mean_ssim:.4f}.pth')
                    print(f"✓ New best model saved!")
        
        print(f"{'='*60}\n")
        
        # 学习率调度
        scheduler.step()
        # scheduler.step(mean_ssim)  # for ReduceLROnPlateau
    
    # Final save
    save_checkpoint({
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': args.epochs
    }, args.ckpt_dir, name='final.pth')
    print("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_noise', type=str, 
                       default='extracted_files/data/train/noise')
    parser.add_argument('--train_clean', type=str, 
                       default='extracted_files/data/train/origin')
    parser.add_argument('--val_noise', type=str, default=None)
    parser.add_argument('--val_clean', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)  # 降低初始学习率
    args = parser.parse_args()
    train(args)