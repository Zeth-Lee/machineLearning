import torch
import os
import numpy as np
from PIL import Image
import argparse

# ===== 自动检测并加载正确的模型 =====
def load_model_auto(ckpt_path, device='cuda'):
    """自动检测checkpoint中的模型架构并加载"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    
    # 检测模型类型
    if 'enc1.0.weight' in state_dict:
        print("Detected: ImprovedDenoiseUNet (with BatchNorm)")
        from improved_model import DenoiseUNet
        model = DenoiseUNet().to(device)
    elif 'e1.0.weight' in state_dict:
        print("Detected: Original DenoiseUNet")
        from models import DenoiseUNet
        model = DenoiseUNet().to(device)
    elif 'head.0.weight' in state_dict:
        print("Detected: LightDenoiseNet")
        from improved_model import LightDenoiseNet
        model = LightDenoiseNet().to(device)
    else:
        raise ValueError("Unknown model architecture in checkpoint!")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 打印模型信息
    if 'epoch' in ckpt:
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    if 'ssim' in ckpt:
        print(f"Validation SSIM: {ckpt['ssim']:.6f}")
    if 'psnr' in ckpt:
        print(f"Validation PSNR: {ckpt['psnr']:.4f}")
    
    return model


def tensor_to_base64(tensor):
    """将tensor转换为base64编码"""
    import base64
    from io import BytesIO
    
    # tensor: [C, H, W] or [H, W], values in [0, 1]
    arr = tensor.squeeze().cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return b64


def tta_inference(model, inp, device):
    """Test Time Augmentation - 8种增强的平均"""
    preds = []
    
    with torch.no_grad():
        # 1. 原始
        p = model(inp)
        preds.append(p)
        
        # 2. 水平翻转
        p = model(torch.flip(inp, [-1]))
        preds.append(torch.flip(p, [-1]))
        
        # 3. 垂直翻转
        p = model(torch.flip(inp, [-2]))
        preds.append(torch.flip(p, [-2]))
        
        # 4. 旋转90度
        p = model(torch.rot90(inp, 1, [-2, -1]))
        preds.append(torch.rot90(p, -1, [-2, -1]))
        
        # 5. 旋转180度
        p = model(torch.rot90(inp, 2, [-2, -1]))
        preds.append(torch.rot90(p, -2, [-2, -1]))
        
        # 6. 旋转270度
        p = model(torch.rot90(inp, 3, [-2, -1]))
        preds.append(torch.rot90(p, -3, [-2, -1]))
        
        # 7. 水平翻转+旋转90度
        inp_flip = torch.flip(inp, [-1])
        p = model(torch.rot90(inp_flip, 1, [-2, -1]))
        p = torch.rot90(p, -1, [-2, -1])
        preds.append(torch.flip(p, [-1]))
        
        # 8. 垂直翻转+旋转90度
        inp_flip = torch.flip(inp, [-2])
        p = model(torch.rot90(inp_flip, 1, [-2, -1]))
        p = torch.rot90(p, -1, [-2, -1])
        preds.append(torch.flip(p, [-2]))
    
    # 平均所有预测
    pred_avg = torch.stack(preds, dim=0).mean(dim=0)
    return pred_avg


def infer(noise_dir, ckpt_path, out_dir=None, tta=True, batch_size=32):
    """
    推理函数
    Args:
        noise_dir: 噪声图像目录
        ckpt_path: 模型checkpoint路径
        out_dir: 输出目录(可选,用于保存去噪图像)
        tta: 是否使用Test Time Augmentation
        batch_size: 批处理大小
    Returns:
        results: {id: base64_string} 字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 自动加载模型
    model = load_model_auto(ckpt_path, device)
    
    # 获取所有测试图像
    files = sorted([f for f in os.listdir(noise_dir) if f.endswith('_n.png')])
    print(f"Found {len(files)} images to process")
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # 批处理推理
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_imgs = []
        batch_ids = []
        
        # 读取批次图像
        for f in batch_files:
            img = Image.open(os.path.join(noise_dir, f)).convert('L')
            arr = np.array(img).astype(np.float32) / 255.0
            batch_imgs.append(arr)
            idx = int(f.replace('_n.png', ''))
            batch_ids.append(idx)
        
        # 转换为tensor
        batch_tensor = torch.from_numpy(np.stack(batch_imgs)).unsqueeze(1).float().to(device)
        
        # 推理
        with torch.no_grad():
            if tta:
                # TTA对每张图单独处理(因为需要不同的增强)
                preds = []
                for j in range(len(batch_imgs)):
                    inp = batch_tensor[j:j+1]
                    pred = tta_inference(model, inp, device)
                    preds.append(pred)
                batch_pred = torch.cat(preds, dim=0)
            else:
                batch_pred = model(batch_tensor)
        
        # 保存结果
        for j, idx in enumerate(batch_ids):
            pred = batch_pred[j]
            b64 = tensor_to_base64(pred.cpu())
            results[idx] = b64
            
            # 可选:保存PNG图像
            if out_dir:
                arr = pred.squeeze().cpu().numpy()
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr, mode='L')
                img.save(os.path.join(out_dir, f"{idx}_denoised.png"))
        
        if (i + len(batch_files)) % 100 == 0 or (i + len(batch_files)) == len(files):
            print(f"Processed {i + len(batch_files)}/{len(files)} images")
    
    print("Inference completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description='Image Denoising Inference')
    parser.add_argument('--noise_dir', type=str, 
                       default='extracted_files/data/test/noise',
                       help='Directory containing noisy images')
    parser.add_argument('--ckpt_path', type=str, 
                       default='./checkpoints/best_*.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='./predictions',
                       help='Directory to save denoised images (optional)')
    parser.add_argument('--tta', action='store_true', default=True,
                       help='Use Test Time Augmentation')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                       help='Disable TTA for faster inference')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # 自动查找最佳checkpoint
    if '*' in args.ckpt_path or not os.path.exists(args.ckpt_path):
        import glob
        ckpt_dir = os.path.dirname(args.ckpt_path) or './checkpoints'
        candidates = glob.glob(os.path.join(ckpt_dir, 'best_*.pth'))
        if candidates:
            # 按修改时间排序,选最新的
            args.ckpt_path = sorted(candidates, key=os.path.getmtime)[-1]
            print(f"Auto-selected checkpoint: {args.ckpt_path}")
        else:
            # 尝试final.pth
            final_path = os.path.join(ckpt_dir, 'final.pth')
            if os.path.exists(final_path):
                args.ckpt_path = final_path
                print(f"Using final checkpoint: {args.ckpt_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    
    # 执行推理
    results = infer(
        noise_dir=args.noise_dir,
        ckpt_path=args.ckpt_path,
        out_dir=args.out_dir,
        tta=args.tta,
        batch_size=args.batch_size
    )
    
    return results


if __name__ == '__main__':
    results = main()
    print(f"Generated predictions for {len(results)} images")