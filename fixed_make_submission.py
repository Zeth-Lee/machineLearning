import csv
import os
import glob
from fixed_inference import infer

def make_submission(noise_dir, ckpt_path, output_csv='submission.csv', tta=True):
    """
    生成提交文件
    Args:
        noise_dir: 测试集噪声图像目录
        ckpt_path: 模型checkpoint路径
        output_csv: 输出CSV文件名
        tta: 是否使用TTA
    """
    print("="*60)
    print("Starting inference for submission...")
    print("="*60)
    
    # 执行推理
    results = infer(
        noise_dir=noise_dir, 
        ckpt_path=ckpt_path, 
        out_dir='./predictions',  # 可选:保存去噪图像
        tta=tta,
        batch_size=64  # 可以调大以加速
    )
    
    print("\n" + "="*60)
    print("Writing submission file...")
    print("="*60)
    
    # 写入CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'denoised_base64'])
        
        # 按id排序
        for idx in sorted(results.keys()):
            writer.writerow([str(idx), results[idx]])
    
    print(f"\n✓ Submission file saved to: {output_csv}")
    print(f"✓ Total predictions: {len(results)}")
    
    # 验证CSV格式
    print("\nValidating submission format...")
    with open(output_csv, 'r') as f:
        lines = f.readlines()
        print(f"  - Header: {lines[0].strip()}")
        print(f"  - Total rows: {len(lines) - 1}")
        print(f"  - First entry: id={lines[1].split(',')[0]}, base64_length={len(lines[1].split(',')[1])}")
        print(f"  - Last entry: id={lines[-1].split(',')[0]}, base64_length={len(lines[-1].split(',')[1])}")
    
    print("\n" + "="*60)
    print("Submission ready! 🎉")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate submission file')
    parser.add_argument('--noise_dir', type=str, 
                       default='extracted_files/data/test/noise',
                       help='Test noise directory')
    parser.add_argument('--ckpt_path', type=str, 
                       default='./checkpoints/best_*.pth',
                       help='Model checkpoint path (supports wildcards)')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV filename')
    parser.add_argument('--tta', action='store_true', default=True,
                       help='Use Test Time Augmentation (default: True)')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                       help='Disable TTA for faster inference')
    
    args = parser.parse_args()
    
    # 自动查找最佳checkpoint
    if '*' in args.ckpt_path or not os.path.exists(args.ckpt_path):
        ckpt_dir = os.path.dirname(args.ckpt_path) or './checkpoints'
        
        # 优先选择best开头的checkpoint
        candidates = glob.glob(os.path.join(ckpt_dir, 'best_*.pth'))
        if candidates:
            # 如果有多个best,选择SSIM最高的(从文件名解析)
            def extract_ssim(path):
                try:
                    # 尝试从文件名提取SSIM: best_epoch10_ssim0.9234.pth
                    import re
                    match = re.search(r'ssim([\d.]+)', path)
                    if match:
                        return float(match.group(1))
                except:
                    pass
                # 如果解析失败,按修改时间
                return os.path.getmtime(path)
            
            args.ckpt_path = max(candidates, key=extract_ssim)
            print(f"Auto-selected best checkpoint: {args.ckpt_path}")
        else:
            # 尝试final.pth
            final_path = os.path.join(ckpt_dir, 'final.pth')
            if os.path.exists(final_path):
                args.ckpt_path = final_path
                print(f"Using final checkpoint: {args.ckpt_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    
    # 生成提交文件
    make_submission(
        noise_dir=args.noise_dir,
        ckpt_path=args.ckpt_path,
        output_csv=args.output,
        tta=args.tta
    )