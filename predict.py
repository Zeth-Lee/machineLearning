import torch
import argparse
import numpy as np
from PIL import Image
import Beihang
import os
import onnxruntime as ort
from tqdm import tqdm
import time


def predict_full_image_onnx(image, session, patch_size=224):
    """
    使用 ONNX Runtime 会话对整张图像进行分块推理（支持 FP16 模型）
    Args:
        image_path: 输入图像路径
        session: ONNX Runtime 推理会话
        patch_size: 分块大小（默认 224）
    Returns:
        pred_mask: 原始尺寸的预测掩码（0-255 uint8）
    """
    # 1. 读取图像
    # image = Image.open(image_path).convert('RGB')
    image_np = np.asarray(image).transpose(1, 2, 0)  # (H, W, 3)
    hh, ww, cc = image_np.shape

    # 2. 填充为 patch_size 的整数倍
    pad_h = (patch_size - hh % patch_size) % patch_size
    pad_w = (patch_size - ww % patch_size) % patch_size
    image_padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad = image_padded.shape[:2]

    # 3. 创建输出 mask
    pred_mask = np.zeros((h_pad, w_pad), dtype=np.float32)

    # 4. 归一化并转换为 float16（关键：匹配 FP16 模型输入）
    image_float = image_padded.astype(np.float32) / 255.0
    # ✅ 如果模型是 FP16，建议输入也用 float16
    image_float16 = image_float.astype(np.float16)  # (H, W, 3)
    image_float16 = np.transpose(image_float16, (2, 0, 1))  # (3, H, W)

    # 5. 获取输入/输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 6. 分块推理
    for i in range(0, h_pad, patch_size):
        for j in range(0, w_pad, patch_size):
            patch = image_float16[:, i:i + patch_size, j:j + patch_size]  # (3, 224, 224)
            patch = np.expand_dims(patch, axis=0)  # (1, 3, 224, 224)

            # ONNX 推理（输入为 float16）
            pred = session.run([output_name], {input_name: patch})[0]  # (1, 1, 224, 224)

            # 二值化（输出可能是 float32 或 float16，统一转为 float32 处理）
            pred_binary = (pred > 0.5).astype(np.float32)
            pred_patch = pred_binary[0, 0]  # (224, 224)
            pred_mask[i:i + patch_size, j:j + patch_size] = pred_patch

    # 7. 去除填充，恢复原始尺寸
    pred_mask = pred_mask[:hh, :ww]

    # 8. 转为 0-255 的 uint8 图像
    pred_mask = (pred_mask * 255).astype(np.uint8)

    return pred_mask


def pre_process(image_path, save_dir):
    seed = 1616
    process_num = 1
    cam = "01"
    untar_dir = "/data/yxq/workspace/data/untar/"
    prestored_data = "/data/yxq/workspace/data/models20250108/config/l04/prestored_data/"
    imgs = Beihang.preprocess(image_path, untar_dir=untar_dir, prestored_data=prestored_data, save_dir=save_dir, cam=cam, seed=seed, process_num=process_num)
    return imgs


def main(args):
    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"模型文件未找到: {args.onnx_model}")

    L1_imgs = pre_process(args.image_path, args.output_path)
    # print("L1_imgs", L1_imgs[0].shape)

    # 设置 providers（优先使用 CUDA）
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        # print("Available providers at import time:", ort.get_available_providers())
        # import pdb
        # pdb.set_trace()
        session = ort.InferenceSession(args.onnx_model, providers=providers)
        active_provider = session.get_providers()[0]
        print(f"✅ 成功加载 ONNX 模型，使用设备: {active_provider}")

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        exit(1)

    os.makedirs(args.output_path, exist_ok=True)

    for i, img in enumerate(L1_imgs):
        start_time = time.time()
        prediction = predict_full_image_onnx(img, session, patch_size=224)
        end_time = time.time()
        print(f"✅ 第{i + 1}/{len(L1_imgs)}张图片推理完成，耗时: {end_time - start_time:.2f} 秒")

        base_name = f"result_{i + 1}.tif"
        out_path = os.path.join(args.output_path, base_name)

        result = Image.fromarray(prediction)
        result.save(out_path)
    print(f"💾 预测结果已保存: {args.output_path}")


if __name__ == "__main__":
    # image_path = "./data/XSD-test/converted.tif"
    # output_path = "./data/XSD-test/predicted-resunet.tif"

    parser = argparse.ArgumentParser(description="ResUnet full image prediction")
    parser.add_argument('--image_path', type=str, default="/data/yxq/workspace/data/L0/", help='Path to input image')
    parser.add_argument('--output_path', type=str, default="/data/yxq/workspace/data/output/python/Beihang/", help='Path to output image')
    # parser.add_argument('--pre_process_path', type=str, default='/output/python/Beihang/')
    parser.add_argument('--onnx_model', type=str, default='/data/yxq/workspace/code/Code_Library/Beihang/test/test/checkpoints/quantized_gpu/resunet_fp16.onnx',
                        help='Path to model checkpoint')
    args = parser.parse_args()
    main(args)