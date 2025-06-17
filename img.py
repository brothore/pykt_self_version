import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

def generate_spectrum(image_path, spectrum_path):
    """
    生成频谱图供用户识别干扰位置
    
    参数:
    image_path: 输入图像路径
    spectrum_path: 频谱图保存路径
    """
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"错误: 无法读取图像 {image_path}: {e}")
        return None
    
    img_np = np.array(img)
    h, w = img_np.shape
    
    # 应用窗函数减少边缘效应
    win_row = np.hanning(h)
    win_col = np.hanning(w)
    window = np.outer(win_row, win_col)
    windowed = img_np * window
    
    # FFT变换
    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 保存频谱图
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig(spectrum_path)
    plt.close()
    
    # 返回图像尺寸和频谱数据
    return img_np, fshift

def manual_fft_denoise(image_array, fshift, points, output_path, bandwidth=5):
    """
    根据手动指定的点进行频域滤波
    
    参数:
    image_array: 原始图像数组
    fshift: FFT频谱数据
    points: 干扰点坐标列表[(y1, x1), (y2, x2), ...]
    output_path: 输出图像保存路径
    bandwidth: 高斯带阻滤波器的带宽
    """
    h, w = image_array.shape
    
    # 创建初始滤波器
    H = np.ones_like(fshift, dtype=np.float32)
    
    # 创建高斯带阻滤波器
    for freq_y, freq_x in points:
        # 创建当前点的陷波滤波器
        notch_filter = np.ones_like(fshift, dtype=np.float32)
        for x in range(w):
            for y in range(h):
                # 计算到干扰频率点的距离
                dist = np.sqrt((y - freq_y)**2 + (x - freq_x)**2)
                
                # 高斯带阻
                notch_filter[y, x] = (1 - np.exp(-0.5 * (dist / bandwidth)**2))
        
        # 同时抑制对称点
        sym_y = h - 1 - freq_y
        sym_x = w - 1 - freq_x
        for x in range(w):
            for y in range(h):
                dist_sym = np.sqrt((y - sym_y)**2 + (x - sym_x)**2)
                notch_filter[y, x] *= (1 - np.exp(-0.5 * (dist_sym / bandwidth)**2))
        
        # 将当前陷波滤波器应用到总滤波器
        H *= notch_filter
    
    # 应用滤波器
    fshift_filtered = fshift * H
    
    # 逆FFT变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 后处理 - 对比度拉伸
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    img_back = np.clip(img_back, min_val, max_val)
    img_back = (img_back - min_val) / (max_val - min_val) * 255
    
    # 保存结果
    result = img_back.astype(np.uint8)
    result_img = Image.fromarray(result)
    result_img.save(output_path)
    
    # 计算频谱图用于显示
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    filtered_spectrum = 20 * np.log(np.abs(fshift_filtered) + 1)
    
    # 保存处理过程图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231), plt.imshow(image_array, cmap='gray')
    plt.title('原始图像'), plt.axis('off')
    
    plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('频谱图'), plt.axis('off')
    
    plt.subplot(233), plt.imshow(filtered_spectrum, cmap='gray')
    plt.title('滤波后频谱'), plt.axis('off')
    
    plt.subplot(234), plt.imshow(H, cmap='gray')
    plt.title('滤波器'), plt.axis('off')
    
    plt.subplot(235), plt.imshow(result, cmap='gray')
    plt.title('滤波后图像'), plt.axis('off')
    
    # 绘制用户选择的点
    plt.subplot(236)
    plt.imshow(magnitude_spectrum, cmap='gray')
    for i, (y, x) in enumerate(points):
        plt.scatter(x, y, c='red', s=50)
        plt.text(x+5, y+5, f'{y}:{x}', color='red', fontsize=12)
    plt.title('选择的干扰点'), plt.axis('off')
    
    plt.tight_layout()
    
    process_img_path = os.path.splitext(output_path)[0] + '_process.jpg'
    plt.savefig(process_img_path)
    plt.close()
    
    print(f"处理完成! 结果保存至 {output_path}")
    print(f"处理过程图保存至 {process_img_path}")

def main():
    if len(sys.argv) < 2:
        print("使用说明:")
        print("1. 生成频谱图: python script.py spectrum 输入图像.jpg 频谱图.jpg")
        print("2. 执行滤波: python script.py denoise 输入图像.jpg 输出图像.jpg [y1:x1 y2:x2 ...] [bandwidth]")
        print("示例:")
        print("  生成频谱图: python script.py spectrum input.jpg spectrum.jpg")
        print("  执行滤波: python script.py denoise input.jpg output.jpg 100:200 150:300 bandwidth=10")
        return
    
    command = sys.argv[1]
    
    if command == "spectrum":
        if len(sys.argv) != 4:
            print("错误: 需要输入图像路径和频谱图输出路径")
            print("正确格式: python script.py spectrum 输入图像.jpg 频谱图.jpg")
            return
        
        input_path = sys.argv[2]
        spectrum_path = sys.argv[3]
        
        # 生成频谱图
        result = generate_spectrum(input_path, spectrum_path)
        if result:
            print(f"频谱图已保存至 {spectrum_path}")
        
    elif command == "denoise":
        if len(sys.argv) < 5:
            print("错误: 需要输入图像路径、输出图像路径和至少一个干扰点坐标")
            print("正确格式: python script.py denoise 输入图像.jpg 输出图像.jpg [y1:x1 y2:x2 ...] [bandwidth=N]")
            return
        
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        points = []
        bandwidth = 5  # 默认带宽
        
        # 解析坐标点和带宽参数
        for arg in sys.argv[4:]:
            if arg.startswith("bandwidth="):
                try:
                    bandwidth = int(arg.split('=')[1])
                except ValueError:
                    print(f"警告: 无效的带宽值 {arg}，使用默认值5")
            else:
                try:
                    parts = arg.split(':')
                    if len(parts) == 2:
                        y = int(parts[0])
                        x = int(parts[1])
                        points.append((y, x))
                except ValueError:
                    print(f"警告: 忽略无效坐标 {arg}")
        
        if not points:
            print("错误: 未指定有效干扰点坐标")
            return
        
        print(f"使用参数: 干扰点={points}, 带宽={bandwidth}")
        
        # 执行滤波
        result = generate_spectrum(input_path, "_temp_spectrum.jpg")
        if result:
            img_array, fshift = result
            manual_fft_denoise(img_array, fshift, points, output_path, bandwidth)
            # 清理临时文件
            os.remove("_temp_spectrum.jpg")
        
    else:
        print(f"错误: 未知命令 '{command}'")
        print("可用命令: spectrum, denoise")

if __name__ == "__main__":
    main()