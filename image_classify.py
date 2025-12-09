import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fft import fft2, ifft2
import warnings
warnings.filterwarnings('ignore')

# 设置路径
CIFAR10_PATH = r"D:\fontsome\神经网络实践\个人分享code\分类任务\cifar-10-batches-py"
RESULTS_PATH = r"D:\fontsome\神经网络实践\个人分享code\分类任务\results"

# 创建结果目录
os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*70)
print("FFT降噪对神经网络泛化能力影响的实验")
print(f"数据集路径: {CIFAR10_PATH}")
print(f"结果保存路径: {RESULTS_PATH}")
print("="*70)

# 1. 加载本地CIFAR-10数据集
print("\n1. 加载本地CIFAR-10数据集...")

def load_cifar10_batch(filepath):
    """加载单个CIFAR-10批次文件"""
    with open(filepath, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        Y = np.array(Y)
    return X, Y

def load_cifar10(data_dir):
    """加载所有CIFAR-10批次文件"""
    # 训练批次
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, f'data_batch_{b}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    
    # 测试批次
    X_test, Y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    
    # 归一化到0-1范围
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # 类别名称
    with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        label_names = [x.decode('utf-8') for x in datadict[b'label_names']]
    
    return X_train, Y_train, X_test, Y_test, label_names

# 加载数据
X_train, Y_train, X_test, Y_test, label_names = load_cifar10(CIFAR10_PATH)

print(f"训练集: {X_train.shape}, 标签: {Y_train.shape}")
print(f"测试集: {X_test.shape}, 标签: {Y_test.shape}")
print(f"类别: {label_names}")

# 为了加快实验速度，使用数据子集
print("\n为了加快训练速度，使用5000个训练样本和2000个测试样本...")
np.random.seed(42)
train_indices = np.random.choice(len(X_train), 5000, replace=False)
test_indices = np.random.choice(len(X_test), 2000, replace=False)

X_train_small = X_train[train_indices]
Y_train_small = Y_train[train_indices]
X_test_small = X_test[test_indices]
Y_test_small = Y_test[test_indices]

print(f"子集训练集: {X_train_small.shape}")
print(f"子集测试集: {X_test_small.shape}")

# 2. 添加真实世界噪声
print("\n2. 添加真实世界噪声...")

def add_realistic_cifar_noise(images, noise_level=0.15):
    """
    为CIFAR-10图像添加真实世界噪声
    """
    noisy_images = images.copy()
    
    for i in range(len(images)):
        img = images[i].copy()
        
        # 1. 高斯噪声（传感器噪声）
        gaussian_noise = np.random.normal(0, noise_level * 0.25, img.shape)
        
        # 2. 颜色通道噪声（模拟白平衡问题）
        for ch in range(3):  # RGB通道
            channel_noise = np.random.normal(0, noise_level * 0.1)
            img[:, :, ch] += channel_noise
        
        # 3. 块状噪声（模拟JPEG压缩伪影）
        if np.random.random() < 0.4:  # 40%的样本添加块状噪声
            block_size = 8
            h, w, c = img.shape
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block_noise = np.random.normal(0, noise_level * 0.15)
                    img[y:min(y+block_size, h), x:min(x+block_size, w), :] += block_noise
        
        # 4. 随机对比度变化
        contrast = np.random.uniform(0.8, 1.2)
        img = img * contrast
        
        # 组合所有噪声
        img = img + gaussian_noise
        
        # 确保像素值在有效范围内
        noisy_images[i] = np.clip(img, 0, 1)
    
    return noisy_images

# 添加噪声
X_train_noisy = add_realistic_cifar_noise(X_train_small, noise_level=0.15)
X_test_noisy = add_realistic_cifar_noise(X_test_small, noise_level=0.15)

# 3. FFT降噪处理
print("\n3. 对噪声图像进行FFT降噪处理...")

def fft_denoise_cifar(images, threshold_percentile=85):
    """
    对CIFAR-10图像进行FFT降噪
    """
    denoised_images = np.zeros_like(images)
    
    for i in range(len(images)):
        img = images[i]
        denoised_img = np.zeros_like(img)
        
        # 对每个颜色通道分别处理
        for ch in range(3):
            channel_img = img[:, :, ch]
            
            # 2D FFT
            fft_img = fft2(channel_img)
            fft_shifted = np.fft.fftshift(fft_img)
            
            # 计算幅度谱
            magnitude_spectrum = np.abs(fft_shifted)
            
            # 设置自适应阈值
            threshold = np.percentile(magnitude_spectrum, threshold_percentile)
            
            # 创建掩码：保留幅度高于阈值的频率成分（主要频率）
            mask = magnitude_spectrum > threshold
            
            # 应用掩码
            fft_filtered = fft_shifted * mask
            
            # 逆FFT
            fft_restored = np.fft.ifftshift(fft_filtered)
            denoised_channel = np.real(ifft2(fft_restored))
            
            # 归一化
            denoised_channel = np.clip(denoised_channel, 0, 1)
            denoised_img[:, :, ch] = denoised_channel
        
        denoised_images[i] = denoised_img
    
    return denoised_images

# 应用FFT降噪
X_train_denoised = fft_denoise_cifar(X_train_noisy, threshold_percentile=85)
X_test_denoised = fft_denoise_cifar(X_test_noisy, threshold_percentile=85)

# 4. 可视化噪声和降噪效果
print("\n4. 可视化噪声和降噪效果...")

def save_visualization_effects(clean, noisy, denoised, labels, label_names, n_samples=8):
    """保存可视化效果图"""
    fig, axes = plt.subplots(4, n_samples, figsize=(20, 12))
    
    # 随机选择样本
    indices = np.random.choice(len(clean), n_samples, replace=False)
    
    for idx, sample_idx in enumerate(indices):
        # 干净图像
        axes[0, idx].imshow(clean[sample_idx])
        axes[0, idx].set_title(f'Clean\n{label_names[labels[sample_idx]]}', 
                               fontsize=10, color='blue')
        axes[0, idx].axis('off')
        
        # 噪声图像
        axes[1, idx].imshow(noisy[sample_idx])
        axes[1, idx].set_title(f'Noisy', fontsize=10, color='red')
        axes[1, idx].axis('off')
        
        # 降噪图像
        axes[2, idx].imshow(denoised[sample_idx])
        axes[2, idx].set_title(f'FFT Denoised', fontsize=10, color='green')
        axes[2, idx].axis('off')
        
        # 噪声分布热图
        noise = np.mean(np.abs(noisy[sample_idx] - clean[sample_idx]), axis=2)
        im = axes[3, idx].imshow(noise, cmap='hot', vmin=0, vmax=0.5)
        axes[3, idx].set_title(f'Noise Intensity', fontsize=10)
        axes[3, idx].axis('off')
        
        # 添加颜色条（只添加一次）
        if idx == n_samples - 1:
            plt.colorbar(im, ax=axes[3, idx], fraction=0.046, pad=0.04)
    
    plt.suptitle('CIFAR-10: FFT Denoising Effects Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(RESULTS_PATH, "01_denoising_effects.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存效果图到: {save_path}")
    plt.show()

save_visualization_effects(X_train_small, X_train_noisy, X_train_denoised, 
                          Y_train_small, label_names, n_samples=8)

# 5. 准备数据用于MLP
print("\n5. 准备数据用于MLP训练...")

# 展平图像数据 (32x32x3 -> 3072)
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

X_train_clean_flat = flatten_images(X_train_small)
X_train_noisy_flat = flatten_images(X_train_noisy)
X_train_denoised_flat = flatten_images(X_train_denoised)

X_test_clean_flat = flatten_images(X_test_small)
X_test_noisy_flat = flatten_images(X_test_noisy)
X_test_denoised_flat = flatten_images(X_test_denoised)

print(f"展平后训练数据形状: {X_train_clean_flat.shape}")
print(f"展平后测试数据形状: {X_test_clean_flat.shape}")

# 6. 训练和评估MLP模型
print("\n6. 训练和评估MLP模型...")

def train_evaluate_mlp(X_train, X_test, y_train, y_test, model_name, max_iter=100):
    """训练和评估MLP模型"""
    print(f"\n{'='*50}")
    print(f"训练 {model_name} 模型")
    print('='*50)
    
    # 创建MLP分类器
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 三个隐藏层
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    
    # 训练模型
    print("训练中...")
    mlp.fit(X_train, y_train)
    
    # 评估模型
    train_acc = mlp.score(X_train, y_train)
    test_acc = mlp.score(X_test, y_test)
    
    print(f"\n{model_name} 结果:")
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"最终损失值: {mlp.loss_:.4f}")
    print(f"迭代次数: {mlp.n_iter_}")
    
    return mlp, train_acc, test_acc

# 训练三个模型
print("\n开始训练三个不同数据集的MLP模型:")

# 6.1 使用干净数据
mlp_clean, train_acc_clean, test_acc_clean = train_evaluate_mlp(
    X_train_clean_flat, X_test_clean_flat, 
    Y_train_small, Y_test_small, 
    "干净数据"
)

# 6.2 使用噪声数据
mlp_noisy, train_acc_noisy, test_acc_noisy = train_evaluate_mlp(
    X_train_noisy_flat, X_test_noisy_flat,
    Y_train_small, Y_test_small,
    "噪声数据"
)

# 6.3 使用FFT降噪数据
mlp_denoised, train_acc_denoised, test_acc_denoised = train_evaluate_mlp(
    X_train_denoised_flat, X_test_denoised_flat,
    Y_train_small, Y_test_small,
    "FFT降噪数据"
)

# 7. 可视化训练过程和结果对比
print("\n7. 可视化训练过程和结果对比...")

def save_training_comparison(models_info, save_name="02_training_comparison.png"):
    """保存训练过程对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 损失曲线对比
    for i, (mlp, label, color) in enumerate(models_info):
        if hasattr(mlp, 'loss_curve_'):
            axes[0, 0].plot(mlp.loss_curve_, color=color, linewidth=2, 
                           label=label, alpha=0.8)
    
    axes[0, 0].set_xlabel('Iterations', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 验证分数对比
    for i, (mlp, label, color) in enumerate(models_info):
        if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
            axes[0, 1].plot(mlp.validation_scores_, color=color, linewidth=2, 
                           label=label, alpha=0.8)
    
    if hasattr(models_info[0][0], 'validation_scores_') and models_info[0][0].validation_scores_:
        axes[0, 1].set_xlabel('Iterations', fontsize=12)
        axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12)
        axes[0, 1].set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 训练准确率对比柱状图
    train_accs = [train_acc_clean, train_acc_noisy, train_acc_denoised]
    test_accs = [test_acc_clean, test_acc_noisy, test_acc_denoised]
    labels_bar = ['Clean Data', 'Noisy Data', 'FFT Denoised']
    colors_bar = ['blue', 'red', 'green']
    
    x_pos = np.arange(len(labels_bar))
    width = 0.35
    
    bars1 = axes[0, 2].bar(x_pos - width/2, train_accs, width, 
                          label='Training', color=colors_bar, alpha=0.7)
    bars2 = axes[0, 2].bar(x_pos + width/2, test_accs, width, 
                          label='Testing', color=colors_bar, alpha=0.7, 
                          hatch='//')
    
    axes[0, 2].set_xlabel('Training Data', fontsize=12)
    axes[0, 2].set_ylabel('Accuracy', fontsize=12)
    axes[0, 2].set_title('Training vs Testing Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels_bar, rotation=15)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 测试准确率对比
    axes[1, 0].bar(labels_bar, test_accs, color=colors_bar, alpha=0.7)
    axes[1, 0].set_xlabel('Training Data', fontsize=12)
    axes[1, 0].set_ylabel('Test Accuracy', fontsize=12)
    axes[1, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (label, acc) in enumerate(zip(labels_bar, test_accs)):
        axes[1, 0].text(i, acc + 0.005, f'{acc:.4f}', 
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 泛化差距对比
    generalization_gaps = [train_acc_clean - test_acc_clean,
                          train_acc_noisy - test_acc_noisy,
                          train_acc_denoised - test_acc_denoised]
    
    bars = axes[1, 1].bar(labels_bar, generalization_gaps, color=colors_bar, alpha=0.7)
    axes[1, 1].set_xlabel('Training Data', fontsize=12)
    axes[1, 1].set_ylabel('Generalization Gap', fontsize=12)
    axes[1, 1].set_title('Generalization Gap (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (label, gap) in enumerate(zip(labels_bar, generalization_gaps)):
        axes[1, 1].text(i, gap + (0.001 if gap >= 0 else -0.01), f'{gap:.4f}', 
                       ha='center', va='bottom' if gap >= 0 else 'top', 
                       fontsize=11, fontweight='bold')
    
    # 改进百分比
    improvement_vs_noise = ((test_acc_denoised - test_acc_noisy) / test_acc_noisy) * 100
    improvement_vs_clean = ((test_acc_denoised - test_acc_clean) / test_acc_clean) * 100
    
    improvements = [0, (test_acc_noisy - test_acc_clean) / test_acc_clean * 100, 
                    improvement_vs_noise]
    
    bars = axes[1, 2].bar(labels_bar, improvements, color=colors_bar, alpha=0.7)
    axes[1, 2].set_xlabel('Training Data', fontsize=12)
    axes[1, 2].set_ylabel('Improvement (%)', fontsize=12)
    axes[1, 2].set_title('Accuracy Improvement Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (label, imp) in enumerate(zip(labels_bar, improvements)):
        axes[1, 2].text(i, imp + (1 if imp >= 0 else -2), f'{imp:+.1f}%', 
                       ha='center', va='bottom' if imp >= 0 else 'top', 
                       fontsize=11, fontweight='bold')
    
    plt.suptitle('FFT Denoising Effects on MLP Performance (CIFAR-10)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(RESULTS_PATH, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存训练对比图到: {save_path}")
    plt.show()

# 准备模型信息
models_info = [
    (mlp_clean, 'Clean Data', 'blue'),
    (mlp_noisy, 'Noisy Data', 'red'),
    (mlp_denoised, 'FFT Denoised', 'green')
]

save_training_comparison(models_info)

# 8. 频域分析可视化
print("\n8. 频域分析可视化...")

def save_frequency_analysis(clean_img, noisy_img, denoised_img, save_name="03_frequency_analysis.png"):
    """保存频域分析图"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    images = [clean_img, noisy_img, denoised_img]
    titles = ['Clean Image', 'Noisy Image', 'FFT Denoised Image']
    
    for row, (img, title) in enumerate(zip(images, titles)):
        # 显示原图
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'{title}', fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # 计算每个通道的频域幅度谱
        for ch in range(3):  # RGB通道
            channel_img = img[:, :, ch]
            
            # 2D FFT
            fft_img = fft2(channel_img)
            fft_shifted = np.fft.fftshift(fft_img)
            
            # 幅度谱（对数变换）
            magnitude_spectrum = np.log(1 + np.abs(fft_shifted))
            
            axes[row, ch+1].imshow(magnitude_spectrum, cmap='hot')
            axes[row, ch+1].set_title(f'Channel {["R", "G", "B"][ch]} Frequency', fontsize=10)
            axes[row, ch+1].axis('off')
    
    plt.suptitle('Frequency Domain Analysis: Clean vs Noisy vs Denoised', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(RESULTS_PATH, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存频域分析图到: {save_path}")
    plt.show()

# 选择一个样本进行频域分析
sample_idx = np.random.randint(0, len(X_train_small))
save_frequency_analysis(X_train_small[sample_idx], 
                       X_train_noisy[sample_idx], 
                       X_train_denoised[sample_idx])

# 9. 错误分析可视化
print("\n9. 错误分析可视化...")

def analyze_and_visualize_errors(models, X_tests, y_test, label_names, save_name="04_error_analysis.png"):
    """分析和可视化错误"""
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    
    model_names = ['Clean Data', 'Noisy Data', 'FFT Denoised']
    colors = ['blue', 'red', 'green']
    
    for row, (model, X_test_flat, model_name, color) in enumerate(zip(models, X_tests, model_names, colors)):
        # 预测
        y_pred = model.predict(X_test_flat)
        
        # 计算错误
        errors = y_pred != y_test
        error_indices = np.where(errors)[0]
        
        # 准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 显示准确率
        axes[row, 0].text(0.5, 0.5, f'{model_name}\nAccuracy: {accuracy:.4f}', 
                         fontsize=14, fontweight='bold', color=color,
                         ha='center', va='center', transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')
        
        # 显示一些错误样本
        if len(error_indices) > 0:
            display_indices = error_indices[:4] if len(error_indices) >= 4 else error_indices
            
            for col, idx in enumerate(display_indices, start=1):
                # 获取对应的原始图像（需要从展平数据中恢复）
                img_idx = idx
                # 注意：这里我们需要使用对应的测试集图像
                # 由于我们使用了不同的测试集（clean, noisy, denoised），需要分别处理
                if model_name == 'Clean Data':
                    img = X_test_small[img_idx]
                elif model_name == 'Noisy Data':
                    img = X_test_noisy[img_idx]
                else:  # FFT Denoised
                    img = X_test_denoised[img_idx]
                
                axes[row, col].imshow(img)
                true_label = label_names[y_test[img_idx]]
                pred_label = label_names[y_pred[img_idx]]
                axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                        fontsize=9, color='red')
                axes[row, col].axis('off')
        
        # 清空多余的子图
        for col in range(len(display_indices) + 1, 5):
            axes[row, col].axis('off')
    
    plt.suptitle('Error Analysis: Misclassified Samples', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(RESULTS_PATH, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存错误分析图到: {save_path}")
    plt.show()

# 准备数据用于错误分析
X_tests_flat = [X_test_clean_flat, X_test_noisy_flat, X_test_denoised_flat]
models_list = [mlp_clean, mlp_noisy, mlp_denoised]

analyze_and_visualize_errors(models_list, X_tests_flat, Y_test_small, label_names)

# 10. 综合结果总结图
print("\n10. 生成综合结果总结图...")

def save_summary_chart(test_accuracies, save_name="05_summary_chart.png"):
    """保存综合结果总结图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    labels = ['Clean Data', 'Noisy Data', 'FFT Denoised']
    colors = ['blue', 'red', 'green']
    
    # 1. 准确率对比雷达图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = test_accuracies
    values += values[:1]
    angles += angles[:1]
    
    ax = axes[0]
    ax = plt.subplot(131, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2, color='purple')
    ax.fill(angles, values, alpha=0.25, color='purple')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.set_title('Test Accuracy Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    # 添加数值标签
    for angle, value, label in zip(angles[:-1], test_accuracies, labels):
        ax.text(angle, value + 0.02, f'{value:.3f}', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. 准确率改进瀑布图
    ax = axes[1]
    x_pos = np.arange(len(labels))
    
    # 计算累积改进
    baseline = test_accuracies[0]
    improvements = [0, test_accuracies[1] - baseline, test_accuracies[2] - test_accuracies[1]]
    
    # 创建瀑布图
    waterfall = np.zeros(len(labels))
    waterfall[0] = baseline
    
    for i in range(1, len(labels)):
        waterfall[i] = waterfall[i-1] + improvements[i]
    
    # 绘制瀑布图
    ax.bar(x_pos, waterfall, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加连接线
    for i in range(len(labels)-1):
        ax.plot([i, i+1], [waterfall[i], waterfall[i+1]], 'k-', linewidth=2)
    
    # 添加数值标签
    for i, (label, value, imp) in enumerate(zip(labels, waterfall, improvements)):
        ax.text(i, value + 0.005, f'{value:.3f}', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
        if i > 0:
            ax.text(i-0.5, waterfall[i-1] + imp/2, f'{imp:+.3f}', 
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Training Data', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Accuracy Improvement Waterfall', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 性能指标对比表
    ax = axes[2]
    ax.axis('tight')
    ax.axis('off')
    
    # 计算各项指标
    train_accs = [train_acc_clean, train_acc_noisy, train_acc_denoised]
    test_accs = test_accuracies
    gaps = [train_acc_clean - test_acc_clean, 
            train_acc_noisy - test_acc_noisy, 
            train_acc_denoised - test_acc_denoised]
    
    # 计算改进百分比
    improv_vs_noise = ((test_acc_denoised - test_acc_noisy) / test_acc_noisy) * 100
    improv_vs_clean = ((test_acc_denoised - test_acc_clean) / test_acc_clean) * 100
    
    # 创建表格数据
    table_data = [
        ['Metric', 'Clean Data', 'Noisy Data', 'FFT Denoised'],
        ['Train Accuracy', f'{train_acc_clean:.4f}', f'{train_acc_noisy:.4f}', f'{train_acc_denoised:.4f}'],
        ['Test Accuracy', f'{test_acc_clean:.4f}', f'{test_acc_noisy:.4f}', f'{test_acc_denoised:.4f}'],
        ['Generalization Gap', f'{gaps[0]:.4f}', f'{gaps[1]:.4f}', f'{gaps[2]:.4f}'],
        ['Improvement vs Noise', '-', '-', f'{improv_vs_noise:+.2f}%'],
        ['Improvement vs Clean', '-', f'{(test_acc_noisy-test_acc_clean)/test_acc_clean*100:+.2f}%', 
         f'{improv_vs_clean:+.2f}%']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 设置单元格颜色
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # 标题行
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif j == 0:  # 指标列
                table[(i, j)].set_facecolor('#f1f1f1')
                table[(i, j)].set_text_props(weight='bold')
            elif i >= 4:  # 改进百分比行
                table[(i, j)].set_facecolor('#e6f7ff')
    
    ax.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold', y=0.98)
    
    plt.suptitle('CIFAR-10: FFT Denoising for Improved Neural Network Generalization', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(RESULTS_PATH, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存综合总结图到: {save_path}")
    plt.show()

# 生成综合结果总结
test_accuracies = [test_acc_clean, test_acc_noisy, test_acc_denoised]
save_summary_chart(test_accuracies)

# 11. 打印最终结果总结
print("\n" + "="*80)
print("实验最终结果总结")
print("="*80)
print(f"{'训练数据':<15} {'训练准确率':<12} {'测试准确率':<12} {'泛化差距':<12} {'改进分析':<20}")
print("-"*80)

improvement_vs_noise = ((test_acc_denoised - test_acc_noisy) / test_acc_noisy) * 100
improvement_vs_clean = ((test_acc_denoised - test_acc_clean) / test_acc_clean) * 100

print(f"{'干净数据':<15} {train_acc_clean:<12.4f} {test_acc_clean:<12.4f} "
      f"{train_acc_clean-test_acc_clean:<12.4f} {'基准':<20}")
print(f"{'噪声数据':<15} {train_acc_noisy:<12.4f} {test_acc_noisy:<12.4f} "
      f"{train_acc_noisy-test_acc_noisy:<12.4f} "
      f"{f'下降{(test_acc_clean-test_acc_noisy)/test_acc_clean*100:.1f}%':<20}")
print(f"{'FFT降噪':<15} {train_acc_denoised:<12.4f} {test_acc_denoised:<12.4f} "
      f"{train_acc_denoised-test_acc_denoised:<12.4f} "
      f"{f'提升{improvement_vs_noise:.1f}% (vs噪声)':<20}")

print("-"*80)
print("关键发现:")
print(f"1. FFT降噪相比噪声数据提升了 {improvement_vs_noise:.2f}% 的测试准确率")
print(f"2. 泛化差距改善: 噪声数据({train_acc_noisy-test_acc_noisy:.4f}) → "
      f"FFT降噪({train_acc_denoised-test_acc_denoised:.4f})")
print(f"3. FFT降噪后的准确率恢复到了干净数据的 {test_acc_denoised/test_acc_clean*100:.1f}%")
print(f"4. 所有可视化结果已保存到: {RESULTS_PATH}")
print("="*80)