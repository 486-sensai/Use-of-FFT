import matplotlib.pyplot as plt
import csv

# 读取8bit音频的CSV采样数据
def read_8bit_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(int(row[0]))  # 8bit数据为0~255的整数
    return data

# 配置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示
plt.figure(figsize=(15, 8))

# 1. 读取降噪前后的采样数据
noisy_data = read_8bit_csv(r"D:\fontsome\神经网络实践\个人分享code\noisy_data.csv")
denoised_data = read_8bit_csv(r"D:\fontsome\神经网络实践\个人分享code\denoised_data.csv")

# 2. 生成时间轴（8KHz采样率，单位：秒）
sample_rate = 8000
time = [i / sample_rate for i in range(len(noisy_data))]

# 3. 绘制带噪音频波形
plt.subplot(2, 1, 1)
plt.plot(time, noisy_data, color='#e74c3c', label='带噪音频（NOISEX-92）', alpha=0.7)
plt.title('8KHz/8bit带噪音频波形（战斗机噪声）', fontsize=14, fontweight='bold')
plt.ylabel('幅值（0~255）', fontsize=12)
plt.ylim(0, 255)  # 8bit音频幅值范围
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 4. 绘制降噪后音频波形
plt.subplot(2, 1, 2)
plt.plot(time, denoised_data, color='#2ecc71', label='FFT降噪后音频', alpha=0.7)
plt.title('FFT低通滤波后的音频波形', fontsize=14, fontweight='bold')
plt.xlabel('时间（秒）', fontsize=12)
plt.ylabel('幅值（0~255）', fontsize=12)
plt.ylim(0, 255)  # 8bit音频幅值范围
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 5. 保存并显示图片
plt.tight_layout()
plt.savefig("audio_denoise_comparison_8bit.png", dpi=300, bbox_inches='tight')
plt.show()

print("对比图已保存为：audio_denoise_comparison_8bit.png")