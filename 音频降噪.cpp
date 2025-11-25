#include<bits/stdc++.h>
#include<fstream>
#include<cmath>
using namespace std;

// 复用你提供的FFT模板核心代码（无需修改）
const int N = 3e6 + 10;
const double pi = acos(-1);
struct Complex {
    double x, y;
    Complex operator+(const Complex& t)const { return {x + t.x, y + t.y}; }
    Complex operator-(const Complex& t)const { return {x - t.x, y - t.y}; }
    Complex operator*(const Complex& t)const { return {x * t.x - y * t.y, x * t.y + y * t.x}; }
}a[N];
int rev[N], tot, bit;
void fft(Complex a[], int inv) {
    for (int i = 0; i < tot; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int mid = 1; mid < tot; mid <<= 1) {
        auto w1 = Complex({cos(pi / mid), inv * sin(pi / mid)});
        for (int i = 0; i < tot; i += mid * 2) {
            auto wk = Complex({1, 0});
            for (int j = 0; j < mid; j++, wk = wk * w1) {
                auto x = a[i + j], y = wk * a[i + j + mid];
                a[i + j] = x + y, a[i + j + mid] = x - y;
            }
        }
    }
}

// 适配8KHz/8bit单声道的WAV文件头结构体
struct WavHeader_8bit {
    char riff[4] = {'R', 'I', 'F', 'F'};
    int32_t file_size;         // 整个文件大小 - 8
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    int32_t fmt_size = 16;     // PCM格式固定16
    int16_t audio_format = 1;  // 1=PCM无压缩
    int16_t channels = 1;      // 单声道（NOISEX-92转制的WAV均为单声道）
    int32_t sample_rate = 8000; // 8KHz采样率（你的噪声文件参数）
    int32_t byte_rate = 8000 * 1; // 字节率=采样率*通道数*位深/8 = 8000*1*8/8
    int16_t block_align = 1;   // 块对齐=通道数*位深/8 = 1
    int16_t bits_per_sample = 8; // 8bit量化位数（你的噪声文件参数）
    char data[4] = {'d', 'a', 't', 'a'};
    int32_t data_size;         // 音频数据字节数（8bit为1字节/采样点）
} header;

// 读取8KHz/8bit单声道的WAV带噪音频（返回uint8_t类型的采样数据）
vector<uint8_t> read_noisy_wav(const string& filename) {
    ifstream fin(filename, ios::binary);
    if (!fin) {
        cerr << "无法打开带噪音频文件：" << filename << endl;
        exit(-1);
    }
    // 读取WAV文件头
    WavHeader_8bit h;
    fin.read((char*)&h, sizeof(WavHeader_8bit));
    // 严格验证文件格式（匹配你的NOISEX-92噪声文件）
    if (h.audio_format != 1 || h.channels != 1 || h.bits_per_sample != 8 || h.sample_rate != 8000) {
        cerr << "文件格式不匹配！要求：8KHz/8bit/单声道/PCM无压缩WAV" << endl;
        exit(-1);
    }
    // 读取音频数据（8bit为单字节，直接存为uint8_t）
    vector<uint8_t> noisy_data(h.data_size);
    fin.read((char*)noisy_data.data(), h.data_size);
    fin.close();
    // 更新全局header，用于写入降噪后的WAV文件
    header = h;
    return noisy_data;
}

// 写入8KHz/8bit单声道的WAV降噪音频
void write_denoised_wav(const string& filename, const vector<uint8_t>& denoised_data) {
    ofstream fout(filename, ios::binary);
    if (!fout) {
        cerr << "无法写入降噪音频文件：" << filename << endl;
        exit(-1);
    }
    // 更新文件头的大小信息（适配降噪后的数据长度）
    header.data_size = denoised_data.size();
    header.file_size = sizeof(WavHeader_8bit) - 8 + header.data_size;
    // 写入文件头和降噪后的音频数据
    fout.write((char*)&header, sizeof(WavHeader_8bit));
    fout.write((char*)denoised_data.data(), denoised_data.size());
    fout.close();
}

// 单帧FFT降噪（适配8bit音频，核心降噪逻辑）
vector<uint8_t> denoise_frame_8bit(const vector<uint8_t>& frame, double cutoff_ratio) {
    int n = frame.size();
    // 计算FFT所需的最小2的幂次长度
    bit = 0;
    while ((1 << bit) < n) bit++;
    tot = 1 << bit;
    // 初始化复数数组：8bit无符号转有符号（中心128为0，消除直流偏置）
    memset(a, 0, sizeof(Complex) * tot);
    for (int i = 0; i < n; i++) {
        a[i].x = (double)frame[i] - 128.0; // 0~255 → -128~127，符合音频信号的正负幅值
    }
    // FFT位逆序置换（模板原有步骤）
    for (int i = 0; i < tot; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    // 正FFT：时域 → 频域
    fft(a, 1);
    // 低通滤波：滤除中间高频区间（噪声主要集中在高频）
    int cutoff = (int)(tot * cutoff_ratio);
    for (int i = cutoff; i < tot - cutoff; i++) {
        a[i].x = 0.0;
        a[i].y = 0.0;
    }
    // 逆FFT：频域 → 时域
    fft(a, -1);
    // 还原为8bit无符号数据（-128~127 → 0~255）
    vector<uint8_t> res(n);
    for (int i = 0; i < n; i++) {
        double val = a[i].x / tot + 128.0; // 逆FFT后归一化，再加128转回原范围
        // 限制数值在8bit范围内（0~255），避免音频失真
        val = max(0.0, min(255.0, val));
        res[i] = (uint8_t)round(val);
    }
    return res;
}

// 导出8bit采样数据到CSV（用于Python绘制降噪前后对比图）
void export_8bit_to_csv(const vector<uint8_t>& data, const string& filename) {
    ofstream fout(filename);
    for (uint8_t val : data) {
        fout << (int)val << "\n"; // 存为整数，方便Python读取
    }
    fout.close();
}

signed main() {
    // ===================== 核心流程：读取带噪音频 → 降噪 → 输出 =====================
    //配置文件路径（采用 NOISEX-92带噪音频文件 "战斗机噪声"）
    string noisy_audio_path = "input.wav";              // 8KHz/8bit带噪音频
    string denoised_audio_path = "f16_denoised.wav";    // 降噪后的输出音频
    
    const int frame_len = 512;        // 分帧长度（8KHz音频适配512/256）
    const double cutoff_ratio = 0.03; // 低通截止比例

    cout << "正在读取带噪音频：" << noisy_audio_path << endl;
    vector<uint8_t> noisy_audio = read_noisy_wav(noisy_audio_path);
    //分帧执行FFT降噪（避免单次处理大数据，提升效率）
    cout << "正在执行FFT低通降噪..." << endl;
    vector<uint8_t> denoised_audio;
    for (int i = 0; i < noisy_audio.size(); i += frame_len) {
        // 提取当前帧
        vector<uint8_t> frame;
        for (int j = i; j < min((int)noisy_audio.size(), i + frame_len); j++) {
            frame.push_back(noisy_audio[j]);
        }
        // 补零到帧长（最后一帧可能长度不足）
        while (frame.size() < frame_len) {
            frame.push_back(128); // 补8bit音频的中心值（直流偏置）
        }
        // 单帧降噪
        vector<uint8_t> denoised_frame = denoise_frame_8bit(frame, cutoff_ratio);
        // 合并降噪后的帧（截断补零部分）
        for (int j = 0; j < frame.size() && i + j < noisy_audio.size(); j++) {
            denoised_audio.push_back(denoised_frame[j]);
        }
    }
    // 写入降噪后的WAV音频
    write_denoised_wav(denoised_audio_path, denoised_audio);
    // 导出采样数据（前5000个点，避免数据量过大）
    int export_len = min(5000, (int)noisy_audio.size());
    vector<uint8_t> noisy_sample(noisy_audio.begin(), noisy_audio.begin() + export_len);
    vector<uint8_t> denoised_sample(denoised_audio.begin(), denoised_audio.begin() + export_len);
    export_8bit_to_csv(noisy_sample, "noisy_data.csv");
    export_8bit_to_csv(denoised_sample, "denoised_data.csv");

    cout << "降噪完成！" << endl;
    cout << "降噪音频文件：" << denoised_audio_path << endl;
    cout << "采样数据已导出：noisy_data.csv | denoised_data.csv（用于绘图）" << endl;

    return 0;
}