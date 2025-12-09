FFT 多场景实践项目：多项式乘法、音频降噪与神经网络优化
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FFT](https://img.shields.io/badge/Core-Algorithm-FFT-green.svg)]()
本项目基于快速傅里叶变换（FFT）算法，完整实现三大典型应用场景，并通过对比实验量化FFT在深度学习领域的核心价值。项目代码模块化程度高，可直接复现结果，同时支持按需扩展至自定义数据场景。

📋 项目简介

FFT作为信号处理与计算科学的里程碑算法，其核心价值在于将O(n²)复杂度的卷积、乘法运算优化为O(n log n)，并实现时域与频域的高效转换。本项目聚焦FFT的工程化落地，覆盖从基础算法优化到深度学习数据预处理的全链路，具体包含：

- 📐 多项式乘法：用FFT突破传统暴力乘法的效率瓶颈

- 🎵 音频降噪：基于STFT（短时傅里叶变换）分离语音与噪声，还原清晰音频

- 🧠 神经网络优化：对比FFT降噪前后CNN模型的性能差异，验证频域预处理的价值

通过本项目可直观理解FFT如何从“算法效率优化”延伸至“模型性能提升”，为信号处理与深度学习交叉领域提供实践参考。
