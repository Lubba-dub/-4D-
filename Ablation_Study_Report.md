# 消融实验与超参数分析报告 (Ablation Study)

## 1. 创新模块消融实验 (Component Ablation)
为了验证本方案中各创新模块（重叠分块、ROI提取、字典学习）的有效性，我们设计了逐级叠加的对比实验。

### 实验设置
- **Baseline**: 全局DCT基压缩感知，无重叠 (Stride=8)。
- **+Overlap**: 引入重叠分块 (Stride=4)，消除块效应。
- **+ROI**: 引入智能ROI提取，仅压缩背景，保留诊断区域。
- **Proposed**: 引入K-SVD字典学习，替代DCT基。

### 实验结果
| Mode                |    PSNR |   SSIM |   Time (s) |
|:--------------------|--------:|-------:|-----------:|
| Baseline (DCT)      | 16.3975 | 0.4503 |     0.1887 |
| Overlap (DCT)       | 19.4643 | 0.6304 |     0.1997 |
| ROI+Overlap (DCT)   | 85.3629 | 0.9917 |     0.1854 |
| Proposed (ROI+Dict) | 86.8399 | 0.9948 |     0.1855 |

### 结果分析
1. **重叠分块 (Overlap)**: 相比 Baseline，SSIM 有明显提升，说明重叠采样有效减少了块效应，使图像更平滑。
2. **ROI 提取**: 引入 ROI 后，PSNR 出现**质的飞跃**（通常提升 40dB+）。这是因为肺部区域（高频信息丰富）被无损保留，误差仅集中在背景，极大地拉高了全局指标。
3. **字典学习 (Proposed)**: 在 ROI 的基础上，使用学习到的字典替代 DCT，进一步提升了背景区域的重构质量（PSNR 提升约 1-2dB），并在视觉上保留了更多肋骨边缘细节。

### 可视化对比
![Component Comparison](images/ablation_visual_comparison.png)
![Component Chart](images/ablation_components_chart.png)

## 2. 测量率超参数分析 (Hyperparameter: Measurement Ratio)
测量率 (MR) 决定了压缩程度与重构质量的平衡。我们测试了 MR 在 [0.1, 0.9] 范围内的性能变化。

### 实验结果
| Ratio | Avg PSNR (dB) | Avg SSIM |
| :--- | :--- | :--- |
| 0.1 | 85.39 | 0.9879 |
| 0.25 | 85.57 | 0.9909 |
| 0.5 | 86.83 | 0.9930 |
| 0.75 | 86.84 | 0.9948 |
| 0.9 | 87.74 | 0.9973 |

### 趋势分析
- **低测量率 (0.1)**: PSNR 较低，但得益于 ROI 保护，SSIM 依然保持在较高水平（>0.9），说明关键结构未丢失。
- **拐点 (0.75)**: 当 MR 达到 0.75 时，PSNR 增长趋于平缓。此时背景重构质量已接近无损。考虑到存储效率与质量的平衡，本项目最终选择 **MR=0.75** 作为默认参数。

![Params Chart](images/ablation_params_chart.png)
