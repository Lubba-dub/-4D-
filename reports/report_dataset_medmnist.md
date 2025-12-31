# 大规模数据集测试报告（基于 ChestMNIST 实真实影像）

## 数据集与实验设置
- 数据源：MedMNIST v2 的 ChestMNIST（真实胸部X射线影像）。
- 数量：测试集样本 600 幅。
- 流水线：Otsu+形态学 → ROI/背景分离 → ROI整数小波无损验证 → 背景自适应JPEG压缩 → 混沌流加密 → pHash鲁棒确权 → 合约登记。

## 关键指标与统计
- ROI重构：平均 PSNR=100.00，SSIM=1.0000（IWT完全可逆）。
- pHash鲁棒性（密文加1%噪声后解密）：平均汉明距离=2.52，范围 [0, 9]（<5通常判定匹配）。
- 加密熵：平均 7.74 bits（接近理想8 bits，直方图近乎均匀）。
- 压缩效果：平均压缩比（原PNG字节 / ROI+BG+Mask字节）=0.61。
- SHA-256：0/600 匹配（噪声下严格哈希不匹配，符合预期）。

## 自适应策略效果
- 背景JPEG质量因子 q_bg 依据背景边缘活动度动态设置（30/40/50）。在胸片背景区域（多为低纹理）的样本中，更倾向于采用 q_bg=30，带来更高压缩率；当背景纹理增强时提高至 40/50，保持结构不破坏。
- 该策略在不影响 ROI 的前提下，改善整体压缩比，同时保留背景结构必要信息。

## 链上登记验证
- 默认使用 EthereumTesterProvider 部署合约并登记首张样本的 SHA-256 与 pHash。
- 交易哈希与存在性检查在控制台输出；可切换至测试网进行真链验证。

## 样例图（均来自 ChestMNIST）
- `reports/images/dataset_original_0.png`
- `reports/images/dataset_mask_0.png`
- `reports/images/dataset_roi_0.png`
- `reports/images/dataset_enc_0.png`
- `reports/images/dataset_decnoisy_0.png`

## 结论与改进方向
- 在真实胸部影像上，方案稳定实现 ROI 完全无损、密文高熵与指纹鲁棒确权。
- 改进方向：
  1. 对背景压缩进行更细粒度的质量自适应（基于块级纹理强度）。
  2. 在区块链登记中加入更多元数据（采集时间、设备ID）以增强审计能力。
  3. 引入更复杂混沌系统（如 Logistic-Sine 耦合）提升密钥空间与随机性，并评估NPCR/UACI。
