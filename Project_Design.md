# 项目设计方案：基于ROI/IWT/混沌加密与区块链确权的医疗影像安全传输

## 目标
- 面向真实医疗场景，实现“高效传输 + 隐私保护 + 可确权”的完整处理链。
- 全流程采用传统DIP算法（非深度学习），满足课程要求。
- 支持开源大规模医疗数据集测试与“链上”登记验真。

## 架构
- 数据层：MedMNIST数据集加载与批处理（ChestMNIST，≥600幅）。
- 处理层：
  - ROI分割：Otsu + 形态学开/闭优化。
  - 压缩：ROI无损（PNG/IWT可逆验证），背景有损（JPEG），掩膜无损（PNG）。
  - 加密：Logistic混沌流加密（XOR）。
  - 认证：DCT感知哈希（pHash）、SHA-256。
- 链上层：Solidity合约 ImageRegistry，登记密文哈希与pHash；默认使用 EthereumTesterProvider（本地内存链），可切换至以太坊测试网（Sepolia）。

## 代码模块
- `src/dipsecure/data_loader.py`：MedMNIST加载（灰度化、数量上限）。
- `src/dipsecure/roi.py`：Otsu、形态学、ROI/背景分离。
- `src/dipsecure/iwt.py`：CDF 5/3 整数小波变换，完美重构验证。
- `src/dipsecure/chaos_crypto.py`：LogisticKey与XOR加密/解密。
- `src/dipsecure/phash.py`：pHash生成与汉明距离。
- `src/dipsecure/metrics.py`：PSNR、SSIM、压缩字节统计（PNG/JPEG/Mask）。
- `src/dipsecure/blockchain.py`：Solidity合约编译、部署、登记与查询（支持EthereumTester与HTTP RPC）。
- `src/run_pipeline.py`：主流程脚本，批量处理、采样可视化、链上登记与汇总指标。

## 合约接口
```
function register(bytes32 contentHash, bytes32 pHash, string patientId, string note)
function exists(bytes32 contentHash) view returns (bool)
```

## 依赖
- Python: opencv-python, scikit-image, PyWavelets, matplotlib, medmnist, web3, eth-tester, py-solc-x, py-evm
- 操作系统：Windows，PowerShell。

## 运行与复现
1. 安装依赖：
   ```
   pip install opencv-python scikit-image PyWavelets matplotlib medmnist web3 eth-tester py-solc-x py-evm
   ```
2. 运行主流程（默认本地内存链）：
   ```
   python DIP_Project/src/run_pipeline.py
   ```
3. 切换至以太坊测试网（需要你提供RPC与私钥）：
   - 设置环境变量：
     - `ETH_RPC_URL`= 你的Sepolia或其他测试网RPC地址
     - `PRIVATE_KEY`= 你的测试账户私钥（建议使用测试账户与最小权限；切勿泄露正式密钥）
   - 运行：
     ```
     ETH_RPC_URL=<rpc> PRIVATE_KEY=<key> python DIP_Project/src/run_pipeline.py
     ```
   - 结果：自动部署合约并登记首张样本的哈希与pHash。

## 指标与输出
- 每张图：
  - ROI无损重构验证（IWT）：PSNR≈100、SSIM≈1。
  - 压缩统计：`roi_png`、`bg_jpg`、`mask_png`与`total`字节。
  - 加密直方图（可视化）、pHash鲁棒性（汉明距离低于5一般视为匹配）。
  - SHA-256对比（有噪情况下Fail）。
- 汇总：
  - 平均PSNR/SSIM、平均pHash汉明距离、SHA匹配计数（应为0）。
  - `reports/images/`下生成样例图。

## 安全与隐私
- 不写入任何真实病人信息；样例`patientId`为伪ID。
- 不保存私钥；使用环境变量注入，避免硬编码。

## 后续优化建议
- 压缩感知（CS）：混沌测量矩阵，采样即压缩+加密，提升前端效率。
- 零水印增强：结合DWT/DCT联合特征，提高对几何攻击的鲁棒性。
- ROI自适应：根据器官类型动态调整形态学参数与JPEG质量因子。

