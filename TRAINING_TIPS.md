# BrainGNN 训练建议

## 当前测试结果分析

你的测试结果：
- Train Loss: 1.25 → 2.38 (波动大)
- Test Acc: 0.4 → 0.6 (不稳定)
- 数据量: 50个样本
- Epochs: 5

**问题原因：**
1. 数据量太少（50 vs 完整1035）
2. 训练时间太短（5 vs 推荐100+ epochs）
3. Batch size太小（8 vs 推荐32-100）

## 推荐训练配置

### 1. 完整训练（如果内存足够）

```bash
python 03-main.py \
  --batchSize 32 \
  --n_epochs 100 \
  --lr 0.001 \
  --stepsize 20 \
  --gamma 0.5
```

预期结果：
- Train Acc: 70-80%
- Test Acc: 65-75%
- 训练时间: 2-4小时（GPU）

### 2. 中等规模训练（内存有限）

```bash
python 03-main_small.py \
  --batchSize 16 \
  --use_subset True \
  --subset_size 300 \
  --n_epochs 50 \
  --lr 0.001
```

预期结果：
- Train Acc: 65-75%
- Test Acc: 60-70%
- 训练时间: 30-60分钟

### 3. 快速验证（测试代码）

```bash
python 03-main_small.py \
  --batchSize 8 \
  --use_subset True \
  --subset_size 50 \
  --n_epochs 20 \
  --lr 0.001
```

## 超参数调优建议

### 学习率
- 初始: 0.001（比默认0.01更稳定）
- 如果loss不下降: 增大到0.005
- 如果loss震荡: 减小到0.0005

### Batch Size
- 内存充足: 64-100
- 内存有限: 16-32
- 测试: 8

### Epochs
- 完整训练: 100-200
- 快速验证: 20-50
- 观察val_loss，如果不再下降可提前停止

### 正则化参数
```python
--lamb0 1      # 分类loss权重
--lamb1 0      # s1单位正则化（通常为0）
--lamb2 0      # s2单位正则化（通常为0）
--lamb3 0.1    # s1熵正则化
--lamb4 0.1    # s2熵正则化
--lamb5 0.1    # s1一致性正则化
```

## 训练监控

### 1. 使用TensorBoard查看训练曲线

```bash
tensorboard --logdir=./log
```

然后访问 http://localhost:6006

### 2. 观察指标

**正常训练：**
- Train Loss: 逐渐下降
- Val Loss: 先下降后趋于平稳
- Train Acc: 逐渐上升到70-80%
- Val Acc: 上升到65-75%

**过拟合迹象：**
- Train Acc很高（>90%）
- Val Acc很低（<60%）
- Train Loss很低，Val Loss很高

**欠拟合迹象：**
- Train Acc和Val Acc都很低（<60%）
- Loss下降缓慢

## 常见问题

### Q1: Loss波动很大
**原因：** Batch size太小或学习率太大
**解决：** 增大batch size到32，降低学习率到0.001

### Q2: 内存不足
**解决方案：**
1. 减小batch size（16 → 8 → 4）
2. 使用数据子集
3. 增加虚拟内存
4. 使用CPU训练（慢但稳定）

### Q3: 训练太慢
**加速方法：**
1. 确保使用GPU（检查CUDA）
2. 增大batch size
3. 减少数据量（用子集）
4. 使用更少的epochs

### Q4: 准确率不提升
**可能原因：**
1. 学习率太小 → 增大到0.005
2. 数据太少 → 使用完整数据集
3. 模型未收敛 → 增加epochs
4. 数据质量问题 → 检查数据预处理

## 完整训练流程

### 步骤1: 快速验证（5分钟）
```bash
python 03-main_small.py --batchSize 8 --use_subset True --subset_size 50 --n_epochs 10
```
目标：确保代码能运行

### 步骤2: 中等规模测试（30分钟）
```bash
python 03-main_small.py --batchSize 16 --use_subset True --subset_size 200 --n_epochs 30 --lr 0.001
```
目标：观察训练趋势，调整超参数

### 步骤3: 完整训练（2-4小时）
```bash
python 03-main.py --batchSize 32 --n_epochs 100 --lr 0.001
```
目标：获得最佳性能

### 步骤4: 评估和保存
- 查看TensorBoard曲线
- 测试集上评估
- 保存最佳模型（自动保存在./model/）

## 预期性能

根据论文，在ABIDE数据集上：
- 最佳Test Accuracy: 70-75%
- 训练时间: 2-4小时（GPU）
- 收敛Epoch: 50-100

你的测试结果（50样本，5 epochs）：
- Test Acc: 60%
- 这是正常的！数据太少，训练不充分

使用完整数据和足够epochs后，应该能达到70%+的准确率。
