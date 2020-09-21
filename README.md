# Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition


这是我们在Python 3和PyTorch上实现的InterSpeech2019论文 "Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition"。我们提出了一种联合 对抗增强训练 来提升 端到端 系统的鲁棒性。具体来说，我们在训练过程中采用了基于掩码的增强网络、基于attention的encoder-decoder网络和discriminant网络的联合组成方案。

# Requirements
Python 3.5, PyTorch 0.4.0.

# Data
### AISHELL
你可以下载[AISHELL](http://www.aishelltech.com/kysjcp)来运行代码。
你可以使用命令 ```sh run.sh``` 来运行AISHELL，但还是建议你一步步的运行命令。

### Your Own Dataset
你需要建立 train, dev and test 路径. 每一个路径上，都有 ```clean_feats.scp``` ```noisy_feats.scp``` 和 ```text``` 文件. 你可以运行 ```python3 data/prepare_feats.py data_dir feat_dir noise_repeat_num``` 产生带有噪声的数据 ```noisy_feats.scp```.

# Model 模型

该系统由增强网络、端到端ASR网络、fbank特征提取网络和判别网络组成。增强网络将有噪声的STFT特征转化为增强的STFT特征。fbank特征提取网络用于提取归一化的对数fbank特征。端到端ASR模型估计输出标签的后验概率。判别网络用于区分增强的特征和干净的特征。

<div align="center">
<img src="https://github.com/bliunlpr/Robust_e2e_gan/blob/master/fig/framework.Jpeg"  height="400" width="495">
</div>

# Training

### E2E ASR training
你可以使用干净的语音数据和多条件训练策略来训练E2E ASR网络，即用干净语音和噪声语音进行优化。

```
python3 asr_train.py --dataroot Your data directory(including train, dev and test dataset) 
```

### Enhancement Training
你可以通过掩码损失函数(mask loss function)来训练增强网络。

```
python3 enhance_base_train.py --dataroot Your data directory
```
或掩码fbank损失函数(mask fbank loss function)。

```
python3 enhance_fbank_train.py --dataroot Your data directory
```
或gan损失函数(gan loss function).

```
python3 enhance_gan_train.py --dataroot Your data directory
```

### Joint Training

你可以通过ASR损失函数来联合训练增强网络和端到端ASR网络。
note： 可能出现 discriminator 的
```
python3 joint_base_train.py --dataroot Your data directory
```

你也可以通过对抗性损失（adversarial loss）来联合训练增强、端到端ASR网络和判别网络。
```
python3 joint_train.py --dataroot Your data directory
```

# Decoding
我们在所有的实验中都使用beam search进行解码。
```
python3 asr_recog.py 
```
