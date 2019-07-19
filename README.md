# Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition

This is an implementation of our InterSpeech2019 paper "Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition" on Python 3, PyTorch. We propose a jointly adversarial enhancement training to boost robustness of end-to-end systems. Specifically, we use a jointly compositional scheme of maskbased enhancement network, attention-based encoder-decoder network and discriminant network during training. 

# Requirements
Python 3.5, PyTorch 0.4.0.

# Data
### AISHELL
You can download [AISHELL](http://www.aishelltech.com/kysjcp) to run the code.

### Your Own Dataset
You need build train, dev and test directory. Each has ```clean_feats.scp``` ```noisy_feats.scp``` and ```text```. 

# Model

The model consists of a generator(G), a discriminator (D) and a classifier(C). The generator G performs the speech enhancement. It transforms the noisy speech signals into the enhanced version. The discriminator D aims to distinguish between the enhanced signals and clean ones. The classifier C classifies senones by features derivated from G. 

<div align="center">
<img src="https://github.com/bliunlpr/Robust_e2e_gan/blob/master/fig/framework.Jpeg"  height="400" width="495">
</div>

# Training

### Baseline(cross-entropy loss)
The baseline system is bulit following the Kaldi chime4/s5 1ch recipe. The acoustic model is a DNN with 7 hidden layers. 
After RBM pre-training, the model is trained by minimizing the cross-entropy loss.

```
python3 main_baseline.py --train_scp Your train directory --val_scp Your val directory --eval_scp  Your test directory 

```

### Deep Adversarial Training
We alternatively train the parameters of D, G and C to fine-tune the model by the Deep Adversarial Training Algorithm. 
Three components are implemented with neural networks and the parameters are updated by stochastic gradient descent.

```
python3 main.py --train_scp Your train directory --val_scp Your val directory --eval_scp  Your test directory 
```

# Decoding
We use the Kaldi WFST decoder for decoding in all the experiments.
```
sh kaldi/decode.sh  

```
