# Attention-Driven Semantic Masking for Text-Guided Flower Image Editing
Pytorch implementation for Generative Adversarial Networks for  Text-Guided Flower Image Editing. The goal is to introduce a generative adversarial network for efficient image manipulation using natural language descriptions.

### Overview
**Attention-Driven Semantic Masking for Text-Guided Flower Image Editing.**  
Hang An, Wenji Yang, Xinxin Ma, Shen Zhou, Xingyang Miao, Luyu Ouyang.<br> Jiangxi Agricultural University, Nanchang University <br>

### Training
All code was developed and tested on CentOS 7 with Python 3.7 (Anaconda) and PyTorch 1.1.

#### [DAMSM](https://github.com/taoxugit/AttnGAN) model includes a text encoder and an image encoder
- Pre-train DAMSM model for flower dataset:
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/flower.yml --gpu 0
```
#### Our Model
- Train the model for flowerdataset:
```
python main.py --cfg cfg/train_flower.yml --gpu 0
```
`*.yml` files include configuration for training and testing. To reduce the number of parameters used in the model, please edit DF_DIM and/or GF_DIM values in the corresponding `*.yml` files.

### Testing
- Test our model on flower dataset:
```
python main.py --cfg cfg/eval_flower.yml --gpu 0
```


### Evaluation

- To generate images for all captions in the testing dataset, change B_VALIDATION to `True` in the `eval_*.yml`. 
- [Fréchet Inception Distance](https://github.com/mseitzer/pytorch-fid).

### Code Structure
- code/main.py: the entry point for training and testing.
- code/trainer.py: creates the networks, harnesses and reports the progress of training.
- code/model.py: defines the architecture.
- code/attention.py: defines the spatial and channel-wise attentions.
- code/VGGFeatureLoss.py: defines the architecture of the VGG-16.
- code/datasets.py: defines the class for loading images and captions.
- code/pretrain_DAMSM.py: trains the text and image encoders, harnesses and reports the progress of training. 
- code/miscc/losses.py: defines and computes the losses.
- code/miscc/config.py: creates the option list.
- code/miscc/utils.py: additional functions.
