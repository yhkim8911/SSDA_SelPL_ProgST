# [Semi-Supervised Domain Adaptation via Selective Pseudo Labeling and Progressive Self-Training (ICPR 2020)](https://arxiv.org/abs/2104.00319)

### Acknowledgement
The implementation is built on the pytorch implementation of [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME), which is the baseline model of our proposed SSDA scheme.

### Prerequisites
+ CUDA
+ Python 3.6+
+ PyTorch 0.4.0+
+ Pillow, numpy, tqdm

## Dataset Structure
```
dataset---
     |
   multi---
     |   |
     |  real
     |  clipart
     |  sketch
     |  painting
   office_home---
     |   |
     |  Art
     |  Clipart
     |  Product
     |  Real
   office---
     |   |
     |  amazon
     |  dslr
     |  webcam
```

### Example
#### Training & Validation
+ DomainNet (clipart, painting, real, sketch)
The proposed SSDA scheme consists of four stages.
An example for running a DA scenario is given as follows.
```
python s1_trainval_baseline.py --net resnet34 --source real --target clipart --num 3
python s2_eval_and_save_features.py --net resnet34 --source real --target clipart --num 3
python s3_selective_pseudo_labeling.py --net resnet34 --source real --target clipart --num 3
python s4_trainval_prog_self_training.py --net resnet34 --source real --target clipart --num 3
```

Or you can run the above stages by simply executing the bash script as follows.
```
bash trainval_SSDA.sh
```

