# Semi-Supervised Domain Adaptation via Selective Pseudo Labeling and Progressive Self-Training (ICPR 2020)

### Acknowledgement
The implementation is built on the pytorch implementation of SSDA_MME, which is the baseline model of our proposed SSDA scheme.

### Prerequisites
+ CUDA
+ Python 3.6+
+ PyTorch 0.4.0+
+ Pillow, numpy, tqdm

## Dataset Structure
dataset---
     |
   multi---
     |   |
     |  real
     |  clipart
     |  product
     |  real
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

### Example
#### Train
+ DomainNet (clipart, painting, real, sketch)
+ Office-home (Art, Clipart, Product, Real)
+ Office (amazon, dslr, webcam)

#### Test
+ DomainNet (clipart, painting, real, sketch)

