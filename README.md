# Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel

> Empirical Study of Transformer’s Attention Mechanism via the Lens of Kernel.

Correspondence to: 
  - Yao-Hung Hubert Tsai (yaohungt@cs.cmu.edu)

## Paper
[**Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel**](https://arxiv.org/pdf/1908.11775.pdf)<br>
[Yao-Hung Hubert Tsai](https://yaohungt.github.io), [Shaojie Bai](https://jerrybai1995.github.io), [Makoto Yamada](https://riken-yamada.github.io), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
Empirical Methods in Natural Language Processing (EMNLP), 2019. 

Please cite our paper if you find our work useful for your research:

```tex
@inproceedings{tsai2019TransformerDissection,
  title={Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Yamada, Makoto and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={EMNLP},
  year={2019},
}
```

## Overview

### What inspires the paper?
*Transformer's attention* and *kernel learning* both concurrently and order-agnostically process all inputs by calculating the similarity between inputs. 

### What have we achieved?
We present a new formulation of attention via the lens of kernel. This formulation highlights naturally the main components of Transformer's attention, enabling better understanding of this mechanism. Recent variants of Transformers can be expressed through these individual components. The approach also paves the way to a larger space of composing Transformer's attention.

### Kernel-Based Formulation of Transformer's Attention

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\mathrm{Attention}\Big(x_q \,\,;\,\, M(x_q, S_{\mathbf{x}_k})\Big) = \sum_{{x_k} \in M(x_q, S_{\mathbf{x}_k})} \frac{k(x_q, x_k)}{\sum_{{x_k}' \in M(x_q, S_{\mathbf{x}_k})}k(x_q, {x_k}')} v(x_k) = \mathbb{E}_{p(x_k|x_q)}\Big[v(x_k)\Big]
" />
</p>

* <img src="https://latex.codecogs.com/gif.latex?x_q" /> : __query__
* <img src="https://latex.codecogs.com/gif.latex?S_{\mathbf{x}_k}" /> : the set for __keys__
* <img src="https://latex.codecogs.com/gif.latex?M(x_q, S_{\mathbf{x}_k})" /> : __set filtering function__, which returns a set with its elements that are visible to <img src="https://latex.codecogs.com/gif.latex?x_q" />
* <img src="https://latex.codecogs.com/gif.latex?k(\cdot , \cdot)" /> : non-negative __kernel__ function
* <img src="https://latex.codecogs.com/gif.latex?v(\cdot)" /> : __value__ function

### Crossmodal Attention for Two Sequences from Distinct Modalities 
<p align="center">
<img src='imgs/cm.png' width="1000px"/>
  
The core of our proposed model are crossmodal transformer and crossmodal attention module.   
  
## Usage

### Prerequisites
- Python 3.6/3.7
- [Pytorch (>=1.0.0) and torchvision](https://pytorch.org/)
- CUDA 10.0 or above

### Datasets

Data files (containing processed MOSI, MOSEI and IEMOCAP datasets) can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0).

To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).

### Run the Code

1. Create (empty) folders for data and pre-trained models:
~~~~
mkdir data pre_trained_models
~~~~

and put the downloaded data in 'data/'.

2. Command as follows
~~~~
python main.py [--FLAGS]
~~~~

Note that the defualt arguments are for unaligned version of MOSEI. For other datasets, please refer to Supplmentary.

### If Using CTC

Transformer requires no CTC module. However, as we describe in the paper, CTC module offers an alternative to applying other kinds of sequence models (e.g., recurrent architectures) to unaligned multimodal streams.

If you want to use the CTC module, plesase install warp-ctc from [here](https://github.com/baidu-research/warp-ctc).

The quick version:
~~~~
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
export WARP_CTC_PATH=/home/xxx/warp-ctc/build
~~~~

### Acknowledgement
Some portion of the code were adapted from the [fairseq](https://github.com/pytorch/fairseq) repo.




