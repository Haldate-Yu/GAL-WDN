# Node Pressure Prediction by Aggregating Long-Range Information
Implementation for ICIOT'24 paper: Node Pressure Prediction by Aggregating Long-Range Information

## Overview
![overview](./imgs/architecture.pdf)
Predicting node pressure accurately is of paramount importance for the management of water distribution networks (WDNs).
Recent advances have highlighted the efficacy of graph neural networks(GNNs), tailored for data with inherent graph structures, 
in addressing this challenge. However, the performance of extant GNN-based approaches is constrained by their limited 
capacity to harness long-range dependencies within the network. 

To address this limitation, we introduce a novel long-range adaptive convolution network. Inspired by the graph kernel, 
our method possesses a broad receptive field, while the flexibility of information aggregation is enhanced through the attention
mechanism. Additionally, we incorporate residuals specifically engineered for WDN applications to further refine our prediction accuracy. Our comprehensive evaluations on three real-world WDN
datasets reveal that our method consistently surpasses existing benchmarks.

### Python environment setup with Conda
```shell
conda create -n gal_wdn python=3.10
# pytorch pyg
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install rdkit-pypi cython
pip install ogb
pip install pyarrow
pip install transformers
pip install configargparse
pip install loguru
pip install wandb
pip install nvidia-ml-py3
pip install tensorboardX
pip install sortedcontainers
pip install pyyaml
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# for wds
pip install epynet
pip install pint
pip install dask
pip install ray
pip install zarr
pip install wntr
pip install pyDOE2
```


## Datasets
The .inp files of the datasets can be found in the `water_networks/` folder. The generation of the datasets can refer to
https://github.com/BME-SmartLab/GraphConvWat