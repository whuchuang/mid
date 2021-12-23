# MID

On Exploring Node-feature and Graph-structure Diversities for Node Drop Graph Pooling (under review)

![](https://github.com/whuchuang/mid/blob/main/model.png)



## Requirements


```
python conda create -c conda-forge -n my-rdkit-env rdkit python =3.7

pip3 install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html

pip install ogb

pip install transformers
```

Note:
This code repository is heavily built on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which is a Geometric Deep Learning Extension Library for PyTorch.


## Run

Just execuate the following command to reproduce the results in the mauscript :
```python
sh classification_TU.sh
```

