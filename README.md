# DeepFD-pyTorch
This is a PyTorch implementation of DeepFD, used as a baseline in BiGAL.
Other than the unsupervised DBSCAN classifier used in the original paper, I also added a supervised 3-layer MLP as a classifier option. The whole embedding part is still always unsupervised.

#### Authors of this code package:
[Tong Zhao](https://github.com/zhao-tong) (tzhao2@nd.edu),
[Kaifeng Yu](https://github.com/kaifeng16) (ykf16@mails.tsinghua.edu.cn).

## Environment settings
- python==3.6.8
- pytorch==1.0.1.post2


## Basic Usage
Before running the model, first you need to create two folders: `results/` and `data/`.

**Data Inputs**

Required input data files are `graph_u2p` and `labels`, the paths need to be modified in `configs/file_paths.json`.

`graph_u2p` is the pickled adjacency matrix in `scipy.sparse.csr_matrix` format, where each none-zero entry stands for a edge.

`labels` is the pickled binary labels in `numpy.ndarray` format, where 1 stands for fraudulent user and 0 stands for benign user.

**Example Usage**

To run the unsupervised model on Cuda with the default GPU card:
```
python -m src.main --cuda 9 --dataSet [YourDataSet] --cls_method [dbscan or mlp]
```

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: weibo_s)
--name        The name of this run. (default: debug)
--cls_method  The classification method to be used. Choose between dbscan and mlp. (default: dbscan)
--epochs      Number of epochs. (default: 10)
--b_sz        Batch size. (default: 100)
--seed        Random seed. (default: 1234)
--hidden_size The size of hidden layer in encoder and decoder. (default: 128)
--emb_size    The size of the embeddings for each user. (default: 2)
--cuda        Which GPU card to use. -1 for CPU, 9 for default GPU, 0~3 for specific GPU. (default: -1)
```





