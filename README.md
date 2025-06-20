# Codes for "Temporal Blocks with Memory Replay for Dynamic Graph Representation Learning"

### Datasets

The statistics of datasets are listed as follows:

| Datasets  |      Domains      | # Nodes | # Edges | # Unique Edges | Time Granularity |    Duration     |
|:---------:|:-----------------:|:-------:|:-------:|:--------------:|:----------------:|:---------------:|
| Infectious | Contact Networks |   410   | 17,298  |     2,765      |  Unix timestamp  |     8 hours     |
|   Haggle   | Contact Networks |   274   | 28,244  |     2,899      |  Unix timestamp  |     4 days      |
|   Enron    | Contact Networks |   184   | 125,235 |     3,125      |  Unix timestamp  |     3 years     |
|   BITotc   | Bitcoin Networks |  4,863  | 28,473  |     28,473     |  Unix timestamp  |     7 years     |
|  BITalpha  | Bitcoin Networks |  3,219  | 19,364  |     19,364     |  Unix timestamp  |     7 years     |
|  USLegis   |Politics Networks |   225   | 60,396  |     26,423     |    Congresses    |  12 congresses  |


Processed datasets can be downloaded [here](https://pan.baidu.com/s/1PjQDAzl9cO68l5_PnsshfA?pwd=qur2). After downloading, place the downloaded datasets under the "/processed_data" directory.

### Model Training

You can train the model by running the following commands as an example:

```python train.py --dataset_name Infectious --model_name TBD --negative_sample_strategy random --num_runs 5 --gpu 0 --load_best_configs```

### Model Evaluation

After training the model, you can evaluate the model by running the following command as an example:

```python evaluate.py --dataset_name Infectious --model_name TBD --negative_sample_strategy random --num_runs 5 --gpu 0 --load_best_configs```

### Parameters

1. `dataset_name` can be chosen from:
   - Infectious
   - Haggle
   - enron
   - bitotc
   - bitcoin
   - USLegis

2. `negative_sample_strategy` can be chosen from:
   - random
   - historical
   - inductive

### Enveriments
- Python 3.9.21
- PyTorch 2.4.0
- pandas 2.2.3
- numpy 2.0.2
- dgl 2.4.0

### Acknowledgments
We are grateful to the authors of TGAT, TCL, DyRep, EdgeBank, GraphMixer and teneNCE for making their project codes publicly available.

[//]: # (### Citation)

[//]: # (Please consider citing our paper when using this project.)