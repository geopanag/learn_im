# Maximizing Influence with Graph Neural Networks

The code to reproduce the analysis for ["Maximizing Influence with Graph Neural Networks"]([https://hal.science/hal-04601553/document](https://arxiv.org/pdf/2108.04623)).

Start by installing the requirements.
```bash
pip install -r requirements.txt
```

## Data
The graphs can be found in [SNAP](https://snap.stanford.edu/data/) repository. The format of graphs is a weighted edgelist (with weighted cascade weights) in .inf, accompanied by an attribute file. The benchmark codes for influence maximization are adapted from [IMM](https://github.com/snowgy/Influence_Maximization/wiki/Home/), [DegreeDiscount](https://github.com/nd7141/influence-maximization/blob/master/IC/degreeDiscount.py), [PMIA](https://github.com/nd7141/influence-maximization/blob/master/IC/ArbitraryP/PMIA.py), [FINDER](https://github.com/FFrankyy/FINDER) and [DeepIS](https://github.com/xiawenwen49/DeepIS), while for influence estimation we develop a python version of [DMP](https://github.com/mateuszwilinski/dynamic-message-passing) inside "diffuse.py".
All benchmark codes can be found in the respective folder. Unzip the data.zip in the "data" folder in the current folder.


## Code
The following scripts use the default parameters mentioned in the paper.

1.Influence estimation and error using the GLIE stored model 
```bash
python influence_predictions.py
```

2.Influence maximization (20 and 100 seeds) using the stored model with Celf-glie and evaluation of the seeds. Note that evaluation can take more then 3 hours for the large datasets.

```bash
python celf_glie.py
```

3.Influence maximization using the stored GNN model and the stored Grim model.

```bash
python grim.py

```

4.Influence maximization using the stored GNN model and the Pun model.

```bash
python pun.py
```


5.Train GNN on the negative samples, using the provided "influence_train_set.csv" constructed as discribed in section 4.1 of the paper.

```bash
python glie_train.py
```

6.Train Grim on the 50 graphs in "dql_graphs" as described in section 4.2 of the paper.

```bash
python train_dqn.py
```

7.The scripts in the preprocessing folder are required to create the "influence_train_set.csv". 





