## Data
The graphs stem from the [SNAP](https://snap.stanford.edu/data/) repository. The format of graphs is a weighted edgelist (with weighted cascade weights) in .inf, accompanied by an attribute file. The benchmark codes for influence maximization are adapted from [IMM](https://github.com/snowgy/Influence_Maximization/wiki/Home/), [DegreeDiscount](https://github.com/nd7141/influence-maximization/blob/master/IC/degreeDiscount.py), [PMIA](https://github.com/nd7141/influence-maximization/blob/master/IC/ArbitraryP/PMIA.py), [FINDER](https://github.com/FFrankyy/FINDER) and [DeepIS](https://github.com/xiawenwen49/DeepIS), while for influence estimation we develop a python version of [DMP](https://github.com/mateuszwilinski/dynamic-message-passing) inside "diffuse.py".
All benchmark codes can be found in the respective folder. Unzip the data.zip in a "data" folder in the current folder.


## Requirements
To run this code you will need the following in python 3.5.2:
* [pytorch 1.5.1](https://pytorch.org/)
* [networkx 1.11](https://networkx.github.io/) 
* [sklearn](https://scikit-learn.org/stable/) 
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)


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





