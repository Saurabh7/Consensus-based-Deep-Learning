

This repository contains the code used in the paper:
[Consensus Based Vertically Partitioned Multi-layer Perceptrons for Edge Computing
](https://link.springer.com/chapter/10.1007/978-3-030-88942-5_20) published at International Conference on Discovery Science 2021.

Steps to run experiments:

1.  Peersim -> Used for generating graph

2.  Flask server -> Used for Data processing, Models, Gossip of network loss

The peersim configs should be present in `config` folder and the datasets in `data` folder in the base directory.

Then the flask server that hosts the models can be run as follows:

```
cd scripts
export FLASK_APP=flask_server.py
python -m flask run
```

In a separate terminal the peersim jar can be run as follows:

```
python run_jar.py
```
