Consensus DL.

Steps:

1.  Peersim -> Generate graph

2.  Flask server -> Data processing, Models, Gossip

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
