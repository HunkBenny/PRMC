# PRMC
Repository tree:
<pre>
│   LSTM_PRMC.ipynb
│   README.md
│   XGBOOST_PRMC.ipynb
│
├───data
│   └───gold
│           test.csv
│           train.csv
│           validation.csv
│
├───emp
│   │   __init__.py
│   │
│   ├───losses
│   │   └───prmc
│   │           keras.py
│   │           xgboost.py
│   │
│   ├───metrics
│   │       maintenance.py
│   │       ranking.py
│   │
│   └───models
│       │   __init__.py
│       │
│       └───prmc
│               keras.py
│
├───packages
│   └───hypopt
│           model_selection.py
│           version.py
│           __init__.py
│
└───preprocessing
        keras.py
</pre>

## packages:
This folder contains code from the hypopt-package that was used during the hyper parameter search.
**Hypopt GitHub:** https://github.com/cgnorthcutt/hypopt

## notebooks:
The notebooks 'LSTM_PRMC.ipynb' and 'XGBOOST_PRMC.ipynb' contain code on training a model with the PRMC as objective.
In both notebooks, after the "ANALYSIS" cell, an analysis is done with use of the PRMC-metric.



#### instance-based PRMC for XGBoost was loosely based on:
https://github.com/bram-janssens/B2Boost
