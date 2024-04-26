# Time-To-Market, Chip Agility Score (CAS), and Cost Modeling

In this repo are the relevant datasets used in our time-to-market model and chip agility score calculations in our [Supply Chain Aware Computer Architecture](https://dl.acm.org/doi/10.1145/3579371.3589052) paper presented at ISCA 2023. Please reach out if you have any questions.

### How to use
There are two relevant files:
* `supply_chain_model.py`: python file that contains the relevant parameters for our time-to-market model and total cost of ownership modeling and methods for calculating time-to-market, chip agility score, and chip creation cost. This file is designed to be imported.
* `ttm_cas_example.ipynb`: Jupyter notebook that shows how use time-to-market, chip agility score, and cost modeling.

To get started, clone the repo, create a new Conda environment, and start the Jupyter notebook. You'll also need to have the `conda-forge` channel enabled.
```
$ git clone git@github.com:PrincetonUniversity/ttm-cas.git
$ cd ttm-cas
$ conda create --name supply_chain --file requirements.txt
$ conda activate supply_chain
$ (supply_chain) jupyter notebook
```

### How to cite
If you use our time-to-market modeling, chip agility score, and/or our datasets, please cite our ISCA 2023 paper and send us a copy. 
```
@inproceedings{ning-supply-chain-isca2023,
author = {Ning, August and Tziantzioulis, Georgios and Wentzlaff, David},
title = {Supply Chain Aware Computer Architecture},
year = {2023},
isbn = {9798400700958},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3579371.3589052},
doi = {10.1145/3579371.3589052},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
articleno = {17},
numpages = {15},
keywords = {economics, chip shortage, semiconductor supply chain, modeling},
location = {Orlando, FL, USA},
series = {ISCA '23}
}
```

Please check out the paper and comments within `supply_chain_model.py` for sources from where we derive our parameters from.
