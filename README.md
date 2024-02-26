# RL for Operational Research

A framework for solving Operational Research problems with tabular RL algorithms.

By Timothé Boulet, Pierre Prévot-Helloco


# Installation

Clone the repository, create a venv (advised), and install the requirements:

```bash
git clone git@github.com:tboulet/RL-for-Operational-Research.git
cd RL-for-Operational-Research
python -m venv venv
source venv/bin/activate  # on linux
venv\Scripts\activate  # on windows
pip install -r requirements.txt
```


# Run the code
 
For training your algorithms on a certain env, run the following command:

```bash
python run.py algo=<algo tag> env=<env tag>
```

For example, to train the random algorithm on the toy env:

```bash
python run.py algo=random env=toy
```

### Algorithms
The algo tag should correspond to a configuration in ``configs/algo/`` where you can specify the algo and its hyperparameters. 

Currently, the following algorithms are available:
 - `random` : Random policy

### Environments

The env tag should correspond to a configuration in ``configs/env/`` where you can specify the env and its hyperparameters.

Currently the following envs are implemented :
- `toy` : A simple toy environment


We use Hydra as our config system. The config folder is `./configs/`. You can modify the config (logging, metrics, number of training episodes) from the `default_config.yaml` file. You can also create your own config file and specify it with the `--config-name` argument :

```bash
python run.py algo=random env=toy --config-name=my_config_name
```

Advice : create an alias for the command above this.