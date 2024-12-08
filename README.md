# Text-to-Speech with PyTorch

## About

This repository contains my implementation of HiFiGan for speech generation. This task is a part of DLA course.

See the task assignment [here]([https://github.com/markovka17/dla/tree/2024/hw1_asr](https://github.com/markovka17/dla/tree/2024/hw3_nv)).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Disclaimer

You probably don't want to use this repo for anything. Seriously, think twice. And read my report [here](https://wandb.ai/aavgustyonok/HiFiGan/reports/HiFiGan--VmlldzoxMDUxMzkwNw?accessToken=p3cv400ubv6dtw0idwqwn7c2x2ug3nzbs14c3gkev73m7zef6o2goea5mjjk5cqf) in order to understand that you don't want to do anything with this repo (except maybe for reading the code).

## How To Use

To reproduce training, run the following command:

```bash
python3 train.py -cn=baseline -datasets.train.data_dir=r"PATH_TO_WAV_DIR"
```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
