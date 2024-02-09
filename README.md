# G2MILP: Learning to Generate Mixed-Integer Linear Programming (MILP) Instances

This is the code for **G2MILP**, a deep learning-based mixed-integer linear programming (MILP) instance generator. 

Page: [https://miralab-ustc.github.io/L2O-G2MILP/](https://miralab-ustc.github.io/L2O-G2MILP/)

## Publications
**"A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability"**. *Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, Feng Wu.* NeurIPS 2023 (Spotlight). [[paper](https://openreview.net/pdf?id=AiEipk1X0c)]

**"G2MILP: Learning to Generate Mixed-Integer Linear Programming Instances for MILP Solvers"**. Jie Wang, Zijie Geng, Xijun Li, Jianye Hao, Yongdong Zhang, Feng Wu. [[paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.24566554.v1)]

## Environment
- Python environment
    - python 3.7
    - pytorch 1.13
    - torch-geometric 2.3
    - ecole 0.7.3
    - pyscipopt 3.5.0
    - community 0.16
    - networkx
    - pandas
    - tensorboardX
    - gurobipy

- MILP Solver
    - [Gurobi](https://www.gurobi.com/) 10.0.1. Academic License.

- Hydra
    - [Hydra](https://hydra.cc/docs/intro/) for managing hyperparameters and experiments.


In order to build the environment, you can follow commands in `scripts/environment.sh`.

Or alternatively, to build the environment from a file,
```
conda env create -f scripts/environment.yml
```

## Usage

Go to the root directory `L2O-G2MILP`. Put the datasets under the `./data` directory. Below is an illustration of the directory structure.
```
L2O-G2MILP
├── conf
├── data
│   ├── mik
│   │   ├── train/
│   │   └── test/
│   ├── mis
│   │   ├── train/
│   │   └── test/
│   └── setcover
│       ├── train/
│       └── test/
├── scripts/
├── src/
├── src_hard/
├── README.md
├── benchmark.py
├── generate.py
├── preprocess.py
├── train-hard.py
└── train.py
```

The hyperparameter configurations are in `./conf/`.
The commands to run for all datasets are in `./scripts/`.
The main part of the code is in `./src/`.
The workflow of G2MILP (using MIS as an example) is as following.

### 1. Preprocessing

To preprocess a dataset,
```
python preprocess.py dataset=mis num_workers=10
```
This will produce graph data for instances and the statistics of the dataset to be used for training. The preprocessed results are saved under `./preprocess/mis/`. 

### 2. Training **G2MILP**

To train G2MILP with default parameters,
```
python train.py dataset=mis cuda=0 num_workers=10 job_name=mis-default
```
The training log is saved under `TRAIN DIR=./outputs/train/${DATE}/${TIME}-${JOB NAME}/`. The model ckpts are saved under `${TRAIN DIR}/model/`. The generated instances and benchmarking results are saved under `${TRAIN DIR}/eta-${eta}/`.

### 3. Generating new instances

To generate new instances with a trained model,
```
python generate.py dataset=mis \
    generator.mask_ratio=0.01 \
    cuda=0 num_workers=10 \
    dir=${TRAIN DIR}
```
The generated instances and benchmarking results are saved under `${TRAIN DIR}/generate/${DATE}/${TIME}`.

### 4. Generating hard instances
To generate hard instances,
```
python train-hard.py dataset=mis \
    cuda=0 num_workers=10 \
    pretrained_model_path=${PRETRAIN PATH}
```
The `${PRETRAIN PATH}` is the path to the pretrained model.

## Citation
If you find this code useful, please consider citing the following papers.
```
@inproceedings{geng2023deep,
  title={A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability},
  author={Geng, Zijie and Li, Xijun and Wang, Jie and Li, Xiao and Zhang, Yongdong and Wu, Feng},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}

@article{wang2023g2milp,
  title={G2MILP: Learning to Generate Mixed-Integer Linear Programming Instances for MILP Solvers},
  author={Wang, Jie and Geng, Zijie and Li, Xijun and Hao, Jianye and Zhang, Yongdong and Wu, Feng},
  journal={Authorea Preprints},
  year={2023},
  publisher={Authorea}
}
```
