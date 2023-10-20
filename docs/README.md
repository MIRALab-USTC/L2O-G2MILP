# G2MILP

## Publication
**A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability. NeurIPS 2023 (Spotlight).**

[Zijie Geng](https://miralab.ai/people/zijie-geng/), [Xijun Li](https://xijunlee.github.io/), [Jie Wang\*](https://miralab.ai/people/jie-wang/), Xiao Li, Yongdong Zhang, Feng Wu

MIRA Lab, University of Science and Technology of China / Noah's Ark Lab, Huawei

## Introduction
G2MILP is the first deep generative framework for mixed-integer linear programming (MILP) instances.
Specifically, it represents MILP instances as bipartite graphs, and iteratively corrupts and replaces parts of the original graphs to generate new ones.

G2MILP can learn to generate novel and realistic MILP instances without prior expert-designed formulations, while preserving the structures and computational hardness of real-world datasets, simultaneously.
Thus the generated instances can facilitate downstream tasks for enhancing MILP solvers under limited data availability.

## Model Architecture

![model architecture](./model.png#pic_center)

We represent MILP instances as weighted bipartite graphs, where variables and constraints are vertices, and non-zero coefficients are edges.
With this representation, we can use graph neural networks (GNNs) to effectively capture essential features of MILP instances.
In this way, we recast the original task as a graph generation problem.

We propose a masked variational autoencoder (VAE) paradigm inspired by masked language models (MLM) and VAE theories.
The proposed paradigm iteratively corrupts and replaces parts of the original graphs using sampled latent vectors.
This approach allows for controlling the degree to which we change the original instances, thus balancing the novelty and the preservation of structures and hardness of the generated instances.

To implement the complicated generation steps, we design a decoder consisting of four modules that work cooperatively to determine multiple components of new instances, involing both structure and and numerical prediction tasks simultaneously.