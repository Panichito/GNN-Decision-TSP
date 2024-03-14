## Overview
This project aims to replicate the research findings presented in the paper [Learning to Solve NP-Complete Problems - A Graph Neural Network for Decision TSP](https://arxiv.org/abs/1809.02721) using PyTorch. The original paper demonstrates the effectiveness of Graph Neural Networks (GNNs) in solving the decision variant of the Traveling Salesperson Problem (TSP), a highly relevant NP-complete problem.

In contrast to the original work, which utilized the [Concorde TSP solver](https://www.math.uwaterloo.ca/tsp/concorde.html) to generate supervised data, this replication employs [Google OR-Tools](https://developers.google.com/optimization) for dataset generation. Leveraging Google OR-Tools offers advantages in terms of speed and efficiency, enabling faster experimentation and iteration in the research process.

Additionally, the [original](https://github.com/machine-reasoning-ufrgs/TSP-GNN) repository associated with the paper has not received updates for over 6 years and is built upon TensorFlow version 1.x, now considered outdated. Consequently, direct utilization of their code is impractical. To address this limitation, the project provides an updated implementation in PyTorch. This updated version ensures both convenience and correctness while encouraging peer review for code interpretation and verification.

*These adjustments provide additional context regarding the outdated TensorFlow version and emphasize the importance of collaboration for ensuring code correctness and transparency.*
