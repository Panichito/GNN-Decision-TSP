## Overview
This project aims to replicate the research findings presented in the paper [Learning to Solve NP-Complete Problems - A Graph Neural Network for Decision TSP](https://arxiv.org/abs/1809.02721) using PyTorch <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png" width=15/>. The original paper demonstrates the effectiveness of Graph Neural Networks (GNNs) in solving the decision variant of the Traveling Salesperson Problem (TSP), a highly relevant NP-complete problem.


In contrast to the original work, which utilized the [Concorde TSP solver](https://www.math.uwaterloo.ca/tsp/concorde.html) to generate supervised data, this replication employs [Google OR-Tools](https://developers.google.com/optimization) <img src="https://pbs.twimg.com/media/DuoN35ZXgAAKzC_.jpg" width=120/>  for dataset generation. Leveraging Google OR-Tools offers advantages in terms of speed and efficiency, enabling faster experimentation and iteration in the research process.

Additionally, it's important to note that while this project aims to replicate the core research findings, it does not fully replicate the [original](https://github.com/machine-reasoning-ufrgs/TSP-GNN) project's code structure. Instead, it offers a *simplified version* that maintains the essence of the research while streamlining the implementation.

Furthermore, the original repository associated with the paper has not received updates since 2019 and is built upon TensorFlow  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" width=15/> version 1.x, now considered outdated. Consequently, direct utilization of their code is impractical. To address this limitation, the project provides an updated implementation in PyTorch. This updated version ensures both convenience and correctness while encouraging peer review for code interpretation and verification.

> These adjustments provide additional context regarding the outdated TensorFlow version and emphasize the importance of collaboration for ensuring code correctness and transparency.

To facilitate experimentation and exploration, the project includes small instances with supervised labels generated from Google OR-Tools. These instances allow for easy testing and adjustment of the training epoch loop, providing a hands-on approach to understanding the model's behavior.
