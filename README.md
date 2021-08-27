# Towards a Self-Adaptive Architecture for Federated-Learning of Industrial Automation Systems

### Abstract 

Emerging Industry 4.0 architectures deploy data-driven applications and artificial intelligence services across multiple locations under varying ownership, and require specific data protection and privacy considerations to not expose confidential data to third parties. For this reason, federated learning provides a framework for optimizing machine learning models in single manufacturing facilities without requiring access to training data. In this paper, we propose a self-adaptive architecture for federated learning of industrial automation systems. Our approach considers the involved entities on the different levels of abstraction of an industrial ecosystem. To achieve the goal of global model optimization and reduction of communication cycles, each factory internally trains the model in a self-adaptive manner and sends it to the centralized cloud server for global aggregation. We model a multi-assignment optimization problem by dividing the dataset into a number of subsets equal to the number of devices. Each device chooses the right subset to optimize the model at each local iteration. Our initial analysis shows the convergence property of the algorithm on a training dataset with different numbers of factories and devices. Moreover, these results demonstrate higher model accuracy with our self-adaptive architecture than the federated averaging approach for the same number of communication cycles.

### Publisher

**Published in:** [2021 International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)](https://ieeexplore.ieee.org/abstract/document/9462039)


Citation
--------
If you are using this repository please use the following citation to reference this work:
```
@INPROCEEDINGS{9462039,
  author={Franco, Nicola and Van, Hoai My and Dreiser, Marc and Weiss, Gereon},
  booktitle={2021 International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)}, 
  title={Towards a Self-Adaptive Architecture for Federated Learning of Industrial Automation Systems}, 
  year={2021},
  volume={},
  number={},
  pages={210-216},
  doi={10.1109/SEAMS51251.2021.00035}}
```

License and Copyright
---------------------

* Copyright (c) 2021 [Fraunhofer Institute for Cognitive Systems IKS](https://www.iks.fraunhofer.de/en.html)
* Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

