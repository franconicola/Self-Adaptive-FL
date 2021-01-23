# Towards a Self-Adaptive Architecture for Federated-Learning of Industrial Automation Systems

### Abstract 

Emerging Industry 4.0 architectures deploy data-driven applications and artificial intelligence services across multiple locations under varying ownership, and require specific data protection and privacy considerations to not expose confidential data to third parties.
For this reason, federated learning provides a framework for optimizing machine learning models in single manufacturing facilities without requiring access to training data. 
In this paper, we propose a self-adaptive architecture for federated learning of industrial automation systems. 
Our approach considers entities on three different levels: centralized cloud server, smart factory, and smart industrial device. 
To achieve global model optimization and reduce communication cycles, each factory internally trains the model in a self-adaptive manner and sends it to the centralized cloud server for global aggregation. 
We model the problem as a multi-assignment optimization by dividing the dataset into a number of subsets equal to the number of devices. Each device chooses the right subset to optimize the model at each local iteration. 
A first analysis shows the convergence property of the algorithm on a training dataset with different numbers of factories and devices. These results demonstrate higher model accuracy than the federated averaging approach for the same number of communication cycles.
