# Defending a Federated Learning system


<iframe width="560" height="315" src="https://www.youtube.com/embed/0Uj5Kv4XZZA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>




This repository contains the code that implements a **Federated Learning (FL)** system (using **Flower** and **PyTorch**), two types of common attacks perfomed on a FL system, that are **model poisoning** and **data poisoning attacks**, and two corresponding defense mechanisms: **gradient checking** and **client reputation**.

Additionally, it also contains an implementation of **Homomorphic Encryption**, used to defend the system from **reverse engineering** attacks.

To reproduce the conda environment used for the development and run the system execute the following commands:
- **conda env create -f environment.yml**
- **conda activate fl_security**
- **./run.sh**

Otherwise, to just run the system execute the following command:
- **./run.sh**


