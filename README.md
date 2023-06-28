# NeuralSEP: A Neural Separation Algorithm for the Rounded Capacity Inequalities

This repository provides implemented codes for the paper, NeuralSEP. 
> Reference

## Python requirement
Clone project and create an environment with conda:
```
conda create -n neuralsep python=3.8
conda activate neuralsep

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

**Note** 
- If you use the different cuda version, please modify the URL for `torch-scatter` in `requirements.txt` before running it; see [here](https://pytorch-geometric.readthedocs.io/en/1.7.2/notes/installation.html).
- If you have any trouble with `dgl`, please refer [here](https://www.dgl.ai/pages/start.html)
- To run cutting plane embedded NeuralSEP, other installations are required.

## Usage
### Training
```
python train_coarsening_model.py 
```

### Evaluating with cutting plane methods
```
cd src/jl
julia experiment_with_random_instances.jl
```

- To run the autoregressive model, the function `learned_rounded_capacity_cuts` in `src/jl/cvrp_cutting.jl` needs to be modified.
- You can change pre-trained model directories in `julia_main.py' for each model.


## Others
### Julia requirement
- Julia >= 1.8.3
- JuMP
- Gurobi
- CPLEX
- TSPLIB
- CVRPLIB
- CVRPSEP
- Graphs
- PyCall
- Pickle

### CPLEX (1.2.10)
A license is required (the student license is free).

### Gurobi (10.0.1)
A license is required (the student license is free).
