# Learning Graph-Level Representation for Drug Discovery
Paper Link: [Learning Graph-Level Representation for Drug Discovery](https://arxiv.org/abs/1709.03741)


## Requirements
- Install [DeepChem(july2017)](https://github.com/deepchem/deepchem/tree/july2017)

## Usage
1.Clone the repository

	git clone https://github.com/microljy/graph_level_drug_discovery.git

2.Training 

	python train.py --gpu 0 --dataset pcba

Our ```train.py``` only supports 6 datasets in MoleculeNet, including Tox21, ToxCast, HIV, MUV, PCBA, SAMPL.
## Result
Database and baseline: [MoleculeNet](https://arxiv.org/abs/1703.00564)

|Dataset  |Split Method|Train|Valid|Test |
|---------|------------|-----|-----|-----|
|Tox21    |Index       |0.965|0.839|0.848|
|Tox21    |Random      |0.964|0.842|0.854|
|Tox21    |Scaffold    |0.971|0.788|0.759|
|ToxCast  |Index       |0.927|0.747|0.734|
|ToxCast  |Random      |0.924|0.746|0.768|
|ToxCast  |Scaffold    |0.929|0.696|0.657|
|PCBA     |Index       |0.904|0.869|0.864|
|PCBA     |Random      |0.899|0.863|0.867|
|PCBA     |Scaffold    |0.907|0.847|0.845|

## Citation
Please cite our work in your publications if it helps your research:

	@article{Li2017Learning,
	  Title={Learning Graph-Level Representation for Drug Discoveryk},
	  Journal={arXiv preprint arXiv:1709.03741},
	  Author={Junying Li, Deng Cai, Xiaofei He},
	  Year={2017},
	}
