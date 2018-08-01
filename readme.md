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

## Citation
Please cite our work in your publications if it helps your research:

	@article{Li2017Learning,
	  Title={Learning Graph-Level Representation for Drug Discoveryk},
	  Journal={arXiv preprint arXiv:1709.03741},
	  Author={Junying Li, Deng Cai, Xiaofei He},
	  Year={2017},
	}
