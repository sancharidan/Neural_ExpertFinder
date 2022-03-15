# Neural Expert Finder
This repository contains the code for Neural Expert Finder - a transformer based framework that fine-tunes pre-trained language models using data pertaining to academic experts' publications and research interests in order to return relevant experts for query research topics. 

In order to train a model and use this repository for custom data, please install anaconda or miniconda and follow the steps below:
- Create a new python environment
```
conda create -n nef python=3.7
```
- Activate the environment
```
conda activate nef
```
- Clone the repository
```
git clone https://github.com/sancharidan/Neural_ExpertFinder.git
```
- Store the input json with expert data in `./Data/input/`
```
[
    {"name": <expert_name_1>, "organization": <org_name>, ... },
    {"name": <expert_name_2>, "organization": <org_name>, ... },
    ...
]
```
