# Neural Expert Finder
This repository contains the code for Neural Expert Finder - a transformer based framework that fine-tunes pre-trained language models using data pertaining to academic experts' publications and research interests in order to return relevant experts for query research topics. 

There are two parts to the usage of this repository:
1. One can upload their custom expert data with bare-minimum information like experts' names and their respective affiliated academic organizations and then run the provided scripts to scrape Google Scholar using the open-sourced `scholarly` library, generate the contrastive training dataset and fine-tune a pre-trained language model using the detailed jupyter notebook guide. 
2. A fine-tuned language model built on the SMU-SCIS dataset and stored on the huggingface model hub has been used to create a demo of a streamlit application for expert finding, this application can be played with as is or modified to cater to a custom model trained on custom dataset as mentioned in point 1 above. 

### Data Pre-processing and Model Training
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
- Store the input json with expert data in `./Data/input/`. The format of the json file should be as below: a list of dictionaries, with each dictionary having fields `name` and `organization` mandatorily, and other fields like `research_areas` etc. optionally. A sample file named `sample_input.json` is included in the `./Data/input` folder for reference.
```
[
    {"name": <expert_name_1>, "organization": <org_name>, ... },
    {"name": <expert_name_2>, "organization": <org_name>, ... },
    ...
]
```
- Install required libraries and packages with the `nef` environment
``` 
pip install -r requirements.txt
```
- If you want to scrape Google Scholar for publications, you have to run the `scholarly_scrape.py` file. Below command generates intermediate files from the scraped data and stores it in `./Data/processed/` folder as `.csv.` files with `INPUT_FILENAME` as the prefix for the filenames. 
```
python scholarly_scrape.py --INPUT_FILENAME <name of input json file in ./Data/input> --RESEARCH_AREA_FIELD <field that stores list of research areas of experts in json, default is None>
```
- Next step is contrastive training data generation using `contrastive_gen.py` which takes as input the intermediate publications file and research areas file generated in previous step. Note that if there is no research areas field in input json, the research areas file is generated from the experts' interests listed on Google Scholar. This script generates a `train.csv` file and stores it in the `./Data/processed/` folder.
```
python contrastive_gen.py --PUB_FILE <specify path to publications file in ./Data/processed/ folder> --RESEARCH_AREA_FILE <specify path to research areas file in ./Data/processed/ folder>
```
- Model Training (fine-tuning) can be done in the [NEF - SciBERT Finetuning.ipynb](https://github.com/sancharidan/Neural_ExpertFinder/blob/master/NEF%20-%20SciBERT%20Finetuning.ipynb) notebook. The model input, model type and model parameters can be set in the notebook prior to fine-tuning. Support for Cuda and GPU is also built-in.

### Streamlit Demo App
