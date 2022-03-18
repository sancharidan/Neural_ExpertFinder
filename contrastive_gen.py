import numpy as np
import pandas as pd

import argparse
import random

###-----Parameters - input from user----###
parser = argparse.ArgumentParser()
parser.add_argument('--PUB_FILE', default = './Data/processed/sample_input_pub.csv', help = 'Specify path to publications df')
parser.add_argument('--RESEARCH_AREA_FILE',default = './Data/processed/sample_input_research_areas.csv', help = 'Specify path to research area df')
# parser.add_argument('--OUTPUT_FILEPATH',  default = "./Data/processed/sample_input_train.csv", help = 'Output filepath .csv')

# parse input arguments
inP = parser.parse_args()
PUB_FILE = inP.PUB_FILE # path to input json
RESEARCH_AREA_FILE = inP.RESEARCH_AREA_FILE # research area field
# OUTPUT_FILEPATH = inP.OUTPUT_FILEPATH # path to output file for storing scraped data in csv format

####--------------------Prepare contrastive data for training---------------------------#####

# function for preparing contrastive dataset
def prep_data(df, column, relation, factor = 1):
    data = []
    all_samples = df[column].unique()
    for fac in df.faculty.unique():
        pos_samples = df[df['faculty'] == fac][column].unique()
        neg_samples = [sample for sample in all_samples if sample not in pos_samples]
        neg_samples = random.sample(neg_samples,min(factor*len(pos_samples),len(neg_samples)))
        
        for sample in pos_samples:
            data.append({'head' : fac, 'relation' : relation, 'tail' : sample, 'label' : 1})
        for sample in neg_samples:
            data.append({'head' : fac, 'relation' : relation, 'tail' : sample, 'label' : 0})
    return pd.DataFrame(data)

pub_df = pd.read_csv(PUB_FILE,index_col=False)
ra_df = pd.read_csv(RESEARCH_AREA_FILE,index_col=False)
ra_df.columns = ['faculty','research_areas']
new_pub = prep_data(pub_df.explode('publications').reset_index().drop(['index'],axis=1), 'publications', 'has authored', factor = 1)
new_ra = prep_data(ra_df.explode('research_areas').reset_index().drop(['index'],axis=1), 'research_areas', 'researches in', factor = 1)

fin_train_data = pd.concat([new_pub, new_ra])
print ('Training Data Statistics -->\n\t# has authored positive samples = {}\n\t# has authored negative samples = {}\
        \n\t# researches in positive samples = {}\n\t# researches in positive samples = {}'\
        .format(new_pub.label.sum(), len(new_pub) - new_pub.label.sum(), new_ra.label.sum(), len(new_ra) - new_ra.label.sum()))

fin_train_data.to_csv('Data/processed/train.csv',index=False)