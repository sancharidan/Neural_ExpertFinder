import numpy as np
import pandas as pd
import json

import argparse
from scholarly import scholarly
import random

###-----Parameters - input from user----###
parser = argparse.ArgumentParser()
parser.add_argument('--INPUT_FILENAME', default = 'sample_input', help = 'Specify name of input json in Data/input/ folder')
parser.add_argument('--RESEARCH_AREA_FIELD',default = None, help = 'Specify field in input json which contains list of \
    research areas for each expert')
# parser.add_argument('--OUTPUT_FILEPATH',  default = "./Data/processed/sample_output.json", help = 'Output filepath')

# parse input arguments
inP = parser.parse_args()
INPUT_FILENAME = inP.INPUT_FILENAME # path to input json
RESEARCH_AREA_FIELD = inP.RESEARCH_AREA_FIELD # research area field
# OUTPUT_FILEPATH = inP.OUTPUT_FILEPATH # path to output file for storing scraped data in csv format

# read input json and convert to df
with open(INPUT_FILENAME,'r') as f:
    inp_json = json.load(f)
input_df = pd.DataFrame(inp_json)

# check if data has already been scraped
print ("\nScraping Google Scholar for {} experts.\n".format(len(input_df)))

gs_data = []
def scrape_gs(row):
    try:    
        author = row['name']
        org = row['organization']
        print ('Getting data for ', author)
#         school = row['school']
        search_query = scholarly.search_author(author + ', ' + org)
        author_data = scholarly.fill(next(search_query), sections = ['publications'])
        publications = [pub['bib']['title'] for pub in author_data['publications']]
        titles = []
        abstracts = []
        for i in range(len(publications)):
            pub = scholarly.fill(author_data['publications'][i])
            if 'title' in pub['bib'].keys():
                titles.append(pub['bib']['title'])
            if 'abstract' in pub['bib'].keys():
                abstracts.append(pub['bib']['abstract'])
            elif 'abstract' not in pub['bib'].keys():
                abstracts.append('')
            
#         
        interests = author_data['interests']
        affil = author_data['affiliation']
        gs_data.append({'faculty':author,'publication_titles':titles,'abstracts':abstracts,'interests':interests,'affiliation':affil})
    except StopIteration:
        print ("\tNo data for",author + ', ' + org)
        try:
            search_query = scholarly.search_author(author)
            author_data = scholarly.fill(next(search_query), sections = ['publications'])
            publications = [pub['bib']['title'] for pub in author_data['publications']]
            titles = []
            abstracts = []
            for i in range(len(publications)):
                pub = scholarly.fill(author_data['publications'][i])
                if 'title' in pub['bib'].keys():
                    titles.append(pub['bib']['title'])
                if 'abstract' in pub['bib'].keys():
                    abstracts.append(pub['bib']['abstract'])
                elif 'abstract' not in pub['bib'].keys():
                    abstracts.append('')
            
            interests = author_data['interests']
            affil = author_data['affiliation']
            gs_data.append({'faculty':author,'publication_titles':titles,'abstracts':abstracts,'interests':interests,'affiliation':affil})
        except StopIteration:
            print ("\t\tNo data for",author)
            pass
    return 

input_df.apply(scrape_gs,axis = 1)

pub_df = pd.DataFrame()
abstracts_df = pd.DataFrame()
gs_int_df = pd.DataFrame()
affil_df = pd.DataFrame()
for scholar in gs_data:
    if 'publications' in scholar.keys():
        tmp_pub = pd.DataFrame({'faculty':scholar['faculty'],'publications':[[l[0] for l in scholar['publications']]]}, index = [0])
        pub_df = pd.concat([pub_df, tmp_pub])
        
        tmp_abs = pd.DataFrame({'faculty':scholar['faculty'],'abstracts':[[l[1] for l in scholar['publications']]]}, index = [0])
        abstracts_df = pd.concat([abstracts_df, tmp_abs])
        print (scholar['faculty'], len(tmp_pub[['publications']].values[0][0]) == len(tmp_abs[['abstracts']].values[0][0]))
   
    tmp_gs_int = pd.DataFrame({'faculty':scholar['faculty'],'gs_interests':[scholar['interests']]}, index = [0])
    gs_int_df = pd.concat([gs_int_df,tmp_gs_int])
    
    tmp_affil = pd.DataFrame({'faculty':scholar['faculty'],'affiliation':[scholar['affiliation']]}, index = [0])
    affil_df = pd.concat([affil_df,tmp_affil])
    
pub_df.explode('publications').reset_index().drop(['index'],axis=1).to_csv('Data/processed/' + INPUT_FILENAME + '_pub.csv',index=False)

abstracts_df.explode('abstracts').reset_index().drop(['index'],axis=1).to_csv('Data/processed/' + INPUT_FILENAME + '_abstracts.csv',index=False)

gs_int_df.explode('gs_interests').reset_index().drop(['index'],axis=1).to_csv('Data/processed/' + INPUT_FILENAME + '_research_areas.csv',index=False)

affil_df.explode('affiliation').reset_index().drop(['index'],axis=1).to_csv('Data/processed/' + INPUT_FILENAME + '_affiliation.csv',index=False)

if RESEARCH_AREA_FIELD:
    input_ra_df = input_df[['name',RESEARCH_AREA_FIELD]]
    input_ra_df.columns = ['faculty',RESEARCH_AREA_FIELD]
    input_ra_df.explode(RESEARCH_AREA_FIELD).reset_index().drop(['index'],axis=1).to_csv('Data/processed/' + INPUT_FILENAME + '_research_areas.csv',index=False)
    

print ('\nScraping Complete, Check Data/processed/ folder for scraped datasets!\n')
print ('\nExperts with no Google Scholar data -->\n\t',[exp for exp in input_df.name.unique() if exp not in pub_df.faculty.unique()])