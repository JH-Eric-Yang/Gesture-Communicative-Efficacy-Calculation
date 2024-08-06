import numpy as np
import pandas as pd
import os
from gensim.models import KeyedVectors
from gensim.test.utils import  datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


### input the model path
glove_file = "glove_840B_300d.txt"
### convert glove model to word2vec model 
### check if the file is already converted
if os.path.exists('glove_word2_vec.txt') == False:

    output_file = "glove_word2_vec.txt"

    glove2word2vec(glove_file, output_file)

    model = KeyedVectors.load_word2vec_format(output_file)
else:
    word2vec_output_file = 'glove_word2_vec.txt'
### load model
    model = KeyedVectors.load_word2vec_format(word2vec_output_file)

##
def calculating_similarity(data_path,tar_word_col_name, response_col_name):
    result = []
    data = pd.read_csv(data_path)
    data[tar_word_col_name] = data[tar_word_col_name].str.lower()
    for i in range(len(data)):
        target = data[tar_word_col_name][i]
        response = data[response_col_name][i]
        print(target, response)
        try:
            similarity = model.similarity(target,response)
            result.append(similarity)
            print(str(i) + ". Successfully compared " + str(target) + " and " + str(response) +
                  " " + str(similarity))
        except Exception as e:
            similarity = "NA"
            result.append(similarity) 
            print(str(i) + ". " + str(target) + " and " + str(response) +
                  " not found")
            print(e)
    return result



def generate_semantic_similarity(output_path,data_path,tar_word_col_name, response_col_name):
    final = calculating_similarity(data_path,tar_word_col_name, response_col_name)
    data = pd.read_csv(data_path)
    data["Similarity"] = final
    data.to_csv(output_path)
    print("File saved successfully")

if __name__ == "__main__":
    ### input output path
    output_path = "data_open_end_sgen1_final_semantic_similarity_new.csv"
    ### input the original data contains the target word and the reposne at each row
    data_path = "data_open_end_sgen1_final.csv"
    ### input the column name for the target word column
    tar_word_col_name = "Target_word"
    ### input the column name for the response column
    response_col_name = "response"

    generate_semantic_similarity(output_path,data_path,tar_word_col_name, response_col_name)
