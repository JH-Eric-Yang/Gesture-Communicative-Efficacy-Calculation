import pandas as pd
import numpy as np

# Function to calculate entropy
def calculate_entropy(series):
    probabilities = series.value_counts(normalize=True)
    print(probabilities)
    entropy = -np.sum(probabilities*np.log2(probabilities))
    return entropy

# Group the data by 'variable_name' and apply the entropy function
# Group the data by 'variable_name' and apply the entropy function

def generate_entropy_file(data_path,output_path,geture_token_col_name,response_col_name):
    data = pd.read_csv(data_path)
    data[geture_token_col_name] = data[geture_token_col_name].str.lower()
    entropy_values = data.groupby(geture_token_col_name)[response_col_name].apply(calculate_entropy)
    entropy_values = entropy_values.reset_index()
    entropy_values.columns = [geture_token_col_name, 'entropy']
    entropy_values.to_csv(output_path, index=False)
    print("File saved successfully")

if __name__ == "__main__":
    ### input output path
    output_path = "data_open_end_sgen1_final_entropy.csv"
    ### input the original data contains the target word and the reposne at each row
    data_path = "data_open_end_sgen1_final.csv"
    ### input the column name for the columns contating the categorization of the entropy calculation (e.g., gesture token)
    geture_token_col_name = "question"
    ### input the column name for the response column
    response_col_name = "response"

    generate_entropy_file(data_path,output_path,geture_token_col_name,response_col_name)