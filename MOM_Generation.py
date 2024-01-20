import json
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import textstat
from rouge_metric import Rouge
from sentence_transformers import SentenceTransformer, util
import re
from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim import models, corpora
from gensim.parsing.preprocessing import preprocess_string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
import pandas as pd
import requests
import openpyxl
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from itertools import combinations
from collections import Counter
from datetime import datetime

import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize


# creating dataframe
############################################################
def txt_to_df(textfile):
    
    # List to store data for DataFrame
    data = {'content': []}

    # Append data to the list
    data['content'].append(textfile)
 
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    return df



# function to remove text before "WEBVTT"
####################################################
def remove_text_before_webvtt(df):
    txt_new = []
    for row in range(len(df)):
        text = df['content'][row]
        # Function to remove text before "WEBVTT"
        index_webvtt = text.find("WEBVTT")
        txt1 =  text[index_webvtt + len("WEBVTT"):] if index_webvtt != -1 else text
        txt_new.append(txt1)
    return txt_new



#  Function to remove the specified pattern and unwanted spaces
####################################################
def remove_timestamp_pattern(df):
    txt_new = []
    for row in range(len(df)):
        text = df['content'][row]
        pattern = r'\b[a-zA-Z]+\s\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}\b'
        txt1 =  re.sub(pattern, '', text)
        # remove additional space from string
        txt1 = re.sub(' +', ' ', txt1)
        txt_new.append(txt1)
    return txt_new
 

# Function to preprocess the content column
####################################################
def preprocess_content(text):
    lines = text.split('\n')
    result = ''
    current_speaker = None

    for line in lines:
        if line.strip().isdigit():
            current_speaker = None
        elif line.strip() and current_speaker is None:
            current_speaker = line.strip()
        elif line.strip():
            result += f'{current_speaker} : {line}\n\n'
    # Replace ".\n\n" with ";\n"
    res = result.replace(".\n\n", ";\n\n")
    return res.strip()
 

# Function to split content into chunks
####################################################
def split_into_chunks(content, max_tokens):
    sentences = sent_tokenize(content)
    chunks = []
    current_chunk = ''
    current_count = 0

    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        sentence_length = len(sentence_words)

        if current_count + sentence_length <= max_tokens:
            current_chunk += ' ' + sentence
            current_count += sentence_length
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_count = sentence_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
 


# create new dataframe with fixed length chunks after content
####################################################   
def df_with_fixed_length_chunk(df, max_tokens):
    # Create a dictionary to store data for the new DataFrame
    data1 = {'content': []}
 
    # Dynamic column names based on the number of chunks
    max_chunks = df['content'].apply(lambda x: len(split_into_chunks(x,max_tokens))).max()
    column_names = [f'chunk_{i+1}' for i in range(max_chunks)]
 
    # Add dynamic columns to the data dictionary
    for col_name in column_names:
        data1[col_name] = []
 
    # Populate the data dictionary with values
    for index, row in df.iterrows():
        
        content = row["content"]
        
    # Split content into chunks
    content_chunks = split_into_chunks(content, max_tokens)
    content_chunks1 = []

    # Replacing ; with a . and appending it to new content list. 
    for chunk in content_chunks:
        chunk1 = chunk.replace(";", ". ")
        content_chunks1.append(chunk1)

    # Append data to the dictionary
    data1['content'].append(content)

    # Populate dynamic columns with chunks (if available)
    for i, chunk in enumerate(content_chunks1):
        col_name = f'chunk_{i+1}'
        data1[col_name].append(chunk)

    # Create a new DataFrame with dynamic columns. 
    new_df = pd.DataFrame(data1) # Till here, the content had the ; at the end. However, the chunks were in the proper format.

    for row in range(len(new_df)):
        temp_content = new_df['content'][row]
        temp_content = temp_content.replace(";", ". ")
        new_df['content'][row] = temp_content
    return new_df


# function for chunk preprocessing -> A : xyz. B : xyz.
####################################################
def chunk_preprocessing(df, chunk_col):
    for row in range(len(df)):
        text = str(df[chunk_col][row])
        #print(text)
        #text = text.split(";")
        text1 = text.replace(";", ". ") 
        df[chunk_col][row] = text1
    return df


# summarization of the chunks
####################################################
def summarize_chunk(text, models, api_key, api_endpoint):
    # Initialize dictionary to store extracted entities
    extracted_entities = {model["name"]: [] for model in models}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model in models:
        # Construct the API request body
        body = {
                "model_id": model["model_id"],
                "inputs": ["Summarize the following sentence: " + text],
                "parameters": {
                    "decoding_method": "greedy",
                    "repetition_penalty": 1,
                    "min_new_tokens": 1,
                    "max_new_tokens": 300,
                    "moderations": {
                        "hap": {
                            "input": True,
                            "threshold": 0.75,
                            "output": True
                        }
                    }
                },
                "template": {
                    "id": "prompt_builder",
                    "data": {
                        "instruction": 
                        '''
                        As an intelligent assistant your job is to generate comrehensive summary from a meeting transcript.
                        
                        Here are your guidelines:
                        - Focus on identifying the underlying intent behind each segment of the meeting, using this to categorize information more effectively. 
                        - Segregate key topics, action items, and next steps based on the intent expressed in the given summary. 
                        - Encapsulate the main ideas and discussions. 
                        - Maintain a context flow. 
                        - Interpret the intent or goal behind these tasks. 
                        - Reflect the future intentions or plans of the meeting participants. 
                        ''',
                        "input_prefix": "Input:",
                        "output_prefix": "Output:",
                        "examples": []
                    }
                }
            }

        # Make the API call
        response = requests.post(url=f"{api_endpoint}generate", headers=headers, json=body)
        response_json = response.json()

        if 'results' in response_json and response_json['results']:
            results = response_json['results'][0]
            if 'generated_text' in results:
                extracted_entity = results['generated_text']
                extracted_entities[model["name"]].append(extracted_entity)
            else:
                print("No 'generated_text' key in the results.")
        else:
            print("No 'results' key in the response.")

    return extracted_entities



# Function to generate summary for all the chunks created dynamically.
####################################################
def add_summary_columns(df, models, api_key, api_endpoint):
    # Identify chunk columns in the DataFrame
    chunk_columns = [col for col in df.columns if col.startswith('chunk_')]

    # Iterate over each row and each chunk column
    for idx, row in df.iterrows():
        for chunk_col in chunk_columns:
            summary_col = f'generated_summary_{chunk_col}'
            chunk_value = row[chunk_col]

            # Check if the chunk exists and is not null
            if pd.notnull(chunk_value):
                extracted_entities = summarize_chunk(chunk_value, models, api_key, api_endpoint)

                # Assuming first model's summary is used
                generated_text_list = extracted_entities.get(models[0]["name"], [])
                if generated_text_list:
                    df.at[idx, summary_col] = generated_text_list[0]
                else:
                    print(f"No generated text for {models[0]['name']} in the response.")
            else:
                df.at[idx, summary_col] = None  # Set None if chunk does not exist

    return df


# Collating the Chunk Summaries
####################################################
def collate_summary(df):
    summary_columns = [col for col in df.columns if col.startswith('generated_')]

    df["final_summary"] = ''

    for row in range(len(df)):
        final_summary = ''
        for col in summary_columns:
            final_summary += df[col][row]

        df["final_summary"][row] = final_summary
    
    return df
            

# MOM Generation
####################################################
def generate_MOM(df, models, api_key, api_endpoint):

    # Initialize lists to store extracted entities and model IDs
    extracted_entities = {model["name"]: [] for model in models}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model_ids = {model["name"]: [] for model in models}

    # Iterate over each input sentence
    for i, sentence in enumerate(df['final_summary']):
        for model in models:
            # Construct the API request body for the current model
            body = {
                "model_id": model["model_id"],
                "inputs": ["Summarize the following sentence: " + sentence],
                #You can change the parameters repetition_penalty,max_new_tokens. If you want to use sampling method for more generative summary update the parameters based on that.
                "parameters": {
                    "decoding_method": "greedy",
                    "repetition_penalty": 1,
                    "min_new_tokens": 1,
                    "max_new_tokens": 300,
                    "moderations": {
                        "hap": {
                            "input": True,
                            "threshold": 0.75,
                            "output": True
                        }
                    }
                },
                      "template": {
                    "id": "prompt_builder",
                    "data": {
                        #Here you can modify the prompt as per the requirement
                        "instruction": 
                        '''
                        [INST]
                        As an intelligent assistant your job is to generate professional, concise, and readable Minutes of Meeting from a meeting summary.
                        Here are your guidelines:
                        -Focus on identifying the underlying intent and corresponding speaker name behind each segment of the meeting, using this to categorize information more effectively.
                        - You only need 4 points for the Meeting i.e., Meeting Name, Key Topics Discussed, Action Items and Next Steps.
                        1. Meeting name : Add product names discussed or the main topic involved. Do not want a generic name.
                        2. Key topics : Encapsulate the main ideas and discussions. Maintain a context flow among the points. Mention top 5 key points. I do not need more than 5 points as key points discussed. Don't make it too verbose and lengthy.
                        3. Action items : Interpret the intent or goal behind these tasks. Mention top 3 key points.
                        4. Next steps : Reflect the future intentions or plans of the meeting participants. Mention top 3 key points.
                        - Do not add anything after Next Steps. End with Next Steps only.
                        - Do not mention anything like "Please provide the actual meeting name, key discussion points, action items, and next steps based on the given input."
                        - Learn from the examples given below
                        Example 1:
                        Here is the summary:
                        
                        The meeting began with John Doe initiating a review of last week's progress. Jane Smith reported the completion of initial coding for the login module, resolving issues with third-party authentication. Alex Johnson discussed the implementation of a new database schema and proposed two indexing strategies, with a vote favoring user activity-based indexing. 
                        Emily White presented UI/UX design proposals focusing on user engagement and navigation. The team agreed to align backend development with the new designs and review them in detail by Wednesday. Mark Brown highlighted QA process improvements, emphasizing continuous testing through automated integration in the deployment pipeline.
                        During the "Any Other Business" segment, Jane Smith raised the need to adjust project timelines due to additional client-requested features, prompting a plan for a separate meeting. No further points were raised, and John Doe concluded the meeting, expressing gratitude for contributions and scheduling the next meeting for the following Monday.
                        
                        Expected response:


                        Meeting Name: Weekly Project Update Meeting

                        * Key Discussion Points:

                        1. Login module coding completed; third-party authentication issues resolved.
                        2. New database schema implemented; user activity-based indexing chosen by vote.
                        3. UI/UX proposals presented, focusing on user engagement; backend development alignment agreed upon.
                        
                        * Action Items:

                        1. Implement user activity-based indexing strategy.
                        2. Provide detailed feedback on UI/UX design proposals by Wednesday.
                        3. Schedule a separate meeting to discuss and adjust project timelines based on additional client features.
                        
                        * Next Steps:

                        1. Thoroughly review UI/UX design proposals and provide feedback by Wednesday.
                        2. Finalize the implementation plan for continuous testing through automated integration in the deployment pipeline.
                        3. Conduct a separate meeting to discuss and adjust project timelines based on additional client features.
                       
                        '''
                        
                        ,
                        "input_prefix": "Input:",
                        "output_prefix": "Output:",
                        "examples": []
                    }
                }
            }
            # Make API call for the current model
            response = requests.post(url=f"{api_endpoint}generate", headers=headers, json=body)
            response_json = response.json()
            #print(response_json)
            if 'results' in response_json and response_json['results']:
                results = response_json['results'][0]
                if 'generated_text' in results:
                    extracted_entity = results['generated_text'].strip()
                    #print(extracted_entity)
                    extracted_entities[model["name"]].append(extracted_entity)
                    model_ids[model["name"]].append(json.dumps(body, indent=4))
                else:
                    print("No 'generated_text' key in the results.")
            else:
                print("No 'results' key in the response.")
                
            
            df[f'Generated_mom_with_{model["name"]}'] = extracted_entities[model["name"]]

    return df


# main function
####################################################
def mom(textfile):

    import json
    import requests
    import os
    from dotenv import load_dotenv
    import pandas as pd
    import textstat
    from rouge_metric import Rouge
    from sentence_transformers import SentenceTransformer, util
    import re
    from gensim.models import Word2Vec
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    from gensim import models, corpora
    from gensim.parsing.preprocessing import preprocess_string
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import ssl
    import pandas as pd
    import requests
    import openpyxl
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import numpy as np
    from itertools import combinations
    from collections import Counter
    from datetime import datetime

    import nltk
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize, word_tokenize


    df = txt_to_df(textfile)

    content_new1 = remove_text_before_webvtt(df)
    df['content'] = content_new1
    content_new2 = remove_timestamp_pattern(df)
    df['content'] = content_new2
    df['content'] = df['content'].apply(preprocess_content)
    new_df = df_with_fixed_length_chunk(df, 5000)

    # Load environment variables
    load_dotenv()
    headers = {
        'Authorization': 'Bearer then your api_key'
    }
    # Set your API key and endpoint
    api_key = os.getenv("bam_api_key", None)
    api_endpoint = "https://bam-api.res.ibm.com/v1/"  # Replace with your actual API endpoint

    if api_key is None or api_endpoint is None:
        print("Error: API Key not set up")
    else:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


    # List of models
    models = [
        {"model_id": "mistralai/mistral-7b-instruct-v0-2", "name": "mistral"}
        # Add more models as needed
    ]

    # List of models
    models_1 = [
        #{"model_id": "ibm/granite-13b-instruct-v2", "name": "Granite"},
        {"model_id": "meta-llama/llama-2-70b-chat", "name": "llama"},
        {"model_id": "mistralai/mistral-7b-instruct-v0-2", "name": "mistral"}
        # Add more models as needed
    ]

    new_df1 = add_summary_columns(new_df, models, api_key, api_endpoint)

    new_df2 = collate_summary(new_df1) 

    new_df3 = generate_MOM(new_df2, models_1, api_key, api_endpoint)

    result = str(new_df3['Generated_mom_with_llama'][0])
    
    return result




# file_path = "/Users/tiyasamukherjee/Documents/GitHub/MOM/webex_transcript/Shashanka+B+R's+Personal+Room.txt"
# print(file_path)

# # Open the file in read mode
# with open(file_path, 'r') as file:
#     # Read the entire content of the file into a string
#     file_content = file.read()
#     print(file_content)

# result = mom(file_content)
# print(result)




















