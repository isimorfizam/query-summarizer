# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import streamlit as st

#from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import CTransformers
from transformers import BitsAndBytesConfig
import pandas as pd
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import llama_cpp
from llama_cpp import Llama
import llama_cpp.llama_tokenizer
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import time
from datetime import datetime
# SET TO WIDE LAYOUT
st.set_page_config(layout="wide")

#_______________________________________________SET VARIABLES_____________________________________________________
EMBEDDING = "all-MiniLM-L6-v2"
COLLECTION_NAME = f'vb_summarizer_{EMBEDDING}_test'
CHROMA_DATA_PATH = 'feedback_360'

#_______________________________________________LOAD MODELS_____________________________________________________
# LOAD MODEL
@st.cache_resource
class LlamaLLM(LLM):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def __init__(self, model_path: str, **kwargs: Any):
        model_path = model_path
        llm = Llama(model_path=model_path)
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(prompt, stop=stop or [])
        return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}

@st.cache_resource
def load_model():
    llm_model =llama_cpp.Llama.from_pretrained(
                                               #pretrained_model_name_or_path = 
                                                # repo_id="TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF",
                                                # filename="mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf",
                                                repo_id="TheBloke/toxicqa-Llama2-7B-GGUF",
                                                filename="toxicqa-llama2-7b.Q5_K_M.gguf",
                                                # repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
                                                # filename="llama-2-7b-chat.Q4_K_M.gguf",
                                                #tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B"),
                                                embedding=True,
                                                verbose=False,
                                                n_ctx=1024,
                                                cache_dir='./model_cached',
                                                n_gpu_layers = 50,
                                                n_threads=3, 
                                                
                                                )

    # llm_model = Llama(
    #             model_path="/Users/isidora/Documents/cake_vs_code/feedback_360/summarizer_meta_filtering/QuerySummarizer/model_cached/models--TheBloke--toxicqa-Llama2-7B-GGUF/snapshots/bc099cd6618be35ad1c35b57287b47f8e0d1768b/toxicqa-llama2-7b.Q5_K_M.gguf" ,  # Download the model file first
    #             n_ctx=1024,  # The max sequence length to use - note that longer sequence lengths require much more resources
    #             n_threads=3,            # The number of CPU threads to use, tailor to your system and the resulting performance
    #             n_gpu_layers=50,     # The number of layers to offload to GPU, if you have GPU acceleration available
    #             embedding=True,
    #             chat_format="llama-2",
    #             )
    #model.to_bettertransformer()
    #from ctransformers import AutoModelForCausalLM

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    #from ctransformers import AutoModelForCausalLM
    #import ctransformers
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    #llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.q4_K_M.gguf", model_type="llama", gpu_layers=0)
    #llm = CTransformers(model = "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.q4_K_M.gguf", model_type = 'llama')
    #print(llm("AI is going to"))

    return llm_model

# LOAD VECTORSTORE
@st.cache_resource
def load_data(embedding) :
    # CREATE EMBEDDING
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding)
    print(os.getcwd())
    db3 = Chroma(collection_name = COLLECTION_NAME, persist_directory="./chroma", embedding_function = embedding_function)
    return db3

# Create a text element and let the reader know the data is loading.
model_load_state = st.text('Loading model...')
# Load 10,000 rows of data into the dataframe.
llm_model = load_model()
# Notify the reader that the data was successfully loaded.
model_load_state.text('Loading model...done!')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
vectorstore = load_data(EMBEDDING)
# Notify the reader that the data was successfully loaded.
data_load_state.text(f'Loading data...done! {vectorstore.get()} {os.getcwd()}')


# INFERENCE
# def prompt_formatter(reviews, type_of_doc):
#     return f"""You are a summarization bot.
#     You will receive {type_of_doc} and you will extract all relevant information from {type_of_doc} and return one paragraph in which you will summarize what was said.
#     {type_of_doc} are listed below under inputs.
#     Inputs: {reviews}
#     Answer :
#     """
# def prompt_formatter(reviews):
#     return f"""You are a summarization bot.
#     You will an input and summarize in one paragraph the meaning of the input.
#     Do not quote from the input and do not repeat what was said in the input.
#     Do not make things up. 
#     Input: {reviews}
#     Answer :
#     """

# def prompt_formatter(reviews):
#     return f"""You are a summarization bot.
#     You will receive reviews of Clockify from different users.
#     You will summarize what these reviews said while keeping the information about each of the user.
#     You will return the answer in the form : Review [number of review] : [summarization of review].
#     Reviews are listed below.
#     Reviews: {reviews}
#     Answer :
#     """

def prompt_formatter(reviews):
    return f"""You are a summarization bot.
    You will receive reviews of Clockify from different users. 
    You will summarize what are good and bad Clockify qualities according to all reviews.
    Reviews are listed below.
    Do not make things up. Use only information from reviews.
    Reviews: {reviews}
    Answer :
    """

def mirror_mirror(inputs, prompt_formatter):
    prompt = prompt_formatter(inputs)
    #llm_model.set_cache()

    # response = llm_model.create_chat_completion(
    #     messages=[
    #                 {
    #         "role": "user",
    #         "content": prompt
    #     }
    #     ],
    #     # response_format={
    #     #     "type": "text",
    #     # },
    #     temperature = 0.4,
    #     min_p = 0.01,
    #     max_tokens = 256,
    #     #presence_penalty = 100,
    #     repeat_penalty = 2,
    # )

    response = llm_model(
            prompt, # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False        # Whether to echo the prompt
            )

    #print(response)
    output_text = response['choices'][0]['text'].replace(prompt,'')
    return prompt, output_text



def summarization(example : list[str], results_df : pd.DataFrame = pd.DataFrame()) -> pd.DataFrame :

    # INFERENCE
    results = []
    scores = []
    for cnt in range(0,3) : 
        print(cnt)
        prompt, result = mirror_mirror(example, prompt_formatter)

        example_embedded = np.array(llm_model.create_embedding(result)["data"][0]["embedding"]).reshape(1, -1)
        result_embedded = np.array(llm_model.create_embedding(example)["data"][0]["embedding"]).reshape(1, -1)

        score = cosine_similarity(example_embedded,result_embedded)
        scores.append(str(score[0][0]))
        #print(score[0])

        if score>0.1 :
            fin_result = result
            max_score = score
            break
        print(result)
        results.append(f'Summary{cnt} : '+result)
        #print(result+'\n\n')

    # tokenize results and example together
    # try  :
    #     fin_result 
    # except :
    # # if fin_result not already defined, use the best of available results
    #     # add example to results so tokenization is done together (due to padding limitations)
    #     results.append(example)
    #     tokenized = tokenizer(results, return_tensors="pt", padding = True)
    #     A = tokenized.input_ids.numpy()
    #     A = sparse.csr_matrix(A)
    #     # calculate cosine similarity of each pair 
    #     # keep only example X result column
    #     scores = cosine_similarity(A)[:,5]
    #     # final result is the one with greaters cos_score
    #     
    #     max_score = max(scores)

    # save final result and its attributes
    try :
        fin_result
    except :
        fin_result = results[np.argmax(scores)]
    max_score = max(scores)
    row = pd.DataFrame({'model' : 'llama_neka_cpp', 'prompt' : prompt, 'reviews' : example, 'summarization' : fin_result, 'scores' :[max_score] })
    results_df = pd.concat([results_df,row], ignore_index = True)

    return results_df

def create_filter(group:str=None, platform:str=None, ReviewerPosition:str=None, Industry:str=None, CompanySize:str=None,
       UsagePeriod:str=None, LinkedinVerified:str=None, Date:str=None, Rating:str=None) :
    keys = ['group', 'Platform', 'ReviewerPosition', 'Industry', 'CompanySize',
            'UsagePeriod', 'LinkedinVerified', 'Date', 'Rating']
    input_keys = [group,platform, ReviewerPosition, Industry, CompanySize, UsagePeriod, LinkedinVerified, Date, Rating]
    
    # create filter dict 
    filter_dict = {}
    for key, in_key in zip(keys, input_keys) :
        if not in_key == None and not in_key == ' ':
            filter_dict[key] = {'$eq' : in_key}

    print(filter_dict)
    return filter_dict

#_______________________________________________UI_____________________________________________________

def clock(starttime):
    """Print a time in a field"""
    tdelta =  datetime.now().replace(microsecond=0) - starttime.replace(microsecond=0)
    minutes, seconds = divmod(int(tdelta.total_seconds()), 60)      
    #hours, minutes = divmod(minutes, 60)       
    return (minutes, seconds)                                                                
    #field.metric(name, f"{hours}:{minutes:02d}:{seconds:02d}")

st.title("Mirror, mirror, on the cloud, what do Clockify users say aloud?")
st.subheader("--Clockify review summarizer--")

col1, col2, col3 = st.columns(3, gap = 'small')

with col1:
   platform = st.selectbox(label = 'Platform',
             options = [' ', 'Capterra', 'Chrome Extension', 'GetApp', 'AppStore', 'GooglePlay',
                        'Firefox Extension', 'JIRA Plugin', 'Trustpilot', 'G2',
                        'TrustRadius']
             )

with col2:
   company_size = st.selectbox(label = 'Company Size',
             options = [' ', '1-10 employees', 'Self-employed', 'self-employed',
                        'Small-Business(50 or fewer emp.)', '51-200 employees',
                        'Mid-Market(51-1000 emp.)', '11-50 employees',
                        '501-1,000 employees', '10,001+ employees', '201-500 employees',
                        '1,001-5,000 employees', '5,001-10,000 employees',
                        'Enterprise(> 1000 emp.)', 'Unknown', '1001-5000 employees']
             )

with col3:
   linkedin_verified = st.selectbox(label = 'Linkedin Verified',
             options = [' ', 'True', 'False'],
             placeholder = 'Choose an option'
             )

num_to_return = int(st.number_input(label = 'Number of documents to return', min_value = 2, max_value = 50, step = 1))

# group = st.selectbox(label = 'Review Platform Group',
#              options = ['Software Review Platforms', 'Browser Extension Stores', 'Mobile App Stores', 'Plugin Marketplace']
#              )


default_value = "Clockify"

query = st.text_area("Query", default_value, height = 50)
#type_of_doc = st.text_area("Type of text", 'text', height = 25)

# result = ''
# score = ''
# reviews = ''

if 'result' not in st.session_state:
    st.session_state['result'] = ''

if 'score' not in st.session_state:
    st.session_state['score'] = ''

if 'reviews' not in st.session_state:
    st.session_state['reviews'] = ''

col11, col21  = st.columns(2, gap = 'small')

with col11:
   button_query = st.button('Conquer and query!')
with col21:
    button_summarize = st.button('Summon the summarizer!')


if  button_query :
    print('Querying')
    # create filter from drop-downs
    filter_dict = create_filter(#group = group,
                                platform = platform,
                                CompanySize = company_size,
                                LinkedinVerified = linkedin_verified
                    )
    # FILTER BY META
    if filter_dict == {} :
        retriever = vectorstore.as_retriever(search_kwargs = {"k": num_to_return})

    elif len(filter_dict.keys()) == 1 :
        retriever = vectorstore.as_retriever(search_kwargs = {"k": num_to_return,
                                                              "filter":  filter_dict})
    else :
        retriever = vectorstore.as_retriever(search_kwargs = {"k": num_to_return,
                                                            "filter":{'$and': [{key : value} for key,value in filter_dict.items()]}
                                                        }
        )
    
    reviews = retriever.get_relevant_documents(query = query)
    # only get page content
    st.session_state['reviews'] = [f'{review.page_content}\n\n' for cnt,review in enumerate(reviews)]
    #print(st.session_state['reviews'])
    result = 'You may summarize now!'

if button_summarize :
    starttime = datetime.now()
    print('Summarization in progress')
    st.session_state['result'] = 'Summarization in progress'
    results_df = summarization("\n".join(st.session_state['reviews']))
    # only one input
    st.session_state['result'] = results_df.summarization[0]
    st.session_state['score'] = results_df.scores[0]
    clock = clock(starttime)
    clock_field = st.text_area('Timer', clock)


col12, col22  = st.columns(2, gap = 'small')

with col12:
   chosen_reviews = st.text_area("Reviews to be summarized", "\n".join(st.session_state['reviews']), height = 275)
with col22:  
    summarized_text = st.text_area("Summarized text", st.session_state['result'], height = 275)

score = st.text_area("Cosine similarity score", st.session_state['score'], height = 25)
