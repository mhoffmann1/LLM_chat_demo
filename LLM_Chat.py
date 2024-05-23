#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:a41ca452-a155-4d3c-aee3-030367fc9b20.png)
# 
# # Introduction
# This jupyter notebook shows step by step Python code that provides basic LLM chat functionality. It tries to solve some of the issues encoutered when interacting with chat (ie. halucinations, stopping criteria, history management). It also shows a minimalistic approach to introduce RAG using pdf document. 
# 
# This is for learning/training purposes. High performance was not the goal. There are no optimizations.
# 
# Author: Marcin Hoffmann (marcin.hoffmann@intel.com)
# 
# ----------------
# 

# # Pre-requirements
# 
# This needs to be done prior to running the code:
# 
# 1. Create user account on HuggingFace: https://huggingface.co/
# 1. Request access to Llama2 models (via HuggingFace: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
# 1. Get model file: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# 1. Install Python dependencies: requirements.txt
# 1. Generate HuggingFace access token
# 

# # L1 - Create text generation pipeline

# ## Imports and basic setup

# In[1]:


from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from torch import cuda
from dotenv import load_dotenv

import transformers
import torch
import os

model_id = "TheBloke/Llama-2-7B-Chat-GGUF"
# Set tokenizer that will convert text into tokens
tokenizer_id = 'meta-llama/Llama-2-7b-chat-hf'
model_file = './llama-2-7b-chat.Q4_K_M.gguf'
#HuggingFace access token needed
hf_auth = ''

load_dotenv()
hf_auth = os.getenv("ACCESS_TOKEN")

#If you have NVIDIA GPU...
if torch.cuda.is_available():
    device = f'cuda:{cuda.current_device()}'
    gpu_layers = 13
else:
    device = f'cpu'
    gpu_layers = 0

#Try adding 'context_length: > 512 '
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 
          'temperature': 0.1, 'context_length': 2048}

#gpu_layers - you can send some layers of the model to GPU if you have any
llm = AutoModelForCausalLM.from_pretrained(model_file, gpu_layers=gpu_layers, model_type='llama', hf=True, **config)
llm.eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token = hf_auth)

#Create our basic pipeline to interact with the model
generate_text = transformers.pipeline(
    model=llm, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text, this will be required in a few steps
    task='text-generation',
    do_sample=True,
    temperature=0.1,
    max_new_tokens=256,
    repetition_penalty=1.15,
)


# ### Generation 
# 
# What do the config parameters do:
# + do_sample=True -> this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary with various strategy-specific adjustments.
# + temperature=0.15, -> modulate the next token probabilities
# + max_new_tokens=256,
# + repetition_penalty=1.15 -> penalty for repetition. 1.0 means no penalty

# Smoke test to check if it works

# In[2]:


output = generate_text('What are the 5 biggest cities in USA?')
print(output)


# In[3]:


print(output[0]['generated_text'])


# # L2 - Create simple chat

# In[4]:


from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain

#Initialize HuggingFacePipeline as LangChain to allow using LangChain features
lc_llm = HuggingFacePipeline(pipeline=generate_text)
conversation=ConversationChain(llm=lc_llm)


# ### Let's ask a question!

# In[5]:


entry='What are the 5 biggest cities in USA?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# ### What is under the hood?

# In[6]:


# Enable verbose mode
conversation=ConversationChain(llm=lc_llm, verbose=True)
entry='What are the 5 biggest cities in USA?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# Follow up question...

# In[8]:


entry='Provide more details about the second one'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# # L3 - Add stopping condition

# In[18]:


# Good starting for more details on stop condition: https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/13

# Add a specific set of words that should mark end of generation
stop_list = [' \nHuman:']
stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x) for x in stop_token_ids]
print(f'Stop marker converted to tokens: {stop_token_ids}')

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            print(f'Current tokens analyzed:{tokenizer.decode(input_ids[0][-len(stop_ids[0])+1:])}')
            if torch.eq(input_ids[0][-len(stop_ids[0])+1:], stop_ids[0][1:]).all():
                print('Stopping condition found!')
                return True
        return False    

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


# ### Update generate_text pipeline with stopping_criteria

# In[19]:


generate_text = transformers.pipeline(
    model=llm, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    #Added stopping criteria
    stopping_criteria=stopping_criteria,
    do_sample=True,
    temperature=0.1,
    max_new_tokens=256,
    repetition_penalty=1.15,
)
lc_llm = HuggingFacePipeline(pipeline=generate_text)
conversation=ConversationChain(llm=lc_llm, verbose=True)


# ### Ask our question

# In[11]:


entry='What are the 5 biggest cities in USA?'
result=conversation.invoke({"input": entry})


# In[12]:


print(f'Human: {entry}\nAI: {result["response"]}')


# In[13]:


entry='What was my previous question?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# ### Modify buffer memory
# 
# + Conversation Buffer Memory - keeps entire conversation up to the allowed max limit. Each request sends the entire aggregation 
# + Conversation Buffer Window Memory - keeps last {k} messages only
# + Conversation Summary Memory - continually summarizes the conversation as it's happening to maintain context from start to finish
# + Conversation Summary Buffer Memory - maintains most recent messages, summarizes the older ones
# 

# In[14]:


from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
#memory=ConversationBufferWindowMemory(k=2)
memory=ConversationSummaryMemory(llm=lc_llm) # we use the same model thas is being used for chat, that is not optimal. Just for demo purposes.
conversation=ConversationChain(llm=lc_llm, memory=memory,verbose=True)


# ### Ask our questions

# In[15]:


entry='What are 5 biggest cities in USA?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# Ask follow up question

# In[16]:


entry='I have an oil leak in my car. What can I do about it?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# In[17]:


entry='Tell me about elephants'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI: {result["response"]}')


# # L4 - system prompt modifications

# In[20]:


from langchain.prompts.prompt import PromptTemplate

pt_template= """You are a friendly and talkative AI assistant. Act as if you were a pirate and have a parrot. You provide lots of specific details in a funny way. If you don't not know the answer to a question, you truthfully say you don't not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PT_PROMPT = PromptTemplate(input_variables=["history", "input"], template=pt_template)
#Add ai_prefix to reflect the different AI prefix for memory buffer
memory=ConversationBufferWindowMemory(k=2, ai_prefix="AI Assistant")
#memory=ConversationSummaryMemory(llm=lc_llm)
conversation=ConversationChain(llm=lc_llm, prompt=PT_PROMPT, memory=memory, verbose=True)


# ### Ask our questions

# In[21]:


entry='What are the 5 biggest cities in USA?'
result=conversation.invoke({"input": entry})
print(f'Human: {entry}\nAI Assistant: {result["response"]}')


# # L5 [optional] - adding our own data (RAG - Reinforced Augmented Generation)
# 
# ![rag_diagram.drawio.png](attachment:41fc1b02-41c4-4ca6-97c2-5de73b6228a5.png)
# 

# In[22]:


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

#pdf_list = ["pdfs/descentrules.pdf", "pdfs/descentwodrules.pdf","pdfs/altarofdespair_rules.pdf"]
pdf_list = ["game_rulebook.pdf"]
documents = []
for pdf in pdf_list:
    print(pdf)
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())
#split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
texts = text_splitter.split_documents(documents)
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings_model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=embeddings_model_kwargs)

directory = Path('./data')
if directory.is_dir():
    print(f'Database already exists. Loading...')
    vector_store_db = Chroma(persist_directory='./data', embedding_function=embeddings)
else:
    print(f'Creating chromadb...')
    vector_store_db = Chroma.from_documents(texts, embeddings, persist_directory='./data')
    vector_store_db.persist()

print(f'Number od document slices: {len(texts)}')


# ## Search types
# 
# Chroma:
# + Maximal marginal relevance - selects examples based on a combination of which examples are most similar to the inputs, while also optimizing for diversity. It does this by finding the examples with the embeddings that have the greatest cosine similarity with the inputs, and then iteratively adding them while penalizing them for closeness to already selected examples.
# + Similarity - This object selects examples based on similarity to the inputs. It does this by finding the examples with the embeddings that have the greatest cosine similarity with the inputs.
# + Similiarity with score - same as above but with optional score_threshold
# 

# In[23]:


question='What is the maximum allowed number of card in hand for the overlord?'
mmr_result=vector_store_db.search(question,search_type='mmr')
print(f'\nMMR:')
print(*mmr_result, sep = "\n")
similarity_search = vector_store_db.search(question,search_type='similarity')
print('\nSimilarity search:')
print(*similarity_search, sep = "\n")

similarity_score_search = vector_store_db.similarity_search_with_relevance_scores(question,score_threshold=0.10)
print('\nSimilarity search with score:')
print(*similarity_score_search, sep = "\n")


# ## Creating Chat object that includes memory and RAG context

# In[24]:


from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory


system_message_prompt = """ 
    It is a conversation between human and an AI. You are the game master for "Descent: Journeys in the Dark" and answer to questions based on the context (delimited by <ctx></ctx) and chat history (delimited by <chs></chs>).
    If you don't know the answer, reply that you are unable to find the information. Do not reply with a question.

    <ctx>
    {context}
    </ctx>

    <chs>
    {chat_history}
    </chs>
Human: {question}
AI:
"""

# Create custom Prompt
qa_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=system_message_prompt
)
# Set memory
qa_memory = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True,k=2,return_only_outputs=False)

# Set the RAG retriever
qa_retriever = vector_store_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 3, 'score_threshold': 0.10})

qa_chain = RetrievalQA.from_chain_type(
    llm=lc_llm, retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': qa_prompt,
        'memory': qa_memory,
        #Set to True for info on prompt, context and stoping criteria
        'verbose': True,
    },
    )
user_input='What is the maximum allowed number of card in hand for the overlord?'
qa_result=qa_chain.invoke({'query': user_input} )
print(qa_result['result'])


# ### Check output object

# In[25]:


print("Contents:")
print(qa_result)


# In[26]:


# Function to extract data from response object
def extract_response(qa_chain_output):
    response = qa_chain_output['result']
    _sources = []
    page_info = []
    for knowledge_source in qa_chain_output['source_documents']:
        _item = knowledge_source.dict()
        _sources.append(_item['metadata'])

    # Remove duplicates:
    seen = set()
    unique_list_of_dict = []
    for d in _sources:
        ident = tuple(d.items())
        if ident not in seen:
            seen.add(ident)
            unique_list_of_dict.append(d)
    for i in unique_list_of_dict:
        i['page'] += 1
    return response, unique_list_of_dict


# In[27]:


output, sources = extract_response(qa_result)
print(f'AI response: {output}\nSources: {sources}')


# ### Final question

# In[28]:


user_input='How much gold does hero lose upon getting killed?'
qa_result=qa_chain.invoke({'query': user_input} )
output, sources = extract_response(qa_result)
print(f'AI response: {output}\nSources: {sources}')

