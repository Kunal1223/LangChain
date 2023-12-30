import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework
st.title('Langchain Demo With Kunal Kishor')
input_text=st.text_input("Search the Person")

#give the propts here
input_prompts = PromptTemplate(
        input_variables = ['name'],
        template = "Tell me something funny about the celebrity {name}"
)

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain = LLMChain(llm=llm , prompt=input_prompts , verbose=True , output_key="person")

input_prompts2 = PromptTemplate(
        input_variables = ['person'],
        template = "Tell me where the {person} born"
)

chain2 = LLMChain(llm=llm , prompt=input_prompts2 , verbose=True , output_key="dob")

parent_chain= SequentialChain(chains = [chain , chain2], input_variables=['name'] , output_variables = ['person' , 'dob'] , verbose = True)

if input_text:
    st.write(parent_chain({'name': input_text}))