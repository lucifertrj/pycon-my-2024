# pip install llama-index-embeddings-fastembed beyondllm streamlit

import streamlit as st

from beyondllm.source import fit
from beyondllm.embeddings import FastEmbedEmbeddings
from beyondllm.retrieve import auto_retriever
from beyondllm.llms import HuggingFaceHubModel
from beyondllm.generator import Generate

st.title("Chat with Doc Chatbot")

@st.cache_resource
def initialize_retriever_and_llm():
    data = fit(
        path = "The_Last_Question.pdf",
        dtype = "pdf",
        chunk_size = 512, #optional
        chunk_overlap = 0 #optional
    )
    st.spinner("Loading the embedding model...")
    embed_model = FastEmbedEmbeddings(model_name="thenlper/gte-large")
    
    st.spinner("Indexing your document...")
    retriever = auto_retriever(
        data = data,
        embed_model = embed_model,
        type = "normal",
        top_k = 3
    )

    HF_TOKEN = st.secrets['HUGGINGFACE_ACCESS_TOKEN']
    llm = HuggingFaceHubModel(
        token = HF_TOKEN,
        model = "mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs = {"max_new_tokens": 1024,
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "repetition_penalty": 1.1,
                        "return_full_text": False
                    }
    )
    return llm,retriever

llm, retriever = initialize_retriever_and_llm()

def generate_response(query):
    prompt = f"<s>[INST] {query} [/INST]"
    pipeline = Generate(question=prompt,llm=llm,retriever=retriever)
    return pipeline.call()
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})