import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGroq(groq_api_key=groq_api_key,
                    model_name="Llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_template(
"""AI Legal Advisor

This is your introduction - Your name is " AI Legal Advisor "

You are an AI Legal Advisor, a comprehensive platform providing accessible and accurate information on Indian laws based on the Constitution of India.

you Objective is to assist users with legal queries by offering clear, concise, and informative responses.

Maintain a friendly, approachable, and professional demeanor. Avoid legal jargon and provide explanations in plain language.

Your knowledge is derived from the Constitution of India and relevant legal data.

Initiate the conversation with a warm greeting and an invitation to ask questions.
Answer user queries comprehensively and professionally.
Handle inquiries about your capabilities with politeness and transparency.

Response Structure:

Contextual Understanding: Clearly identify the context of the user's query.
Direct Answer: Provide a clear and concise answer to the query.
Detailed Explanation: Offer additional details and explanations when necessary to ensure understanding.
Limitations: If unable to provide a comprehensive answer, politely acknowledge the limitation.

Provide response having the structure

Context: {context}?
Query in Focus: {input}

Let’s dive in and explore! 🎓
Context:\n {context}?\n
Question: \n{input}\n

Answer:
"""
)

def get_vector_store():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./COI") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Google embeddings
    
def main():

    st.set_page_config(
        page_title='The legal AI', 
        layout='wide',
        page_icon="The Legal 🎗"               
    )
    st.sidebar.title("The legal AI ")
    st.logo("logo\sidebar_logo.png", icon_image="logo\only_logo.png")
    with st.sidebar.container(): 
        st.image('logo\legal ai logo .jpg', use_column_width=True, caption='The Legal AI 🎗')
        with st.expander("About Us",icon=":material/info:"):
            st.success("Hello! Welcome to your ultimate platform for all legal queries. We've integrated Article 363A of the Constitution of India to offer precise and reliable information on Indian laws. Our mission is to make legal knowledge accessible to everyone. Just ask your questions, and our intelligent system will provide clear and concise answers. Whether you're seeking legal advice or simply curious about the law, The Legal AI is here to assist you..")
        st.sidebar.markdown("---")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    def print_praise():
        praise_quotes = """
    Ashish Kumar
    Avnish singh
    Disha gupta
    kunj bhasin
    Akrati gupta 

    2nd year Student
    B.Tech(Hons) CSE AI-ML
        """
        title = "**Developed By -**\n\n"
        return title + praise_quotes

    with st.sidebar:
        st.title("Start the App by Clicking Here ✅")
        doc=st.button("Start Documents Embedding")
        
        if doc:

            with st.spinner("Processing..."):
                get_vector_store()
                st.info("VectorDB Store is Ready")
                st.success("You're good to go !! ")
                st.success("Ask Questions now...")

        st.sidebar.write("---\n")
        st.sidebar.success(print_praise())
        st.sidebar.write("---\n")
        st.sidebar.info("Special Thanks to our Mentor\n\nDr.Ankur Rai, Professor, \n\nGLA UNIVERSITY, Mathura")
        st.sidebar.write("---\n")

    if user_question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

    import time 

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                document_chain=create_stuff_documents_chain(model,prompt_template)
                retriever=st.session_state.vectors.as_retriever()
                retrieval_chain=create_retrieval_chain(retriever,document_chain)

                start=time.process_time()

                response=retrieval_chain.invoke({'input':user_question})

                print("Response time :",time.process_time()-start)

                st.write(response['answer'])
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")    

        message = {"role": "assistant", "content": response['answer']}
        st.session_state.messages.append(message)
    st.sidebar.title("Looking to Restart your Conversation 🔄")
    st.sidebar.button('Start a New Chat', on_click=clear_chat_history)

    st.sidebar.write("---\n")

if __name__ == "__main__":
    main()
