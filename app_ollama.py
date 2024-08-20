import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader


from dotenv import load_dotenv

load_dotenv()
# groq_api_key=os.getenv('GROQ_API_KEY')
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text():
    text=""
    pdf_reader= PdfReader("COI.pdf")
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text
def get_case_text(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
AI Legal Advisor

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
    Answer:
    """
    model = Ollama(model="llama3")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question})
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["input_documents"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

    return response["output_text"]

def handle_additional_pdf_upload(uploaded_files):
    if uploaded_files:
        additional_docs = []
        for uploaded_file in uploaded_files:
            st.session_state.loader = PyPDFLoader.from_file(uploaded_file)
            new_docs = st.session_state.loader.load()
            additional_docs.extend(new_docs)
        
        st.session_state.docs.extend(additional_docs)
        new_final_documents = st.session_state.text_splitter.split_documents(additional_docs)
        st.session_state.final_documents.extend(new_final_documents)
        st.session_state.vectors.add_documents(new_final_documents)
        st.success("Additional PDFs added successfully!")



def main():
    
    st.set_page_config(
        page_title='The legal AI', 
        layout='wide',
        page_icon="🎗"               
    )
    st.logo("sidebar_logo.png", icon_image="only_logo.png")
    
    with st.sidebar.container(): 
        st.image('legal ai logo .jpg', use_column_width=True, caption='The Legal AI 🎗')
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

    if user_question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(user_question)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    st.sidebar.write("---\n")


if __name__ == "__main__":
    main()