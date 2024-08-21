import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
import tempfile

load_dotenv()


# Define global variables
model = None
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

def load_model():
    global model
    genai.configure(api_key=google_api_key)
    model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192", temperature=0.5)

prompt_template = ChatPromptTemplate.from_template(
    """
AI Legal Advisor

This is your introduction - Your name is "AI Legal Advisor"

You are an AI Legal Advisor, a comprehensive platform providing accessible and accurate information on Indian laws based on the Constitution of India.

Your Objective is to assist users with legal queries by offering clear, concise, and informative responses.

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

Provide response having the structure:

Context: {context}?
Query in Focus: {input}

Let’s dive in and explore! 🎓
Answer:
    """
)

def get_vector_store(progress):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        progress.progress(10)
        
        st.session_state.loader = PyPDFDirectoryLoader("./COI")
        st.session_state.docs = st.session_state.loader.load()
        progress.progress(30)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        progress.progress(60)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        progress.progress(100)

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def handle_additional_pdf_upload(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())  # Extract the text from the PDF and add to the list
    return documents

def add_documents_to_vector_store(documents, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
    embedded_docs = embeddings.embed_documents([doc.page_content for doc in documents])  # Embed the documents
    vector_store.add_documents(documents)  # Add the documents with the embedded vectors
    return vector_store

def main():
    global model
    global groq_api_key
    global google_api_key

    st.set_page_config(
        page_title='Justice Genie',
        layout="wide",
        page_icon="🎗"
    )
    
    st.sidebar.title("Justice Genie")
    st.sidebar.image('logo/legal ai logo .jpg', use_column_width=True, caption='Justice Genie 🎗')

    # Initialize session state for the question
    if "question" not in st.session_state:
        st.session_state.question = ""

    # Theme toggle handling
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    def toggle_theme():
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

    button_label = "Switch to Light Theme" if st.session_state.theme == 'dark' else "Switch to Dark Theme"
    st.button(button_label, on_click=toggle_theme)

    # Apply theme styles
    if st.session_state.theme == 'dark':
        st.markdown("""  
            <style>  
            .appview-container {  
                background-color: #000000;  
            }  
            .appview-container .chat-message, .output-container {  
                color: #ffffff !important;  
                background-color: #333333;
            }  
            </style>  
        """, unsafe_allow_html=True)
    else:
        st.markdown("""  
            <style>  
            .appview-container {  
                background-color: #60caf7;
            }  
            .appview-container .chat-message, .output-container {  
                color: #333333 !important;
                background-color: #f0f0f0;
            }  
            </style>  
        """, unsafe_allow_html=True)

    if groq_api_key and google_api_key:
        load_model()
    else:
        st.warning('Please enter valid API keys!')

    # Button for voice input
    if st.button("Speak"):
        spoken_question = speech_to_text()
        if spoken_question:
            st.session_state.question = spoken_question

    # Input via chat (no value parameter; using session state)
    user_question = st.chat_input("Type your question here", disabled=not (groq_api_key and google_api_key and st.session_state.get('embedding_done', False)))

    # If there is a pre-filled question from speech input, use it
    if st.session_state.question and not user_question:
        user_question = st.session_state.question

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.session_state.question = ""

    def print_praise():
        praise_quotes = """
    Disha Gupta
    Ashish Kumar
    Avnish Singh 
    Kunj Bhasin
    Akrati Gupta 
    2nd year Students
    B.Tech(Hons) CSE 
    Specialization in AI and Analytics 
        """
        title = "*Developed By -*\n"
        return title + praise_quotes

    if google_api_key and groq_api_key:
        with st.spinner("Getting ready for you ... "):
            progress_bar = st.progress(0)
            get_vector_store(progress_bar)
            st.success("Chat is unlocked. You can ask Questions now...")
            st.session_state.embedding_done = True
    else:
        st.error("Please Enter API Keys first")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    document_chain = create_stuff_documents_chain(model, prompt_template)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    start = time.process_time()

                    response = retrieval_chain.invoke({'input': user_question})

                    print("Response time:", time.process_time() - start)

                    st.write(response['answer'])

                    # With a Streamlit expander
                    with st.expander("Document Similarity Search"):
                        # Find the relevant chunks
                        for i, doc in enumerate(response["context"]):
                            st.write(doc.page_content)
                            st.write("--------------------------------")

            message = {"role": "assistant", "content": response['answer']}
            st.session_state.messages.append(message)

    st.sidebar.title("Want to restart the conversation🔂,\n Click the button below 🗣️👇")
    st.sidebar.button('Start a New Chat', on_click=clear_chat_history)


    with st.sidebar.expander("Upload Case File(s)📂", icon=":material/info:"):
        pdf_docs = st.file_uploader("Case file store. Upload a PDF file or multiple PDFs📃", type="pdf", accept_multiple_files=True)
        if pdf_docs:
            documents = handle_additional_pdf_upload(pdf_docs)
            if documents:
                st.session_state.vectors = add_documents_to_vector_store(documents, st.session_state.vectors)
        st.success("Case file uploaded successfully")
    st.sidebar.markdown(print_praise())
    with st.sidebar.container(): 
        with st.expander("About Us",icon=":material/info:"):
            st.success("Hello! Welcome to your ultimate platform for all legal queries. We've integrated Article 363A of the Constitution of India to offer precise and reliable information on Indian laws. Our mission is to make legal knowledge accessible to everyone. Just ask your questions, and our intelligent system will provide clear and concise answers. Whether you're seeking legal advice or simply curious about the law, Justice Genie is here to assist you..")


if __name__ == "__main__":
    main()
