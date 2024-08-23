# Justice Genie 🎗

**Justice Genie** is an AI-powered legal advisor platform built using Streamlit, LangChain, and Google Generative AI. This tool allows users to upload legal documents (PDFs) and interact with the AI to get insights, analyze documents, and receive answers to legal queries based on a growing knowledge base. The platform is designed to provide reliable and context-aware legal advice in a user-friendly manner.

## Features
- **PDF Document Analysis**: Users can upload legal documents in PDF format, which the AI analyzes to provide insights and relevant information.
- **Knowledge Base**: As more documents are uploaded, Justice Genie’s knowledge base expands, enabling the AI to provide increasingly accurate advice.
- **Question-Answering System**: Users can ask questions related to the uploaded legal documents, and the AI will generate precise answers based on the content.
- **Dynamic PDF Embedding**: The `embed_pdf_via_iframe` function allows for the embedding of large, externally hosted PDFs (e.g., from Google Drive) directly into the platform.

## Installation

To install and run Justice Genie locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/justice-genie.git
   cd justice-genie
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

Once the app is running, you can interact with Justice Genie as follows:

1. **Upload a PDF:** Use the upload feature to add legal documents.
2. **Ask Questions:** Utilize the AI to answer legal queries based on the uploaded documents.
3. **View Analysis:** The platform will analyze the content and provide a detailed view of the document’s key points.

## How It Works

- **Natural Language Processing:** The platform leverages advanced NLP models to understand and process legal language.
- **Embeddings & Knowledge Base:** The system uses embeddings to create a semantic understanding of the text, building a robust knowledge base over time.

## Future Enhancements
- **Jurisdiction-Specific Advice:** Improve the AI’s capability to provide jurisdiction-specific legal advice.
- **Enhanced Accuracy:** Continuously train the model with new data to enhance the accuracy and reliability of the legal advice.
- **Trust Building:** Implement features to ensure the AI's advice is trustworthy for legal professionals.

## Contributing

We welcome contributions! Please follow the standard GitHub flow:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- **Google Generative AI & Groq Models:** For providing the foundational AI technology.
- **Streamlit:** For making it easy to build and deploy interactive web apps.
- **LangChain:** For enabling seamless integration of language models into the app.

