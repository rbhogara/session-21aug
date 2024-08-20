### Prerequisites
1. Python 3.9+ 
2. Git 
3. Text Editor 

## Download [Ollama](https://ollama.com/download)

## Download [code](https://github.com/rbhogara/talk-to-docs/archive/refs/heads/main.zip)
Enter directory of code

## Dependencies

To Create virtual environment (Optional):
On MAC/Linux :
```
python3 -m venv talk-to-docs
source talk-to-docs/bin/activate
```
On Windows :
```
python3 -m venv talk-to-docs
talk-to-docs\Scripts\activate
```
To run this application, you need to have the following dependencies installed:

Ollama Dependencies -
```
ollama pull llama3
ollama pull nomic-embed-text
```
Python Dependencies - 
```
pip install -r requirements.txt
```
## Local Chatbot

You can use the below command to have a local chat bot
```
ollama run llama3
```


## Usage

1. Run the application using the following command:
```
streamlit run app.py
```
2. Upload your documents (PDF, DOCX, PPTX) by clicking the "Upload files" button.

3. Once the documents are uploaded, click the "Generate Embeddings" button to process the documents and create the necessary embeddings.

4. Enter your question in the text input field and click the "Query Documents" button to get the answer based on the uploaded documents.


## Resources
This application uses the following libraries and tools:

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://langchain.com/) for the language model and document processing.
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF file processing.
- [python-docx](https://python-docx.readthedocs.io/) for DOCX file processing.
- [python-pptx](https://python-pptx.readthedocs.io/) for PPTX file processing.
- [Chroma](https://www.trychroma.com/) for the vector database.
- [Ollama](https://www.anthropic.com/models) for the language model.


## Feedback
Please fill the feedback.
https://forms.office.com/r/LUhbkY1Y3f
