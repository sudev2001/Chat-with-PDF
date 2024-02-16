import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#To read the each and every pages and extracting the text
def get_pdf(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Divide the long text into smaller parts having 10000words using RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap= 1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Converting the relevant chunks into vector embeddings and store those in a vector database using  GoogleGenerativeAIEmbeddings and FAISS
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,\n\n
    If the answer is not in provided context just say, "Sorry, I can't get you.", don't provide the wrong answer and the error message\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

import streamlit as st

def main():
    st.set_page_config("Chat With PDF")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button👇", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.header("Chat with your PDF📜")
    
    with st.expander("About this App"):
        st.markdown(
            """
            GenerativeAI Model by Gemini-pro.
            Welcome to the GenerativeAI model developed by Gemini-pro. You can utilize this tool to ask queries about the uploaded PDF, and it will provide you with the necessary answers.
            Feel free to input your questions and get the information you need.
            """
        )
    user_question = st.text_input("Ask your Query from the PDF File😊")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()