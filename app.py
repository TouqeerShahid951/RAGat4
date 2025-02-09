import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_data
def get_pdf_text_from_resources():
    text = ""
    resources_folder = "Resources"

    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Checking for Resources folder: {os.path.abspath(resources_folder)}")

    if not os.path.exists(resources_folder):
        st.error(f"The '{resources_folder}' directory does not exist.")
        return None

    all_files = os.listdir(resources_folder)
    st.write(f"All files in Resources folder: {all_files}")

    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

    if not pdf_files:
        st.warning(f"No PDF files found in the '{resources_folder}' directory.")
        return None

    st.write(f"PDF files found: {pdf_files}")

    for filename in pdf_files:
        file_path = os.path.join(resources_folder, filename)
        st.write(f"Processing file: {file_path}")
        try:
            pdf_reader = PdfReader(file_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    st.write(f"Extracted text from page {page_num + 1} of {filename}")
                else:
                    st.warning(f"No text could be extracted from page {page_num + 1} of '{filename}'. The page might be scanned or image-based.")
        except Exception as e:
            st.error(f"Error processing '{filename}': {str(e)}")

    if not text:
        st.warning("No text could be extracted from any of the PDF files.")
        return None

    st.write(f"Total characters extracted: {len(text)}")
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n
    Audience: \n{audience}\n

    Please provide an answer suitable for the specified audience.

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "audience"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, audience, vector_store):
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question, "audience": audience}
        , return_only_outputs=True)

    return response["output_text"]

def format_response(response, audience):
    if audience == "Space Science for Kids":
        return f"🚀✨ {response} \n\nWant to know more? Ask another question!"
    elif audience == "Space Science for Student":
        return f"🔭📚 {response} \n\nReferences available upon request."
    elif audience == "Space Science for Professional":
        return f"📊🌌 {response} \n\nPeer-reviewed sources available in citations."
    return response

def main():
    st.set_page_config("Space Science Chatbot")
    st.title("🌌 Space Science Chatbot")
    st.subheader("Explore the cosmos at your level!")

    audience = st.selectbox(
        "Select your audience type:",
        ("Space Science for Kids", "Space Science for Student",
         "Space Science for Professional")
    )

    st.write("Starting PDF processing...")
    with st.spinner("Processing PDFs from Resources folder..."):
        raw_text = get_pdf_text_from_resources()
        if raw_text is None:
            st.error("Unable to process PDFs. Please check the warnings and errors above.")
            return

        st.write("PDF processing completed. Creating text chunks...")
        text_chunks = get_text_chunks(raw_text)
        st.write(f"Number of text chunks created: {len(text_chunks)}")

        st.write("Creating vector store...")
        vector_store = get_vector_store(text_chunks)

    st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about space science:")

    if user_question:
        with st.spinner("Generating response..."):
            response = user_input(user_question, audience, vector_store)
            formatted_response = format_response(response, audience)
            st.write("Reply:", formatted_response)

if __name__ == "__main__":
    main()