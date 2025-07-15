import streamlit as st
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import pandas as pd
import base64
import io
import re

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime

def get_pdf_text(pdf_docs):
    """Extracts text from one or more uploaded PDFs with enhanced error handling."""
    text = ""

    if isinstance(pdf_docs, list):  
        for pdf_doc in pdf_docs:
            try:
                if pdf_doc.size == 0:
                    st.warning(f"Skipping empty file: {pdf_doc.name}")
                    continue
                
                if not pdf_doc.name.lower().endswith('.pdf'):
                    st.warning(f"Skipping non-PDF file: {pdf_doc.name}")
                    continue
                
                extracted_text = extract_pdf_text(pdf_doc)
                if extracted_text.strip():
                    text += extracted_text + "\n\n"
                    st.success(f"Successfully processed: {pdf_doc.name}")
                else:
                    st.warning(f"No text extracted from: {pdf_doc.name}")
                    
            except Exception as e:
                st.error(f"Error processing {pdf_doc.name}: {str(e)}")
                continue
    else: 
        try:
            if pdf_docs.size == 0:
                st.error("Uploaded file is empty")
                return ""
            
            text += extract_pdf_text(pdf_docs)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    return text

def extract_pdf_text(pdf_doc):
    """Extracts text from a single PDF file-like object with multiple fallback methods."""
    pdf_text = ""
    
    pdf_doc.seek(0)
    
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_doc.read()), strict=False)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"
        return pdf_text
    except PdfReadError as e:
        st.warning(f"PyPDF2 PdfReadError: {str(e)}. This might be a corrupted or unsupported PDF format.")
        pdf_doc.seek(0)  # Reset file pointer for next attempt
    except Exception as e:
        st.warning(f"PyPDF2 failed with error: {str(e)}. Trying alternative method...")
        pdf_doc.seek(0)
    
    try:
        pdf_bytes = pdf_doc.read()
        pdf_doc.seek(0)
        
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
            except Exception as page_error:
                st.warning(f"Skipping page {page_num + 1} due to error: {str(page_error)}")
                continue
        
        if pdf_text.strip():
            return pdf_text
        else:
            raise Exception("No text could be extracted from any pages")
            
    except Exception as e:
        st.error(f"All PDF extraction methods failed: {str(e)}")
        st.error("Please try:")
        st.error("1. Re-saving the PDF from a different application")
        st.error("2. Using a different PDF file")
        st.error("3. Checking if the PDF is password-protected or corrupted")
        return ""

def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def classify_question_type(question):
    """
    Classify if the question requires RAG (document-specific) or general AI knowledge
    """
    
    rag_keywords = [
        'what', 'who', 'when', 'where', 'which', 'how many', 'list', 'mention', 'state', 'according to',
        'in the document', 'in the pdf', 'in the resume', 'in the report', 'based on the document',
        'extract', 'find', 'show', 'display', 'quote', 'cite', 'reference'
    ]
    
    analysis_keywords = [
        'analyze', 'evaluate', 'assess', 'rate', 'score', 'suggest', 'recommend', 'improve', 'optimize',
        'predict', 'estimate', 'compare', 'contrast', 'pros and cons', 'advantages', 'disadvantages',
        'ats score', 'readability', 'effectiveness', 'quality', 'performance', 'how to', 'what should',
        'best practices', 'tips', 'advice', 'strategy', 'approach', 'method', 'technique'
    ]
    
    question_lower = question.lower()
    
    for keyword in analysis_keywords:
        if keyword in question_lower:
            return "hybrid"  
    
    for keyword in rag_keywords:
        if keyword in question_lower:
            return "rag" 
    
    return "hybrid"

def get_rag_chain(model_name, api_key=None):
    """Traditional RAG chain for document-specific questions"""
    if model_name == "Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

def get_hybrid_chain(model_name, api_key=None):
    """Hybrid chain that uses document context + AI knowledge for analysis"""
    if model_name == "Google AI":
        prompt_template = """
        You are an intelligent assistant that can analyze documents and provide insights using both the document content and your general knowledge.
        
        Document Content:
        {context}
        
        Question: {question}
        
        Instructions:
        1. First, carefully read and understand the document content provided above
        2. Use the document content as the primary source of information
        3. Apply your general knowledge and analytical skills to provide insights, analysis, or recommendations
        4. If the question requires analysis (like ATS score, readability, effectiveness), use both document content and your expertise
        5. Be specific and provide actionable insights
        6. If you need more information that's not in the document, mention what additional details would be helpful
        
        Provide a comprehensive and helpful response:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

def get_general_chain(model_name, api_key=None):
    """General AI chain for questions without document context"""
    if model_name == "Google AI":
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
        return model

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None:
        st.warning("Please provide API key before processing.")
        return
    
    try:
        question_type = classify_question_type(user_question)
        
        user_question_output = user_question
        response_output = ""
        
        if pdf_docs is not None and len(pdf_docs) > 0:
            pdf_text = get_pdf_text(pdf_docs)
            
            if not pdf_text.strip():
                st.error("No text could be extracted from the PDF documents. Please check if the files are valid PDFs.")
                return
            
            text_chunks = get_text_chunks(pdf_text, model_name)
            
            if not text_chunks:
                st.error("No text chunks could be created. The PDF might be empty or corrupted.")
                return
            
            vector_store = get_vector_store(text_chunks, model_name, api_key)
            
            if model_name == "Google AI":
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question)
                
                if question_type == "rag":
                    chain = get_rag_chain(model_name, api_key)
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    response_output = response['output_text']
                    
                elif question_type == "hybrid":
                    chain = get_hybrid_chain(model_name, api_key)
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    response_output = response['output_text']
        
        else:
            if model_name == "Google AI":
                model = get_general_chain(model_name, api_key)
                response_output = model.invoke(user_question).content
        
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else ["No PDF uploaded"]
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        st.markdown(
            f"""
            <style>
                .chat-message {{
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    display: flex;
                }}
                .chat-message.user {{
                    background-color: #2b313e;
                }}
                .chat-message.bot {{
                    background-color: #475063;
                }}
                .chat-message .avatar {{
                    width: 20%;
                }}
                .chat-message .avatar img {{
                    max-width: 78px;
                    max-height: 78px;
                    border-radius: 50%;
                    object-fit: cover;
                }}
                .chat-message .message {{
                    width: 80%;
                    padding: 0 1.5rem;
                    color: #fff;
                }}
                .chat-message .info {{
                    font-size: 0.8rem;
                    margin-top: 0.5rem;
                    color: #ccc;
                }}
                .question-type {{
                    font-size: 0.7rem;
                    color: #888;
                    margin-bottom: 0.5rem;
                }}
            </style>
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">
                    <div class="question-type">Question type: {question_type.upper()}</div>
                    {user_question_output}
                </div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{response_output}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if len(conversation_history) == 1:
            conversation_history = []
        elif len(conversation_history) > 1:
            last_item = conversation_history[-1]
            conversation_history.remove(last_item)
            
        for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                    </div>    
                    <div class="message">{question}</div>
                </div>
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                    </div>
                    <div class="message">{answer}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Download conversation history
        if len(st.session_state.conversation_history) > 0:
            df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
        st.snow()
        
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        st.error("Please try:")
        st.error("1. Re-uploading the PDF files")
        st.error("2. Checking if the PDF files are not corrupted")
        st.error("3. Using different PDF files")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Hybrid RAG + AI Chatbot :books:")
    
    # Add explanation
    st.markdown("""
    **🔍 How it works:**
    - **Document Questions**: "What skills are mentioned?" → Uses RAG (retrieval)
    - **Analysis Questions**: "What's the ATS score?" → Uses AI knowledge + document context
    - **General Questions**: Works even without PDFs uploaded
    """)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    linkedin_profile_link = "https://www.linkedin.com/in/snsupratim/"
    kaggle_profile_link = "https://www.kaggle.com/snsupratim/"
    github_profile_link = "https://github.com/snsupratim/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
            api_key = None
            pdf_docs = None
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()
                else:
                    st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button (Optional)", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    valid_pdfs = []
                    for pdf in pdf_docs:
                        if pdf.size > 0 and pdf.name.lower().endswith('.pdf'):
                            valid_pdfs.append(pdf)
                        else:
                            st.warning(f"Skipping invalid file: {pdf.name}")
                    
                    if valid_pdfs:
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("No valid PDF files found")
            else:
                st.info("You can still ask general questions without uploading PDFs!")

    user_question = st.text_input("Ask any question (works with or without PDFs)")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""

if __name__ == "__main__":
    main()
