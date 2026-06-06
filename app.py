import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS

# ----------------------------
# API KEY
# ----------------------------

GOOGLE_API_KEY = "YOUR GOOGLE GEMINI API KEY"          # Enter your gemini api here

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄"
)

st.title("📄 RAG PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about it."
)

# ----------------------------
# PDF Upload
# ----------------------------

pdf_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if pdf_file:

    with st.spinner("Reading PDF..."):

        pdf_reader = PdfReader(pdf_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    # ----------------------------
    # Split Text
    # ----------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # ----------------------------
    # Embeddings
    # ----------------------------

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 


    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    st.success("PDF processed successfully!")

    # ----------------------------
    # Ask Question
    # ----------------------------

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        docs = vector_store.similarity_search(
            question,
            k=4
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a helpful PDF assistant.

Answer ONLY from the provided context.

If the answer is not present,
say:

'I could not find this information in the PDF.'

Context:
{context}

Question:
{question}
"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )

        response = llm.invoke(prompt)

        if isinstance(response.content, list):
          answer = response.content[0]["text"]
        else:
          answer = response.content

        st.subheader("Answer")

        st.markdown(answer)

        with st.expander(
            "Retrieved Context"
        ):

            st.write(context)

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.caption(
    "Built using Gemini + FAISS + LangChain + Streamlit"
)