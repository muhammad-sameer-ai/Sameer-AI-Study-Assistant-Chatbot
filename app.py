import os
import io
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="Study Assistant & Paper Generator", layout="wide")

st.title("📚 AI Study Assistant & Question Paper Generator")
st.caption("Upload your PDFs, ask strict grounded questions, and generate formal exam papers.")

with st.sidebar:
    st.header("1. Setup & Upload")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_files = st.file_uploader("Upload Notes/PDFs", type=["pdf"], accept_multiple_files=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def create_pdf_paper(paper_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    custom_style = ParagraphStyle(
        'PaperStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        spaceAfter=10
    )
    
    story = []
    lines = paper_text.split('\n')
    for line in lines:
        if line.strip():
            formatted_line = line.replace('**', '<b>').replace('**', '</b>')
            story.append(Paragraph(formatted_line, custom_style))
        else:
            story.append(Spacer(1, 8))
            
    doc.build(story)
    buffer.seek(0)
    return buffer

if api_key and uploaded_files:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    @st.cache_resource(show_spinner="Processing uploaded documents...")
    def process_documents(files):
        raw_text = get_pdf_text(files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_texts(chunks, embedding=embeddings)
        return vector_store

    vector_store = process_documents(uploaded_files)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    tab1, tab2 = st.tabs(["💬 Document Chat", "📝 Generate Question Paper"])

    with tab1:
        st.subheader("Ask questions from your uploaded document")
        user_query = st.text_input("Enter your question:")
        if user_query:
            with st.spinner("Searching document..."):
                docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"""
                You are a strict academic study assistant. Answer the question based ONLY on the provided context below.
                If the answer cannot be found in the context, explicitly reply: "I cannot answer this based on the uploaded material."
                Do NOT use any outside knowledge.

                Context:
                {context}

                Question: {user_query}

                Answer:
                """
                response = llm.invoke(prompt)
                st.write("**Answer:**")
                st.info(response.content)

    with tab2:
        st.subheader("Generate Full Examination Paper")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_mcqs = st.number_input("Number of MCQs", min_value=0, max_value=20, value=5)
        with col2:
            num_short = st.number_input("Short Questions", min_value=0, max_value=10, value=3)
        with col3:
            num_long = st.number_input("Long Questions", min_value=0, max_value=5, value=2)

        paper_title = st.text_input("Subject / Paper Title", value="Midterm Examination")

        if st.button("Generate Paper"):
            with st.spinner("Extracting content and formatting exam paper..."):
                docs = retriever.invoke("Key concepts, definitions, summaries, and main topics in document")
                context = "\n\n".join([doc.page_content for doc in docs])
                paper_prompt = f"""
                Using ONLY the provided context, generate a complete exam question paper titled '{paper_title}'.
                The exam paper MUST follow this exact format:

                TITLE: {paper_title}
                Total Marks: [Calculate appropriate total]
                Time Allowed: 2 Hours

                SECTION A: MULTIPLE CHOICE QUESTIONS ({num_mcqs} Questions)
                Create {num_mcqs} MCQs based strictly on the uploaded text. Provide 4 options (A, B, C, D) for each.

                SECTION B: SHORT ANSWER QUESTIONS ({num_short} Questions)
                Create {num_short} clear, concise short-answer questions.

                SECTION C: LONG ANSWER QUESTIONS ({num_long} Questions)
                Create {num_long} comprehensive essay or analytical long questions.

                Do not include answers or answer keys in the generated paper.

                Context:
                {context}
                """
                
                response = llm.invoke(paper_prompt)
                st.session_state['generated_paper'] = response.content

        if 'generated_paper' in st.session_state:
            st.markdown("### Preview Generated Paper")
            st.text_area("Paper Preview", st.session_state['generated_paper'], height=350)
            
            pdf_data = create_pdf_paper(st.session_state['generated_paper'])
            
            st.download_button(
                label="📄 Download Exam Paper (PDF)",
                data=pdf_data,
                file_name="Question_Paper.pdf",
                mime="application/pdf"
            )

else:
    st.info("Please enter your Gemini API Key and upload at least one PDF in the sidebar to start.")
