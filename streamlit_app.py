import streamlit as st
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
import tempfile
import langdetect
from langdetect.lang_detect_exception import LangDetectException
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkblue
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="IRDAI Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 IRDAI Circular Document Analyzer")
st.markdown("Upload PDF documents to analyze and structure IRDAI circulars with headers and sub-headers")

# Sidebar for Azure OpenAI configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Azure OpenAI Configuration
    azure_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        placeholder="https://your-resource.openai.azure.com/",
        help="Your Azure OpenAI service endpoint"
    )
    
    api_key = st.text_input(
        "Azure OpenAI API Key",
        type="password",
        placeholder="Enter your API key",
        help="Your Azure OpenAI API key"
    )
    
    deployment_name = st.text_input(
        "Deployment Name",
        placeholder="gpt-35-turbo",
        help="Name of your deployed model"
    )
    
    api_version = st.selectbox(
        "API Version",
        ["2025-01-01-preview"],
        index=0
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files containing IRDAI circulars"
    )

def extract_english_text(text):
    try:
        sentences = re.split(r'[.!?]+', text)
        english_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                try:
                    lang = langdetect.detect(sentence)
                    if lang == 'en':
                        english_sentences.append(sentence)
                except LangDetectException:
                    if re.search(r'\b(the|and|or|of|to|in|for|with|by|from|at|is|are|was|were)\b', sentence.lower()):
                        english_sentences.append(sentence)
        
        english_text = '. '.join(english_sentences) + '.'
        logger.info({english_text})
    
    except Exception as e:
        st.warning(f"Language detection error: {e}. Using original text.")
        return text
        
def get_summary_prompt(text):
    return f"""
You are a domain expert in insurance compliance and regulation.

Your task is to generate a **clean, concise, section-wise summary** of the input IRDAI/regulatory document while preserving the **original structure and flow** of the document.

---

### Mandatory Summarization Rules:

1. **Follow the original structure strictly** — maintain the same order of:
   - Section headings
   - Subheadings
   - Bullet points
   - Tables
   - Date-wise event history
   - UIDAI / IRDAI / eGazette circulars

2. **Do NOT rename or reformat section titles** — retain the exact headings from the original file.

3. **Each section should be summarized in 1–5 lines**, proportional to its original length:
   - Keep it brief, but **do not omit the core message**.
   - Avoid generalizations or overly descriptive rewriting.

4. If a section contains **definitions**, summarize them line by line (e.g., Definition A: …).

5. If the section contains **tabular data**, preserve **column-wise details**:
   - Include every row and column in a concise bullet or structured format.
   - Do not merge or generalize rows — maintain data fidelity.

6. If a section contains **violations, fines, or penalties**, mention each item clearly:
   - List out exact violation titles and actions taken or proposed.

7. For **date-wise circulars or history**, ensure that:
   - **No dates are skipped or merged.**
   - Maintain **chronological order**.
   - Mention full references such as "IRDAI Circular dated 12-May-2022".

---

### Output Format:

- Follow the exact **order and structure** of the input file.
- Do **not invent new headings** or sections.
- Avoid decorative formatting, markdown, or unnecessary bolding — use **clean plain text**.

---

### Guideline:

Now, generate a section-wise structured summary of the document below:
--------------------
{text}
"""

def create_pdf_styles():
    styles = getSampleStyleSheet()
    
    def safe_add_style(name, style):
        if name not in styles:
            styles.add(style)
    
    safe_add_style('IRDAITitle', ParagraphStyle(
        name='IRDAITitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=darkblue,
        spaceAfter=20,
        alignment=TA_CENTER
    ))
    
    safe_add_style('IRDAIMainHeader', ParagraphStyle(
        name='IRDAIMainHeader',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=darkblue,
        spaceAfter=12,
        spaceBefore=16,
        leftIndent=0
    ))
    
    safe_add_style('IRDAISubHeader', ParagraphStyle(
        name='IRDAISubHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=blue,
        spaceAfter=8,
        spaceBefore=10,
        leftIndent=20
    ))
    
    safe_add_style('IRDAISubSubHeader', ParagraphStyle(
        name='IRDAISubSubHeader',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=black,
        spaceAfter=6,
        spaceBefore=8,
        leftIndent=40
    ))
    
    safe_add_style('IRDAIBodyText', ParagraphStyle(
        name='IRDAIBodyText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        spaceBefore=3,
        leftIndent=0,
        alignment=TA_JUSTIFY
    ))
    
    safe_add_style('IRDAIBulletText', ParagraphStyle(
        name='IRDAIBulletText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        spaceBefore=2,
        leftIndent=20,
        bulletIndent=10
    ))
    
    return styles

def parse_structured_text_to_pdf(text, filename="irdai_summary.pdf"):
    
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    styles = create_pdf_styles()
    
    story = []
    
    title = Paragraph("IRDAI Document Analysis Summary", styles['IRDAITitle'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('## '):
            header_text = line[3:].strip()
            para = Paragraph(header_text, styles['IRDAIMainHeader'])
            story.append(para)
            
        elif line.startswith('### '):
            header_text = line[4:].strip()
            para = Paragraph(header_text, styles['IRDAISubHeader'])
            story.append(para)
            
        elif line.startswith('#### '):
            header_text = line[5:].strip()
            para = Paragraph(header_text, styles['IRDAISubSubHeader'])
            story.append(para)
            
        elif line.startswith('• ') or line.startswith('- '):
            bullet_text = line[2:].strip()
            para = Paragraph(f"• {bullet_text}", styles['IRDAIBulletText'])
            story.append(para)
            
        elif re.match(r'^\d+\.\s', line):
            para = Paragraph(line, styles['IRDAIBulletText'])
            story.append(para)
            
        else:
            if line:
                para = Paragraph(line, styles['IRDAIBodyText'])
                story.append(para)
    
    doc.build(story)
    
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def initialize_azure_openai(endpoint, api_key, deployment_name, api_version):
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0.3
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {str(e)}")
        return None

def load_pdf_documents(uploaded_files):
    all_documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            all_documents.extend(documents)
            
            st.success(f"✅ Loaded {len(documents)} pages from {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    return all_documents

def analyze_documents_summary(documents, llm):
    try:
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        english_content = extract_english_text(combined_content)
        
        page_count = len(documents)
        
        summary_prompt = get_summary_prompt(english_content)
        
        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template="{prompt}"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        with st.spinner("🔄 Generating document summary..."):
            result = chain.run(prompt=summary_prompt)
        
        return result
    
    except Exception as e:
        st.error(f"Error during summary generation: {str(e)}")
        return None

if uploaded_files and azure_endpoint and api_key and deployment_name:
    
    with col2:
        st.header("📋 Document Processing")
        
        if st.button("🚀 Generate Document Summary", type="primary"):
            # Initialize Azure OpenAI
            llm = initialize_azure_openai(azure_endpoint, api_key, deployment_name, api_version)
            
            if llm:
                # Load PDF documents
                documents = load_pdf_documents(uploaded_files)
                
                if documents:
                    st.info(f"📊 Total pages loaded: {len(documents)}")
                    
                    # Generate document summary
                    summary_result = analyze_documents_summary(documents, llm)
                    
                    if summary_result:
                        st.success("✅ Document summary generated successfully!")
                        
                        # Display results
                        st.header("📄 Document Summary")
                        st.markdown("---")
                        st.text(summary_result)
                        
                        # Generate PDF
                        with st.spinner("🔄 Generating PDF..."):
                            pdf_data = parse_structured_text_to_pdf(summary_result)
                        
                        # Download options
                        col_txt, col_pdf = st.columns(2)
                        
                        with col_txt:
                            st.download_button(
                                label="📥 Download as Text",
                                data=summary_result,
                                file_name="irdai_document_summary.txt",
                                mime="text/plain"
                            )
                        
                        with col_pdf:
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_data,
                                file_name="irdai_document_summary.pdf",
                                mime="application/pdf"
                            )

elif uploaded_files:
    with col2:
        st.warning("⚠️ Please complete the Azure OpenAI configuration in the sidebar before analyzing documents.")

else:
    with col2:
        st.info("👈 Please upload PDF files to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. **Configure Azure OpenAI**: Enter your Azure OpenAI endpoint, API key, and deployment details in the sidebar
2. **Upload Documents**: Select one or more PDF files containing IRDAI circulars
3. **Analyze**: Click the 'Generate Document Summary' button to process the documents
4. **Review Results**: The analysis will appear with proper formatting
5. **Download**: Save the analysis results as text or formatted PDF

### 🔧 Features:
- **English Text Processing**: Automatically extracts and processes only English content
- **Structured PDF Output**: Creates professionally formatted PDF with proper headers and subheaders
- **Hierarchical Formatting**: Maintains document structure with appropriate indentation and styling
- **Multiple Download Options**: Save as both text and PDF formats

### 📋 Requirements:
- Valid Azure OpenAI service subscription
- PDF files containing IRDAI circulars
- Stable internet connection for API calls

### 📄 PDF Features:
- Professional document formatting
- Hierarchical header structure
- Proper indentation and spacing
- Color-coded headers for easy navigation
- Generation timestamp
""")
