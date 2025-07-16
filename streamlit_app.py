import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
import tempfile

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
    
    # Analysis Type Selection
    st.header("📊 Analysis Options")
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Structured Analysis", "Document Summary", "Both"],
        index=0,
        help="Select the type of analysis to perform"
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

# Define the structured analysis prompt template
structured_prompt_template = PromptTemplate(
    input_variables=["document_content"],
    template="""You are a document analyzer and writer. There are some input IRDAI circulars. Read through the documents and create an output where there will be headers and sub-headers, under which the pointers to be mentioned.

Document Content:
{document_content}

Please analyze the document and structure it with:
1. Clear headers and sub-headers
2. Key points organized under relevant sections
3. Important regulatory information highlighted
4. Actionable items clearly identified

Output Format:
- Use markdown formatting for headers (# ## ###)
- Use bullet points for key information
- Maintain logical flow and structure
- Include any deadlines or compliance requirements

Analysis:"""
)

def get_summary_prompt(text, page_count):
    """Generate summary prompt for the document"""
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
Ensure that the **total summary length does not exceed ~50% of the English content pages** from the input document (total pages: {page_count}).
Now, generate a section-wise structured summary of the document below:
--------------------
{text}
"""

# Create summary prompt template
summary_prompt_template = PromptTemplate(
    input_variables=["document_content", "page_count"],
    template="{document_content}"
)

def initialize_azure_openai(endpoint, api_key, deployment_name, api_version):
    """Initialize Azure OpenAI LLM"""
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0.3,
            max_tokens=2000
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {str(e)}")
        return None

def load_pdf_documents(uploaded_files):
    """Load PDF documents using PyPDFLoader"""
    all_documents = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            all_documents.extend(documents)
            
            st.success(f"✅ Loaded {len(documents)} pages from {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    return all_documents

def analyze_documents_structured(documents, llm, prompt_template):
    """Analyze documents using structured analysis"""
    try:
        # Combine all document content
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Create LLM chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Generate analysis
        with st.spinner("🔄 Performing structured analysis..."):
            result = chain.run(document_content=combined_content)
        
        return result
    
    except Exception as e:
        st.error(f"Error during structured analysis: {str(e)}")
        return None

def analyze_documents_summary(documents, llm):
    """Analyze documents using summary generation"""
    try:
        # Combine all document content
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        page_count = len(documents)
        
        # Generate summary prompt
        summary_prompt = get_summary_prompt(combined_content, page_count)
        
        # Create a simple prompt template for summary
        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template="{prompt}"
        )
        
        # Create LLM chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Generate summary
        with st.spinner("🔄 Generating document summary..."):
            result = chain.run(prompt=summary_prompt)
        
        return result
    
    except Exception as e:
        st.error(f"Error during summary generation: {str(e)}")
        return None

# Process documents when uploaded and configuration is complete
if uploaded_files and azure_endpoint and api_key and deployment_name:
    
    with col2:
        st.header("📋 Document Processing")
        
        if st.button("🚀 Analyze Documents", type="primary"):
            # Initialize Azure OpenAI
            llm = initialize_azure_openai(azure_endpoint, api_key, deployment_name, api_version)
            
            if llm:
                # Load PDF documents
                documents = load_pdf_documents(uploaded_files)
                
                if documents:
                    st.info(f"📊 Total pages loaded: {len(documents)}")
                    
                    # Perform analysis based on selected type
                    if analysis_type == "Structured Analysis":
                        analysis_result = analyze_documents_structured(documents, llm, structured_prompt_template)
                        
                        if analysis_result:
                            st.success("✅ Structured analysis completed successfully!")
                            
                            # Display results
                            st.header("📄 Structured Analysis Results")
                            st.markdown("---")
                            st.markdown(analysis_result)
                            
                            # Download option
                            st.download_button(
                                label="📥 Download Structured Analysis",
                                data=analysis_result,
                                file_name="irdai_structured_analysis.md",
                                mime="text/markdown"
                            )
                    
                    elif analysis_type == "Document Summary":
                        summary_result = analyze_documents_summary(documents, llm)
                        
                        if summary_result:
                            st.success("✅ Document summary generated successfully!")
                            
                            # Display results
                            st.header("📄 Document Summary")
                            st.markdown("---")
                            st.text(summary_result)
                            
                            # Download option
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary_result,
                                file_name="irdai_document_summary.txt",
                                mime="text/plain"
                            )
                    
                    else:  # Both analyses
                        # Structured Analysis
                        analysis_result = analyze_documents_structured(documents, llm, structured_prompt_template)
                        
                        # Document Summary
                        summary_result = analyze_documents_summary(documents, llm)
                        
                        if analysis_result and summary_result:
                            st.success("✅ Both analyses completed successfully!")
                            
                            # Create tabs for results
                            tab1, tab2 = st.tabs(["📋 Structured Analysis", "📄 Document Summary"])
                            
                            with tab1:
                                st.markdown(analysis_result)
                                st.download_button(
                                    label="📥 Download Structured Analysis",
                                    data=analysis_result,
                                    file_name="irdai_structured_analysis.md",
                                    mime="text/markdown"
                                )
                            
                            with tab2:
                                st.text(summary_result)
                                st.download_button(
                                    label="📥 Download Summary",
                                    data=summary_result,
                                    file_name="irdai_document_summary.txt",
                                    mime="text/plain"
                                )
                            
                            # Combined download
                            combined_result = f"# IRDAI Document Analysis\n\n## Structured Analysis\n\n{analysis_result}\n\n## Document Summary\n\n{summary_result}"
                            st.download_button(
                                label="📥 Download Combined Analysis",
                                data=combined_result,
                                file_name="irdai_complete_analysis.md",
                                mime="text/markdown"
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
2. **Select Analysis Type**: Choose between Structured Analysis, Document Summary, or Both
3. **Upload Documents**: Select one or more PDF files containing IRDAI circulars
4. **Analyze**: Click the 'Analyze Documents' button to process the documents
5. **Review Results**: The analysis will appear based on your selected type
6. **Download**: Save the analysis results for future reference

### 🔧 Analysis Types:
- **Structured Analysis**: Creates organized content with headers, sub-headers, and bullet points
- **Document Summary**: Generates a concise, section-wise summary preserving original structure
- **Both**: Performs both analyses and presents results in separate tabs

### 📋 Requirements:
- Valid Azure OpenAI service subscription
- PDF files containing IRDAI circulars
- Stable internet connection for API calls
""")
