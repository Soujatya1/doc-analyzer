import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
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
        ["2023-05-15", "2023-07-01-preview", "2023-08-01-preview"],
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

# Define the prompt template
prompt_template = PromptTemplate(
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

def initialize_azure_openai(endpoint, api_key, deployment_name, api_version):
    """Initialize Azure OpenAI LLM"""
    try:
        llm = AzureOpenAI(
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

def analyze_documents(documents, llm, prompt_template):
    """Analyze documents using Azure OpenAI"""
    try:
        # Combine all document content
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Create LLM chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Generate analysis
        with st.spinner("🔄 Analyzing documents with Azure OpenAI..."):
            result = chain.run(document_content=combined_content)
        
        return result
    
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
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
                    
                    # Analyze documents
                    analysis_result = analyze_documents(documents, llm, prompt_template)
                    
                    if analysis_result:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Display results
                        st.header("📄 Analysis Results")
                        st.markdown("---")
                        st.markdown(analysis_result)
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Analysis",
                            data=analysis_result,
                            file_name="irdai_circular_analysis.md",
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
2. **Upload Documents**: Select one or more PDF files containing IRDAI circulars
3. **Analyze**: Click the 'Analyze Documents' button to process the documents
4. **Review Results**: The structured analysis will appear with headers, sub-headers, and key points
5. **Download**: Save the analysis as a markdown file for future reference

### 🔧 Requirements:
- Valid Azure OpenAI service subscription
- PDF files containing IRDAI circulars
- Stable internet connection for API calls
""")
