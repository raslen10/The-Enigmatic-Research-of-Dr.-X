import streamlit as st
from research_processor import ResearchProcessor
import pandas as pd
from enum import Enum

# Configuration
class TranslationLanguage(Enum):
    NONE = "None"
    ARABIC = "Arabic"
    FRENCH = "French"
    SPANISH = "Spanish"
    GERMAN = "German"

class SummaryStrategy(Enum):
    ABSTRACTIVE = "abstractive"
    EXTRACTIVE = "extractive"

# App layout (must be the first Streamlit command)
st.set_page_config(
    page_title="Dr. X Research Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def inject_custom_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #003366;
            --secondary-color: #4CAF50;
            --background-color: #f8f9fa;
            --card-background: white;
        }
        
        .main {
            background-color: var(--background-color);
        }
        
        .stButton>button {
            background-color: var(--secondary-color);
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .card {
            background-color: var(--card-background);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }
        
        .answer-card {
            background-color: var(--primary-color);
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .sidebar .sidebar-content {
            background-color: #e9ecef;
            padding: 1.5rem 1rem;
        }
        
        .metric-box {
            background-color: var(--card-background);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .tab-content {
            padding: 1.5rem 0;
        }
        
        h1, h2, h3 {
            color: var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize processor (with caching)
@st.cache_resource
def get_processor():
    processor = ResearchProcessor()
    processor.load_documents()
    return processor

# UI Components
def display_document_list(processor):
    st.subheader("Available Documents")
    documents = processor.list_documents()
    
    if not documents:
        st.info("No documents found in the database")
        return
    
    cols = st.columns(3)
    for i, doc in enumerate(documents):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div class="card" style="background-color: #f0f0f0; color: #003366; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="color: #003366;">{doc['source']}</h4>
                    <p><strong>Type:</strong> {doc['type']}</p>
                    <p><strong>Language:</strong> {doc['language']}</p>
                    <p><strong>Chunks:</strong> {doc['chunks']}</p>
                </div>
                """, unsafe_allow_html=True)

def display_performance_metrics(processor):
    st.subheader("System Performance")
    perf_data = processor.get_performance_data()
    
    if not perf_data:
        st.info("No performance data available yet")
        return
    
    df = pd.DataFrame([{
        "Process": m.process,
        "Tokens": m.tokens,
        "Time (s)": f"{m.time_taken:.2f}",
        "Tokens/s": f"{m.tokens_per_sec:.1f}"
    } for m in perf_data])
    
    st.dataframe(
        df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Process": "Process",
            "Tokens": st.column_config.NumberColumn("Tokens", format="%d"),
            "Time (s)": "Time (s)",
            "Tokens/s": "Tokens/s"
        }
    )

def render_qa_tab(processor):
    st.header("Research Q&A System")
    
    with st.form("qa_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Ask about Dr. X's research",
                placeholder="What was the main finding about...?",
                key="qa_question"
            )
        with col2:
            target_lang = st.selectbox(
                "Translate answer to",
                [lang.value for lang in TranslationLanguage],
                key="qa_lang"
            )
        
        submitted = st.form_submit_button("Submit Question")
        
        if submitted and question:
            with st.spinner("Analyzing research..."):
                answer, sources, metrics = processor.ask_question(
                    question,
                    target_lang if target_lang != TranslationLanguage.NONE.value else None
                )
                
                st.markdown("### Answer")
                st.markdown(f"""
                    <div class="answer-card">
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                
                if sources:
                    st.markdown("### Source Documents")
                    for src in sources:
                        with st.expander(f"{src['source']} [{src['position']}]"):
                            st.text(src['content'])

def render_translation_tab(processor):
    st.header("Document Translation")
    
    with st.form("translation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            source_file = st.selectbox(
                "Select document to translate",
                [doc['source'] for doc in processor.list_documents()],
                key="trans_file"
            )
        with col2:
            target_lang = st.selectbox(
                "Target language",
                [lang.value for lang in TranslationLanguage if lang != TranslationLanguage.NONE],
                key="trans_lang"
            )
        
        submitted = st.form_submit_button("Translate Document")
        
        if submitted and source_file:
            with st.spinner("Translating document..."):
                translated, metrics = processor.translate_document(
                    source_file,
                    target_lang
                )
                
                st.markdown("### Translation Result")
                st.markdown(f"""
                    <div class="card">
                        {translated[:2000] + ('...' if len(translated) > 2000 else '')}
                    </div>
                """, unsafe_allow_html=True)
                
                if len(translated) > 2000:
                    st.warning("Displaying first 2000 characters. Download full translation below.")
                    
                    st.download_button(
                        label="Download Full Translation",
                        data=translated,
                        file_name=f"translated_{source_file}.txt",
                        mime="text/plain"
                    )

def render_summarization_tab(processor):
    st.header("Document Summarization")
    
    with st.form("summary_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            source_file = st.selectbox(
                "Select document to summarize",
                [doc['source'] for doc in processor.list_documents()],
                key="sum_file"
            )
        with col2:
            strategy = st.radio(
                "Summary type",
                [strat.value for strat in SummaryStrategy],
                key="sum_strat",
                horizontal=True
            )
        
        submitted = st.form_submit_button("Generate Summary")
        
        if submitted and source_file:
            with st.spinner("Creating summary..."):
                summary, scores, metrics = processor.summarize_document(
                    source_file,
                    strategy
                )
                
                st.markdown("### Summary")
                st.markdown(f"""
                    <div class="card">
                        {summary}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Summary Quality Metrics")
                cols = st.columns(3)
                cols[0].metric("ROUGE-1 F1", f"{scores['rouge1'].fmeasure:.3f}")
                cols[1].metric("ROUGE-2 F1", f"{scores['rouge2'].fmeasure:.3f}")
                cols[2].metric("ROUGE-L F1", f"{scores['rougeL'].fmeasure:.3f}")

# Main App
def main():
    inject_custom_css()
    processor = get_processor()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Dr. X Research Analyzer")
        st.markdown("""
        ### Research Analysis System
        Advanced tools for processing and analyzing Dr. X's research documents.
        """)
        
        display_document_list(processor)
        st.markdown("---")
        display_performance_metrics(processor)
    
    # Main content
    st.title("Dr. X Research Analysis System")
    
    tab1, tab2, tab3 = st.tabs([
        "📝 Ask Questions", 
        "🌍 Translate Documents", 
        "✂️ Summarize Documents"
    ])
    
    with tab1:
        render_qa_tab(processor)
    
    with tab2:
        render_translation_tab(processor)
    
    with tab3:
        render_summarization_tab(processor)

if __name__ == "__main__":
    main()