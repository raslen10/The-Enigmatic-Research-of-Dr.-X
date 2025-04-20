# Dr. X Research Analysis System

![Project Banner](https://via.placeholder.com/800x200?text=Dr.+X+Research+Analyzer)

A comprehensive system for analyzing Dr. X's mysterious research publications with advanced NLP capabilities.

## Features

- **Document Processing**: Handles PDF, Word, Excel, CSV formats
- **Multilingual Translation**: Preserves formatting and technical terms
- **Intelligent Summarization**: Abstractive and extractive methods with ROUGE evaluation
- **RAG Q&A System**: Context-aware question answering with source citations
- **Performance Tracking**: Detailed metrics for all operations

## Project Structure

```
drx-research-analyzer/
├── app.py                 # Streamlit application
├── research_processor.py  # Core processing functions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # Sample documents (PDFs, Word, Excel)
└── chroma_db/             # Vector database storage
```

## Setup Instructions

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/)

```bash
# Install required models
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Documents

Place all research documents in the `data` folder:
- PDF files (.pdf)
- Word documents (.docx)
- Excel/CSV files (.xlsx, .xls, .csv)

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage Guide

### 1. Ask Questions
- Enter questions about Dr. X's research
- Optionally translate answers to different languages
- View cited sources with exact document positions

### 2. Translate Text
- Preserves original formatting and technical terms
- Optional context field for domain-specific terms
- Supports multiple target languages

### 3. Summarize Documents
- Choose between abstractive or extractive methods
- Automatic quality evaluation with ROUGE metrics
- Adjustable summary length

## Technical Details

### Core Technologies
- **Ollama**: For running LLM models locally
- **ChromaDB**: Vector database for document storage
- **Streamlit**: Interactive web application
- **Nomic Embeddings**: For document vectorization

### Performance Optimization
- Parallel document processing
- Token-based chunking
- Detailed performance metrics tracking

## BibTeX Citation

```bibtex
@software{DrXResearchAnalyzer,
  author = {Raslen Guesmi},
  title = {Dr. X Research Analysis System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/raslen10/drx-research-analyzer}}
```

## License

MIT License - See [LICENSE](LICENSE) for details

4. requirements.txt

```
streamlit==1.25.0
chromadb==0.4.15
ollama==0.1.6
pypdf2==3.0.1
pandas==2.0.3
python-docx==0.8.11
tiktoken==0.5.1
rouge-score==0.1.2
langdetect==1.0.9
```

How to Run the Project
Set up Ollama:

```
ollama pull llama3
ollama pull nomic-embed-text
ollama serve
```
Install dependencies:

```
pip install -r requirements.txt
```
Add documents:

Place all research files in the data folder

Run the Streamlit app:

```
streamlit run app.py
```
Access the UI:

Open the provided URL in your browser (usually http://localhost:8501)