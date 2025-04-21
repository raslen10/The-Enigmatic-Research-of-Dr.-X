import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import pandas as pd
from docx import Document
import ollama
import tiktoken
from rouge_score import rouge_scorer
from langdetect import detect
import streamlit as st
import plotly.express as px
import torch
from enum import Enum
import tempfile
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_processor.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_FOLDER = "data"
CHROMA_DB_PATH = "chroma_db"
MAX_CHUNK_SIZE = 500  # tokens
MAX_WORKERS = 4

# Model Configuration
MODEL_CONFIG = {
    "embedding": {
        "default": "nomic-embed-text",
        "options": {
            "nomic-embed-text": {
                "description": "General purpose embeddings (768 dim)",
                "performance": 0.92,
                "speed": "fast"
            },
            "llama2": {
                "description": "Llama 2 embeddings (4096 dim)",
                "performance": 0.89,
                "speed": "medium"
            }
        }
    },
    "llm": {
        "translation": {
            "best": "mixtral",
            "options": {
                "mixtral": {
                    "description": "Mixture of Experts model - best for multilingual tasks",
                    "performance": 0.95,
                    "speed": "medium"
                },
                "llama3": {
                    "description": "Meta's latest model - good balance",
                    "performance": 0.91,
                    "speed": "fast"
                },
                "mistral": {
                    "description": "Efficient 7B parameter model",
                    "performance": 0.88,
                    "speed": "fast"
                }
            }
        },
        "summarization": {
            "best": "llama3",
            "options": {
                "llama3": {
                    "description": "Best for abstractive summarization",
                    "performance": 0.94,
                    "speed": "medium"
                },
                "mistral": {
                    "description": "Good for extractive summarization",
                    "performance": 0.89,
                    "speed": "fast"
                },
                "gemma": {
                    "description": "Google's model - good for technical summaries",
                    "performance": 0.92,
                    "speed": "medium"
                }
            }
        },
        "qa": {
            "best": "llama3",
            "options": {
                "llama3": {
                    "description": "Best for question answering with context",
                    "performance": 0.93,
                    "speed": "medium"
                },
                "command-r": {
                    "description": "Good for research-oriented QA",
                    "performance": 0.91,
                    "speed": "fast"
                },
                "mixtral": {
                    "description": "Good for complex multi-part questions",
                    "performance": 0.90,
                    "speed": "medium"
                }
            }
        }
    }
}

@dataclass
class PerformanceMetrics:
    process: str
    model: str
    tokens: int
    time_taken: float
    tokens_per_sec: float

@dataclass
class RougeMetrics:
    source: str
    strategy: str
    model: str
    rouge1: float
    rouge2: float
    rougeL: float
    timestamp: float

class TranslationLanguage(Enum):
    NONE = "None"
    ARABIC = "Arabic"
    FRENCH = "French"
    SPANISH = "Spanish"
    GERMAN = "German"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    RUSSIAN = "Russian"

class SummaryStrategy(Enum):
    ABSTRACTIVE = "abstractive"
    EXTRACTIVE = "extractive"

class ModelManager:
    """Manages model selection and provides performance comparisons."""
    
    @staticmethod
    def get_best_model(task: str) -> str:
        """Get the best model for a specific task."""
        return MODEL_CONFIG["llm"][task]["best"]
    
    @staticmethod
    def get_model_options(task: str) -> Dict:
        """Get available models for a task with their specs."""
        return MODEL_CONFIG["llm"][task]["options"]
    
    @staticmethod
    def get_embedding_model() -> str:
        """Get the default embedding model."""
        return MODEL_CONFIG["embedding"]["default"]
    
    @staticmethod
    def get_embedding_options() -> Dict:
        """Get available embedding models with their specs."""
        return MODEL_CONFIG["embedding"]["options"]
    
    @staticmethod
    def compare_models(task: str) -> str:
        """Generate a comparison table for models of a specific task."""
        options = ModelManager.get_model_options(task)
        best = ModelManager.get_best_model(task)
        
        rows = []
        for name, specs in options.items():
            rows.append({
                "Model": f"**{name}** {'⭐' if name == best else ''}",
                "Description": specs["description"],
                "Performance Score": specs["performance"],
                "Speed": specs["speed"]
            })
        
        df = pd.DataFrame(rows)
        return df.to_markdown(index=False)

class ResearchProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_context_size = 1024
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.performance_data = []
        self.rouge_data = []
        self._init_vector_db()
        self._validate_config()
        self._load_model_comparisons()

    def _load_model_comparisons(self):
        """Generate model comparison documentation."""
        self.model_comparisons = {
            "translation": ModelManager.compare_models("translation"),
            "summarization": ModelManager.compare_models("summarization"),
            "qa": ModelManager.compare_models("qa"),
            "embeddings": ModelManager.compare_models("embeddings")
        }

    def _init_vector_db(self):
        """Initialize the ChromaDB vector database."""
        try:
            logger.info("Initializing ChromaDB client at %s", CHROMA_DB_PATH)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.embedding_model = ModelManager.get_embedding_model()
            self.nomic_ef = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434",
                model_name=self.embedding_model
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="drx_research",
                embedding_function=self.nomic_ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection initialized successfully with %s", self.embedding_model)
        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s", str(e))
            raise

    def _validate_config(self):
        """Validate and create required directories."""
        try:
            logger.info("Validating configuration and creating directories")
            os.makedirs(DATA_FOLDER, exist_ok=True)
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            logger.info("Directories validated: %s, %s", DATA_FOLDER, CHROMA_DB_PATH)
        except Exception as e:
            logger.error("Configuration validation failed: %s", str(e))
            raise

    def add_uploaded_files(self, uploaded_files) -> Dict[str, int]:
        """Process and add uploaded files to the database."""
        file_counts = {"pdf": 0, "docx": 0, "excel": 0, "csv": 0, "text": 0, "errors": 0}
        if not uploaded_files:
            logger.warning("No files uploaded")
            return file_counts

        for uploaded_file in uploaded_files:
            try:
                logger.info("Processing file: %s", uploaded_file.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name

                texts, metadatas = self.process_file(file_path)
                if not texts:
                    logger.warning("No texts extracted from %s", uploaded_file.name)
                    file_counts["errors"] += 1
                    os.unlink(file_path)
                    continue

                chunks, chunk_metadatas = self.chunk_text(texts, metadatas)
                logger.info("Generated %d chunks for %s", len(chunks), uploaded_file.name)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    embeddings = list(executor.map(
                        lambda chunk: ollama.embeddings(
                            model=self.embedding_model,
                            prompt=chunk
                        )['embedding'],
                        chunks
                    ))
                logger.info("Generated embeddings for %s", uploaded_file.name)

                ids = [f"{uploaded_file.name}-{i}" for i in range(len(chunks))]
                self.collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=chunk_metadatas,
                    ids=ids
                )

                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext == '.pdf':
                    file_counts["pdf"] += 1
                elif ext == '.docx':
                    file_counts["docx"] += 1
                elif ext in ('.xlsx', '.xls'):
                    file_counts["excel"] += 1
                elif ext == '.csv':
                    file_counts["csv"] += 1
                else:
                    file_counts["text"] += 1

                os.unlink(file_path)

            except Exception as e:
                logger.error("Error processing file %s: %s", uploaded_file.name, str(e))
                file_counts["errors"] += 1
                if os.path.exists(file_path):
                    os.unlink(file_path)
                continue

        logger.info("File processing summary: %s", file_counts)
        return file_counts

    def _track_performance(self, process: str, model: str, input_tokens: int, output_tokens: int, time_taken: float) -> PerformanceMetrics:
        """Track and store performance metrics."""
        try:
            metrics = PerformanceMetrics(
                process=process,
                model=model,
                tokens=input_tokens + output_tokens,
                time_taken=time_taken,
                tokens_per_sec=(input_tokens + output_tokens)/time_taken if time_taken > 0 else 0
            )
            self.performance_data.append(metrics)
            logger.info("Performance tracked for %s with %s: %s tokens/sec", process, model, metrics.tokens_per_sec)
            return metrics
        except Exception as e:
            logger.error("Error tracking performance for %s: %s", process, str(e))
            raise

    def detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            lang = detect(text[:500])
            logger.info("Detected language: %s", lang)
            return lang
        except Exception as e:
            logger.warning("Language detection failed: %s", str(e))
            return "unknown"

    def process_pdf(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process PDF file into chunks with metadata."""
        texts = []
        metadatas = []
        try:
            logger.info("Processing PDF: %s", file_path)
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        texts.append(text)
                        metadatas.append({
                            "source": os.path.basename(file_path),
                            "page": page_num,
                            "type": "pdf",
                            "position": f"page_{page_num}",
                            "language": self.detect_language(text)
                        })
            logger.info("Extracted %d pages from %s", len(texts), file_path)
        except Exception as e:
            logger.error("Error processing PDF %s: %s", file_path, str(e))
        return texts, metadatas

    def process_docx(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process DOCX file into chunks with metadata."""
        texts = []
        metadatas = []
        try:
            logger.info("Processing DOCX: %s", file_path)
            doc = Document(file_path)
            for para_num, para in enumerate(doc.paragraphs, 1):
                if para.text.strip():
                    texts.append(para.text)
                    metadatas.append({
                        "source": os.path.basename(file_path),
                        "paragraph": para_num,
                        "type": "docx",
                        "position": f"para_{para_num}",
                        "language": self.detect_language(para.text)
                    })
            logger.info("Extracted %d paragraphs from %s", len(texts), file_path)
        except Exception as e:
            logger.error("Error processing DOCX %s: %s", file_path, str(e))
        return texts, metadatas

    def process_excel(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process Excel/CSV file into chunks with metadata."""
        texts = []
        metadatas = []
        try:
            logger.info("Processing Excel/CSV: %s", file_path)
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.concat(pd.read_excel(file_path, sheet_name=None).values())
            text = df.to_markdown(index=False)
            if text.strip():
                texts.append(text)
                metadatas.append({
                    "source": os.path.basename(file_path),
                    "type": "excel" if file_path.endswith('.xlsx') else "csv",
                    "position": "full_content",
                    "language": self.detect_language(text)
                })
            logger.info("Extracted content from %s", file_path)
        except Exception as e:
            logger.error("Error processing Excel/CSV %s: %s", file_path, str(e))
        return texts, metadatas

    def process_file(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Route file to appropriate processor based on extension."""
        try:
            if not os.path.exists(file_path):
                logger.warning("File does not exist: %s", file_path)
                return [], []
            
            if file_path.endswith('.pdf'):
                return self.process_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self.process_docx(file_path)
            elif file_path.endswith(('.xlsx', '.xls', '.csv')):
                return self.process_excel(file_path)
            else:
                try:
                    logger.info("Processing text file: %s", file_path)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        return [content], [{
                            "source": os.path.basename(file_path),
                            "type": "text",
                            "position": "full_content",
                            "language": self.detect_language(content)
                        }]
                except Exception as e:
                    logger.error("Error processing text file %s: %s", file_path, str(e))
                    return [], []
        except Exception as e:
            logger.error("General error processing file %s: %s", file_path, str(e))
            return [], []

    def chunk_text(self, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Break down texts into token-limited chunks with rich metadata."""
        chunks = []
        chunk_metadatas = []
        global_chunk_counter = 1
        try:
            logger.info("Chunking %d texts", len(texts))
            for text, metadata in zip(texts, metadatas):
                try:
                    tokens = self.tokenizer.encode(text)
                    total_tokens = len(tokens)
                    num_chunks = (total_tokens // MAX_CHUNK_SIZE) + (1 if total_tokens % MAX_CHUNK_SIZE != 0 else 0)
                    for i in range(0, total_tokens, MAX_CHUNK_SIZE):
                        chunk_tokens = tokens[i:i+MAX_CHUNK_SIZE]
                        chunk_text = self.tokenizer.decode(chunk_tokens)
                        chunk_metadata = {
                            **metadata,
                            "chunk_number": global_chunk_counter,
                            "document_chunk_sequence": (i // MAX_CHUNK_SIZE) + 1,
                            "total_chunks_in_doc": num_chunks,
                            "token_count": len(chunk_tokens),
                            "start_token": i,
                            "end_token": min(i + MAX_CHUNK_SIZE, total_tokens) - 1,
                            "token_percentage": f"{(i + len(chunk_tokens)) / total_tokens * 100:.1f}%",
                            "is_continuation": i > 0
                        }
                        chunks.append(chunk_text)
                        chunk_metadatas.append(chunk_metadata)
                        global_chunk_counter += 1
                except Exception as e:
                    logger.error("Error chunking text: %s", str(e))
                    continue
            logger.info("Generated %d chunks", len(chunks))
        except Exception as e:
            logger.error("General error in chunk_text: %s", str(e))
        return chunks, chunk_metadatas

    def _retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant chunks from vector database."""
        try:
            query_embedding = ollama.embeddings(
                model=self.embedding_model,
                prompt=query
            )['embedding']
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results['documents'][0], results['metadatas'][0]
        except Exception as e:
            logger.error("Error retrieving chunks: %s", str(e))
            return [], []

    def translate_text(self, text: str, target_lang: str, context: str = None) -> Tuple[str, PerformanceMetrics]:
        """Translate text using the best model for translation."""
        start_time = time.time()
        model = ModelManager.get_best_model("translation")
        try:
            prompt = f"""Translate to {target_lang}. Preserve formatting and technical terms.
            {f"Context: {context}" if context else ""}
            Text: {text}"""
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={'temperature': 0.1}
            )
            translated = response['response']
            metrics = self._track_performance(
                f"translate_to_{target_lang}",
                model,
                len(self.tokenizer.encode(prompt)),
                len(self.tokenizer.encode(translated)),
                time.time() - start_time
            )
            return translated, metrics
        except Exception as e:
            logger.error("Translation error: %s", str(e))
            return f"Error: {str(e)}", None

    def translate_document(self, source_file: str, target_lang: str) -> Tuple[str, Optional[PerformanceMetrics]]:
        """Document translation using the best translation model."""
        start_time = time.time()
        model = ModelManager.get_best_model("translation")
        try:
            results = self.collection.get(
                where={"source": source_file},
                include=["documents", "metadatas"]
            )
            if not results['documents']:
                logger.warning("Document not found in database.")
                return "Document not found in database.", None
            
            translated_chunks = []
            total_input, total_output = 0, 0
            for doc, meta in zip(results['documents'], results['metadatas']):
                chunk, chunk_metrics = self.translate_text(
                    text=doc,
                    target_lang=target_lang,
                    context=f"Original language: {meta.get('language', 'unknown')}"
                )
                if chunk_metrics is None:
                    continue
                translated_chunks.append(chunk)
                total_input += chunk_metrics.tokens
                total_output += len(self.tokenizer.encode(chunk))
            
            translated_text = "\n\n".join(translated_chunks)
            metrics = PerformanceMetrics(
                process=f"translate_document_to_{target_lang}",
                model=model,
                tokens=total_input + total_output,
                time_taken=time.time() - start_time,
                tokens_per_sec=(total_input + total_output)/(time.time() - start_time) if (time.time() - start_time) > 0 else 0
            )
            logger.info("Document translated: %s to %s using %s", source_file, target_lang, model)
            return translated_text, metrics
        except Exception as e:
            logger.error("Translation error for %s: %s", source_file, str(e))
            return f"Error: {str(e)}", None

    def summarize_document(self, source_file: str, strategy: str = "abstractive") -> Tuple[str, dict, Optional[PerformanceMetrics]]:
        """Generate document summary with the best model for summarization."""
        start_time = time.time()
        model = ModelManager.get_best_model("summarization")
        try:
            results = self.collection.get(
                where={"source": source_file},
                include=["documents", "metadatas"]
            )
            if not results['documents']:
                logger.warning("Document not found: %s", source_file)
                return "Document not found", {}, None
            
            context = "\n\n".join(
                f"Source: {meta['source']} [{meta['position']}]\nContent: {doc}"
                for doc, meta in zip(results['documents'], results['metadatas'])
            )
            
            prompt = f"""Create {strategy} summary. Be concise and technical.
            Document: {context}
            Summary:"""
            
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={'temperature': 0.3}
            )
            summary = response['response']
            
            sample_ref = " ".join(results['documents'][:3])
            scores = self.scorer.score(sample_ref[:5000], summary) if sample_ref.strip() else {}
            
            metrics = self._track_performance(
                f"summarize_{strategy}",
                model,
                len(self.tokenizer.encode(prompt)),
                len(self.tokenizer.encode(summary)),
                time.time() - start_time
            )
            
            if scores:
                self.rouge_data.append(RougeMetrics(
                    source=source_file,
                    strategy=strategy,
                    model=model,
                    rouge1=scores['rouge1'].fmeasure,
                    rouge2=scores['rouge2'].fmeasure,
                    rougeL=scores['rougeL'].fmeasure,
                    timestamp=time.time()
                ))
            
            logger.info("Document summarized: %s with strategy %s using %s", source_file, strategy, model)
            return summary, scores, metrics
        except Exception as e:
            logger.error("Summarization error for %s: %s", source_file, str(e))
            return f"Error: {str(e)}", {}, None

    def ask_question(self, question: str, target_lang: str = None) -> Tuple[str, List[Dict], dict]:
        """Answer question using the best QA model with optional translation."""
        start_time = time.time()
        qa_model = ModelManager.get_best_model("qa")
        try:
            documents, metadatas = self._retrieve_relevant_chunks(question)
            if not documents:
                logger.warning("No relevant chunks found for question: %s", question)
                return "No relevant information found.", [], {}
            
            context = "\n\n".join(
                f"Source: {meta['source']} [{meta['position']}]\nContent: {doc}"
                for doc, meta in zip(documents, metadatas)
            )
            
            prompt = f"""Answer using this context. Be precise and cite sources.
            Context: {context}
            Question: {question}
            Answer:"""
            
            response = ollama.generate(
                model=qa_model,
                prompt=prompt,
                options={'temperature': 0.2}
            )
            answer = response['response']
            
            translation_metrics = None
            if target_lang:
                answer, translation_metrics = self.translate_text(
                    answer, target_lang, context
                )
            
            sources = [{
                "source": meta['source'],
                "position": meta['position'],
                "content": doc[:200] + "..."
            } for doc, meta in zip(documents, metadatas)]
            
            rag_metrics = self._track_performance(
                "rag_qa",
                qa_model,
                len(self.tokenizer.encode(prompt)),
                len(self.tokenizer.encode(answer)),
                time.time() - start_time
            )
            
            logger.info("Question answered: %s using %s", question, qa_model)
            return answer, sources, {
                "rag": rag_metrics,
                "translation": translation_metrics
            }
        except Exception as e:
            logger.error("Q&A error for question %s: %s", question, str(e))
            return f"Error: {str(e)}", [], {}

    def clear_database(self) -> bool:
        """Clear the vector database."""
        try:
            self.chroma_client.delete_collection("drx_research")
            self._init_vector_db()
            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error("Error clearing database: %s", str(e))
            return False

    def list_documents(self) -> List[Dict]:
        """List all documents in the database with counts."""
        try:
            results = self.collection.get(include=["metadatas"])
            unique_sources = {}
            for meta in results['metadatas']:
                source = meta['source']
                if source not in unique_sources:
                    unique_sources[source] = {
                        "type": meta['type'],
                        "language": meta.get('language', 'unknown'),
                        "chunks": 1
                    }
                else:
                    unique_sources[source]['chunks'] += 1
            logger.info("Listed %d unique documents", len(unique_sources))
            return [{"source": k, **v} for k, v in unique_sources.items()]
        except Exception as e:
            logger.error("Error listing documents: %s", str(e))
            return []

    def get_performance_data(self) -> List[PerformanceMetrics]:
        """Get all collected performance metrics."""
        return self.performance_data

    def get_rouge_data(self) -> List[Dict]:
        """Get ROUGE metrics as a list of dictionaries."""
        return [{
            "Source": r.source,
            "Strategy": r.strategy,
            "Model": r.model,
            "ROUGE-1": f"{r.rouge1:.3f}",
            "ROUGE-2": f"{r.rouge2:.3f}",
            "ROUGE-L": f"{r.rougeL:.3f}",
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp))
        } for r in self.rouge_data]

    def get_model_comparison(self, task: str) -> str:
        """Get model comparison table for a specific task."""
        return self.model_comparisons.get(task, "No comparison available")

    def reset_performance_data(self):
        """Clear performance metrics."""
        self.performance_data = []
        logger.info("Performance data reset")

def main():
    st.set_page_config(
        page_title="Dr. X Research Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Dark mode CSS
    st.markdown("""
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #7c3aed;
            --accent: #f59e0b;
            --dark: #1e293b;
            --light: #f8fafc;
            --background: #0f172a;
            --text: #e2e8f0;
            --border: #334155;
        }

        .stApp {
            background-color: var(--background);
            color: var(--text);
        }

        .stButton>button {
            background-color: var(--primary);
            color: var(--light);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
        }

        .stButton>button:hover {
            background-color: #1d4ed8;
        }

        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>select {
            background-color: #1e293b;
            color: var(--text);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
        }

        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3 {
            color: var(--light);
        }

        .metric-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            color: var(--text);
        }

        .answer-card {
            background-color: #1e293b;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }

        .stDataFrame {
            background-color: #1e293b;
            color: var(--text);
            border-radius: 8px;
        }

        .stDataFrame table {
            background-color: #1e293b;
            color: var(--text);
        }

        .stSpinner > div > div {
            border-top-color: var(--primary) !important;
        }

        .model-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid var(--accent);
        }

        .model-name {
            color: var(--accent);
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .best-model {
            border-left: 4px solid #10b981;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize processor in session state
    if 'processor' not in st.session_state:
        st.session_state.processor = ResearchProcessor()
        st.session_state.documents_updated = False

    processor = st.session_state.processor

    st.title("🔍 Dr. X Research Analyzer")
    st.markdown("Advanced AI-powered tool for processing, analyzing, and visualizing research documents")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📂 Upload Documents",
        "📝 Ask Questions",
        "🌍 Translate Documents",
        "✂️ Summarize Documents",
        "📊 Dashboard",
        "🤖 Model Info"
    ])

    with tab1:
        st.header("Upload Research Documents")
        uploaded_files = st.file_uploader(
            "Select files to analyze (PDF, CSV, Excel, DOCX)",
            type=["pdf", "csv", "xlsx", "xls", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if st.button("Process Files", key="upload_btn"):
            if uploaded_files:
                with st.spinner("Processing files..."):
                    file_counts = processor.add_uploaded_files(uploaded_files)
                    if file_counts["errors"] > 0:
                        st.warning(f"Processed {len(uploaded_files)} file(s) with {file_counts['errors']} error(s). Check research_processor.log for details.")
                    else:
                        st.success(f"Processed {len(uploaded_files)} file(s) successfully")
                    
                    cols = st.columns(3)
                    for i, (k, v) in enumerate(file_counts.items()):
                        if v > 0:
                            cols[i%3].metric(label=k.capitalize(), value=v)
                    
                    st.session_state.documents_updated = True

        st.subheader("Documents in Database")
        docs = processor.list_documents()
        if docs:
            st.dataframe(
                pd.DataFrame([{
                    "Document": doc['source'],
                    "Type": doc['type'],
                    "Language": doc['language'],
                    "Chunks": doc['chunks']
                } for doc in docs]),
                use_container_width=True
            )
        else:
            st.info("No documents available")

    with tab2:
        st.header("Research Q&A System")
        col1, col2 = st.columns([3, 1])

        with col1:
            question = st.text_input(
                "Ask about Dr. X's research",
                placeholder="What was the main finding about...?",
                key="question_input"
            )
        with col2:
            target_lang = st.selectbox(
                "Translate answer to",
                [lang.value for lang in TranslationLanguage],
                index=0,
                key="qa_lang"
            )

        if st.button("Submit Question", key="ask_btn"):
            if question:
                with st.spinner("Analyzing research..."):
                    answer, sources, metrics = processor.ask_question(
                        question,
                        target_lang if target_lang != TranslationLanguage.NONE.value else None
                    )

                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                    if sources:
                        st.markdown("### Source Documents")
                        for src in sources:
                            with st.expander(f"📄 {src['source']} [{src['position']}]"):
                                st.text(src['content'])

                    if metrics:
                        st.markdown("### Performance Metrics")
                        st.json(metrics)
            else:
                st.warning("Please enter a question")

    with tab3:
        st.header("Document Translation")
        docs = processor.list_documents()

        if docs:
            col1, col2 = st.columns(2)

            with col1:
                source_file = st.selectbox(
                    "Select document to translate",
                    [doc['source'] for doc in docs],
                    key="trans_file"
                )
            with col2:
                target_lang = st.selectbox(
                    "Target language",
                    [lang.value for lang in TranslationLanguage if lang != TranslationLanguage.NONE],
                    key="trans_lang"
                )

            if st.button("Translate Document", key="trans_btn"):
                with st.spinner("Translating document..."):
                    translated, metrics = processor.translate_document(source_file, target_lang)

                    st.markdown("### Translation Result")
                    st.markdown(f'<div class="answer-card">{translated[:2000]}</div>', unsafe_allow_html=True)

                    if len(translated) > 2000:
                        st.warning("Displaying first 2000 characters")

                    if metrics:
                        st.markdown("### Performance Metrics")
                        st.json({
                            "Process": metrics.process,
                            "Model": metrics.model,
                            "Tokens": metrics.tokens,
                            "Time (s)": f"{metrics.time_taken:.2f}",
                            "Tokens/Sec": f"{metrics.tokens_per_sec:.2f}"
                        })
        else:
            st.warning("No documents available for translation")

    with tab4:
        st.header("Document Summarization")
        docs = processor.list_documents()

        if docs:
            col1, col2 = st.columns(2)

            with col1:
                                source_file = st.selectbox(
                    "Select document to summarize",
                    [doc['source'] for doc in docs],
                    key="sum_file"
                )
            with col2:
                strategy = st.selectbox(
                    "Summarization strategy",
                    [s.value for s in SummaryStrategy],
                    key="sum_strategy"
                )

            if st.button("Summarize Document", key="sum_btn"):
                with st.spinner("Summarizing document..."):
                    summary, rouge_scores, metrics = processor.summarize_document(
                        source_file,
                        strategy
                    )

                    st.markdown("### Summary Result")
                    st.markdown(f'<div class="answer-card">{summary}</div>', unsafe_allow_html=True)

                    if rouge_scores:
                        st.markdown("### ROUGE Scores")
                        st.json({
                            "ROUGE-1": f"{rouge_scores.get('rouge1', {}).get('fmeasure', 0):.3f}",
                            "ROUGE-2": f"{rouge_scores.get('rouge2', {}).get('fmeasure', 0):.3f}",
                            "ROUGE-L": f"{rouge_scores.get('rougeL', {}).get('fmeasure', 0):.3f}"
                        })

                    if metrics:
                        st.markdown("### Performance Metrics")
                        st.json({
                            "Process": metrics.process,
                            "Model": metrics.model,
                            "Tokens": metrics.tokens,
                            "Time (s)": f"{metrics.time_taken:.2f}",
                            "Tokens/Sec": f"{metrics.tokens_per_sec:.2f}"
                        })
        else:
            st.warning("No documents available for summarization")

    with tab5:
        st.header("Performance Dashboard")
        st.markdown("Visualize processing metrics and model performance")

        # Performance Metrics Visualization
        perf_data = processor.get_performance_data()
        if perf_data:
            perf_df = pd.DataFrame([{
                "Process": p.process,
                "Model": p.model,
                "Tokens": p.tokens,
                "Time (s)": p.time_taken,
                "Tokens/Sec": p.tokens_per_sec
            } for p in perf_data])

            st.subheader("Processing Performance")
            fig = px.bar(
                perf_df,
                x="Process",
                y="Tokens/Sec",
                color="Model",
                title="Processing Speed by Task and Model",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ROUGE Scores Visualization
            rouge_data = processor.get_rouge_data()
            if rouge_data:
                rouge_df = pd.DataFrame(rouge_data)
                st.subheader("Summarization Quality (ROUGE Scores)")
                fig = px.line(
                    rouge_df,
                    x="Timestamp",
                    y=["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                    title="ROUGE Scores Over Time",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Clear Performance Data
            if st.button("Reset Performance Data", key="reset_perf"):
                processor.reset_performance_data()
                st.success("Performance data cleared")
                st.experimental_rerun()
        else:
            st.info("No performance data available")

    with tab6:
        st.header("Model Information")
        st.markdown("Compare available models and their performance characteristics")

        task = st.selectbox(
            "Select task to compare models",
            ["translation", "summarization", "qa", "embeddings"],
            key="model_task"
        )

        st.markdown(f"### Model Comparison for {task.capitalize()}")
        comparison_table = processor.get_model_comparison(task)
        st.markdown(comparison_table)

        # Display model details
        st.subheader("Model Details")
        options = ModelManager.get_model_options(task) if task != "embeddings" else ModelManager.get_embedding_options()
        for model, specs in options.items():
            with st.container():
                st.markdown(f'<div class="model-card {"best-model" if model == ModelManager.get_best_model(task) else ""}">', unsafe_allow_html=True)
                st.markdown(f'<div class="model-name">{model} {"⭐ Best" if model == ModelManager.get_best_model(task) else ""}</div>', unsafe_allow_html=True)
                st.markdown(f"**Description**: {specs['description']}")
                st.markdown(f"**Performance Score**: {specs['performance']:.2f}")
                st.markdown(f"**Speed**: {specs['speed']}")
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()