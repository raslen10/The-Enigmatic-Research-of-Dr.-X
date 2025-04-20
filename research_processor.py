import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import pandas as pd
import docx
import ollama
import tiktoken
from rouge_score import rouge_scorer
from langdetect import detect

# Configuration
DATA_FOLDER = "data"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
MAX_CHUNK_SIZE = 500  # tokens
MAX_WORKERS = 4

@dataclass
class PerformanceMetrics:
    process: str
    tokens: int
    time_taken: float
    tokens_per_sec: float

class ResearchProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.performance_data = []
        self._init_vector_db()
        self._validate_config()
    
    def _init_vector_db(self):
        """Initialize the ChromaDB vector database."""
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.nomic_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="drx_research",
            embedding_function=self.nomic_ef,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _validate_config(self):
        """Validate and create required directories."""
        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    def _track_performance(self, process: str, input_tokens: int, output_tokens: int, time_taken: float) -> PerformanceMetrics:
        """Track and store performance metrics."""
        metrics = PerformanceMetrics(
            process=process,
            tokens=input_tokens + output_tokens,
            time_taken=time_taken,
            tokens_per_sec=(input_tokens + output_tokens)/time_taken if time_taken > 0 else 0
        )
        self.performance_data.append(metrics)
        return metrics
    
    def detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            return detect(text[:500])
        except:
            return "unknown"
    
    def _validate_document(self, file_path: str) -> bool:
        """Validate if file is supported and accessible."""
        return (os.path.exists(file_path) and \
               file_path.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt')))
    
    def process_pdf(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process PDF file into chunks with metadata."""
        texts = []
        metadatas = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
                        metadatas.append({
                            "source": os.path.basename(file_path),
                            "page": page_num,
                            "type": "pdf",
                            "position": f"page_{page_num}",
                            "language": self.detect_language(text)
                        })
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
        
        return texts, metadatas
    
    def process_docx(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process DOCX file into chunks with metadata."""
        texts = []
        metadatas = []
        
        try:
            doc = docx.Document(file_path)
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
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {str(e)}")
        
        return texts, metadatas
    
    def process_excel(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Process Excel/CSV file into chunks with metadata."""
        texts = []
        metadatas = []
        
        try:
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
        except Exception as e:
            print(f"Error processing Excel {file_path}: {str(e)}")
        
        return texts, metadatas
    
    def process_file(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Route file to appropriate processor based on extension."""
        if not self._validate_document(file_path):
            return [], []
            
        if file_path.endswith('.docx'):
            return self.process_pdf(file_path)
        #elif file_path.endswith('.pdf'):
        #    return self.process_docx(file_path)
        #elif file_path.endswith(('.xlsx', '.xls', '.csv')):
        #    return self.process_excel(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    return [content], [{
                        "source": os.path.basename(file_path),
                        "type": "text",
                        "position": "full_content",
                        "language": self.detect_language(content)
                    }]
            except Exception as e:
                print(f"Error processing text file {file_path}: {str(e)}")
                return [], []
    
    def chunk_text(self, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Break down texts into token-limited chunks with rich metadata."""
        chunks = []
        chunk_metadatas = []
        global_chunk_counter = 1
        
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
                print(f"Error chunking text: {str(e)}")
                continue
        
        return chunks, chunk_metadatas

    def load_documents(self, data_folder: str = DATA_FOLDER) -> Dict[str, int]:
        """Load and process all documents in the data folder."""
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder '{data_folder}' not found")
        
        file_counts = {
            "pdf": 0, "docx": 0, "excel": 0, 
            "csv": 0, "text": 0, "errors": 0
        }
        
        for root, _, files in tqdm(os.walk(data_folder), desc="Processing folders"):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    texts, metadatas = self.process_file(file_path)
                    if not texts:
                        continue
                    
                    chunks, chunk_metadatas = self.chunk_text(texts, metadatas)
                    
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        embeddings = list(executor.map(
                            lambda chunk: ollama.embeddings(
                                model=EMBEDDING_MODEL,
                                prompt=chunk
                            )['embedding'],
                            chunks
                        ))
                    
                    ids = [f"{os.path.basename(file_path)}-{i}" for i in range(len(chunks))]
                    self.collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        metadatas=chunk_metadatas,
                        ids=ids
                    )
                    
                    # Update counts
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.pdf': file_counts["pdf"] += 1
                    elif ext == '.docx': file_counts["docx"] += 1
                    elif ext in ('.xlsx', '.xls'): file_counts["excel"] += 1
                    elif ext == '.csv': file_counts["csv"] += 1
                    else: file_counts["text"] += 1
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    file_counts["errors"] += 1
        
        return file_counts
    
    def _retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant chunks from vector database."""
        try:
            query_embedding = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=query
            )['embedding']
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return results['documents'][0], results['metadatas'][0]
        except Exception as e:
            print(f"Error retrieving chunks: {str(e)}")
            return [], []
    
    def translate_text(self, text: str, target_lang: str, context: str = None) -> Tuple[str, PerformanceMetrics]:
        """Unified translation method for both documents and arbitrary text."""
        start_time = time.time()
        prompt = f"""Translate to {target_lang}. Preserve formatting and technical terms.
        {f"Context: {context}" if context else ""}
        Text: {text}"""
        
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={'temperature': 0.1}
        )
        translated = response['response']
        
        metrics = self._track_performance(
            f"translate_to_{target_lang}",
            len(self.tokenizer.encode(prompt)),
            len(self.tokenizer.encode(translated)),
            time.time() - start_time
        )
        
        return translated, metrics

    def translate_document(self, source_file: str, target_lang: str) -> Tuple[str, PerformanceMetrics]:
        """Document translation using the unified translate_text method."""
        start_time = time.time()
        
        try:
            results = self.collection.get(
                where={"source": source_file},
                include=["documents", "metadatas"]
            )
            
            if not results['documents']:
                return "Document not found in database.", None
            
            translated_chunks = []
            total_input, total_output = 0, 0
            
            for doc, meta in zip(results['documents'], results['metadatas']):
                chunk, chunk_metrics = self.translate_text(
                    text=doc,
                    target_lang=target_lang,
                    context=f"Original language: {meta.get('language', 'unknown')}"
                )
                translated_chunks.append(chunk)
                total_input += chunk_metrics.tokens
                total_output += len(self.tokenizer.encode(chunk))
            
            translated_text = "\n\n".join(translated_chunks)
            
            metrics = PerformanceMetrics(
                process=f"translate_document_to_{target_lang}",
                tokens=total_input + total_output,
                time_taken=time.time() - start_time,
                tokens_per_sec=(total_input + total_output)/(time.time() - start_time)
            )
            
            return translated_text, metrics
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Error: {str(e)}", None
    
    def summarize_document(self, source_file: str, strategy: str = "abstractive") -> Tuple[str, dict, PerformanceMetrics]:
        """Generate document summary with quality metrics."""
        start_time = time.time()
        
        try:
            results = self.collection.get(
                where={"source": source_file},
                include=["documents", "metadatas"]
            )
            
            if not results['documents']:
                return "Document not found", {}, None
            
            context = "\n\n".join(
                f"Source: {meta['source']} [{meta['position']}]\nContent: {doc}"
                for doc, meta in zip(results['documents'], results['metadatas'])
            )
            
            prompt = f"""Create {strategy} summary. Be concise and technical.
            Document: {context}
            Summary:"""
            
            response = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={'temperature': 0.3}
            )
            summary = response['response']
            
            # Calculate ROUGE scores safely
            scores = {}
            sample_ref = " ".join(results['documents'][:3])
            if sample_ref and summary:  # Only calculate if we have reference and summary
                try:
                    scores = self.scorer.score(sample_ref[:5000], summary)
                except Exception as e:
                    print(f"ROUGE calculation error: {str(e)}")
                    scores = {}
            
            metrics = self._track_performance(
                f"summarize_{strategy}",
                len(self.tokenizer.encode(prompt)),
                len(self.tokenizer.encode(summary)),
                time.time() - start_time
            )
            
            return summary, scores, metrics
            
        except Exception as e:
            print(f"Summarization error: {str(e)}")
        return f"Error: {str(e)}", {}, None
    def ask_question(self, question: str, target_lang: str = None) -> Tuple[str, List[Dict], dict]:
        """Answer question using document context with optional translation."""
        start_time = time.time()
        
        try:
            documents, metadatas = self._retrieve_relevant_chunks(question)
            if not documents:
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
                model=LLM_MODEL,
                prompt=prompt,
                options={'temperature': 0.2}
            )
            answer = response['response']
            
            # Translation if needed
            translation_metrics = None
            if target_lang:
                answer, translation_metrics = self.translate_text(
                    answer, target_lang, context
                )
            
            # Prepare sources
            sources = [{
                "source": meta['source'],
                "position": meta['position'],
                "content": doc[:200] + "..."
            } for doc, meta in zip(documents, metadatas)]
            
            # Track performance
            rag_metrics = self._track_performance(
                "rag_qa",
                len(self.tokenizer.encode(prompt)),
                len(self.tokenizer.encode(answer)),
                time.time() - start_time
            )
            
            return answer, sources, {
                "rag": rag_metrics,
                "translation": translation_metrics
            }
            
        except Exception as e:
            print(f"Q&A error: {str(e)}")
            return f"Error: {str(e)}", [], {}

    def clear_database(self) -> bool:
        """Clear the vector database."""
        try:
            self.chroma_client.delete_collection("drx_research")
            self._init_vector_db()
            return True
        except Exception as e:
            print(f"Error clearing database: {str(e)}")
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
                    
            return [{"source": k, **v} for k, v in unique_sources.items()]
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []

    def get_performance_data(self) -> List[PerformanceMetrics]:
        """Get all collected performance metrics."""
        return self.performance_data

    def reset_performance_data(self):
        """Clear performance metrics."""
        self.performance_data = []