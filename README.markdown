# Dr. X Research Analyzer

Dr. X Research Analyzer is an advanced AI-powered Streamlit application designed for processing, analyzing, and visualizing research documents. It supports multiple file formats (PDF, DOCX, Excel, CSV), offers multilingual translation, summarization, question-answering, and performance visualization, leveraging local LLMs via Ollama and vector storage with ChromaDB.

## Table of Contents
- [Features](#features)
- [Workflow Diagram](#workflow-diagram)
- [Setup Instructions](#setup-instructions)
  - [Local Setup](#local-setup)
  - [Google Colab Setup](#google-colab-setup)
- [Model Selection and Comparison](#model-selection-and-comparison)
- [Running the Application](#running-the-application)
  - [Locally](#running-locally)
  - [On Google Colab](#running-on-google-colab)
- [Project Structure](#project-structure)
- [Optimization Notes](#optimization-notes)
- [Downloading Files](#downloading-files)
- [License](#license)

## Features
- **Document Processing**: Supports PDF, DOCX, Excel, and CSV files with chunking and metadata extraction.
- **Vector Storage**: Uses ChromaDB for efficient document retrieval.
- **Multilingual Translation**: Translates documents and answers into multiple languages (Arabic, French, Spanish, etc.).
- **Summarization**: Provides abstractive and extractive summarization with ROUGE score evaluation.
- **Question Answering**: Context-aware Q&A with source citation.
- **Performance Dashboard**: Visualizes processing speed and ROUGE scores using Plotly.
- **Model Comparison**: Displays performance and speed metrics for available models.
- **Dark Theme UI**: Streamlit-based interface with custom CSS for a modern look.

## Workflow Diagram

The following diagram illustrates the workflow from file upload to task execution in the Dr. X Research Analyzer. It covers the steps of document processing, vector storage, model inference, and result display.

### Mermaid Diagram
```mermaid
graph TD
    A[User Uploads Files<br>PDF, DOCX, Excel, CSV] --> B[Streamlit Interface]
    B --> C[ResearchProcessor]
    C --> D{File Type Processor}
    D -->|PDF| E[PyPDF2 Extractor]
    D -->|DOCX| F[python-docx Extractor]
    D -->|Excel/CSV| G[pandas Extractor]
    E --> H[Text Chunks]
    F --> H
    G --> H
    H --> I[Tokenizer<br>tiktoken]
    I --> J[Chunk Metadata<br>Source, Position, Language]
    J --> K[Embedding Generation<br>nomic-embed-text]
    K --> L[ChromaDB<br>Vector Storage]
    L --> M{Task Selection}
    M -->|Translation| N[Mixtral Model<br>Multilingual Translation]
    M -->|Summarization| O[Llama3 Model<br>Abstractive/Extractive]
    M -->|Question Answering| P[Llama3 Model<br>Context-Aware QA]
    N --> Q[Performance Metrics<br>Tokens, Time, Speed]
    O --> Q
    P --> Q
    O --> R[ROUGE Scores<br>rouge1, rouge2, rougeL]
    Q --> S[Streamlit Dashboard<br>Plotly Visualizations]
    R --> S
    N --> T[Result Display<br>Translated Text]
    O --> U[Result Display<br>Summary]
    P --> V[Result Display<br>Answer with Sources]
    T --> B
    U --> B
    V --> B
```

### Diagram Explanation
1. **File Upload**: Users upload files (PDF, DOCX, Excel, CSV) via the Streamlit interface.
2. **File Processing**:
   - The `ResearchProcessor` routes files to specific extractors (`PyPDF2` for PDF, `python-docx` for DOCX, `pandas` for Excel/CSV).
   - Extracted text is tokenized using `tiktoken` and split into chunks (max 500 tokens).
   - Metadata (source, position, language) is generated for each chunk.
3. **Embedding and Storage**:
   - Chunks are embedded using `nomic-embed-text` via Ollama.
   - Embeddings and metadata are stored in ChromaDB for efficient retrieval.
4. **Task Execution**:
   - **Translation**: Uses `mixtral` to translate chunks or answers into the target language.
   - **Summarization**: Uses `llama3` for abstractive or extractive summarization, with ROUGE scores computed.
   - **Question Answering**: Uses `llama3` for context-aware answers, retrieving relevant chunks from ChromaDB.
5. **Metrics and Visualization**:
   - Performance metrics (tokens, time, tokens/sec) are tracked for all tasks.
   - ROUGE scores are computed for summarization.
   - Results and metrics are visualized in a Streamlit dashboard using Plotly.
6. **Result Display**:
   - Translated text, summaries, or answers (with sources) are displayed in the Streamlit UI.

To view the diagram, copy the Mermaid code into [Mermaid Live Editor](https://mermaid.live/) or render it in a Mermaid-compatible platform like GitHub.

## Setup Instructions

### Local Setup
To run the application locally, you need to set up Ollama, install dependencies, and configure the environment. Note that a GPU is not required, as Ollama can run on CPU, but performance may be slower.

1. **Install Dependencies**:
   ```bash
   pip install streamlit pandas numpy chromadb PyPDF2 python-docx ollama tiktoken rouge-score langdetect plotly torch tqdm
   ```

2. **Install Ollama**:
   - Download and install Ollama from [ollama.com](https://ollama.com).
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Pull the required models (see [Model Selection](#model-selection-and-comparison)):
     ```bash
     ollama pull nomic-embed-text
     ollama pull mixtral
     ollama pull llama3
     ollama pull mistral
     ```

3. **Directory Setup**:
   - Create a `data` folder for uploaded files.
   - Create a `chroma_db` folder for ChromaDB persistence.
   ```bash
   mkdir data chroma_db
   ```

4. **Verify Ollama**:
   - Ensure Ollama is running on `http://localhost:11434`.
   - Test model availability:
     ```bash
     ollama list
     ```

### Google Colab Setup
Since you mentioned lacking a GPU locally, Google Colab is an excellent choice for running the application with GPU acceleration. The setup involves running Ollama in Colab and installing dependencies.

1. **Create a Colab Notebook**:
   - Open Google Colab and create a new notebook.
   - Enable GPU (optional): Go to `Runtime` > `Change runtime type` > Select `GPU`.

2. **Install Dependencies**:
   Run the following in a Colab cell:
   ```bash
   !pip install streamlit pandas numpy chromadb PyPDF2 python-docx ollama tiktoken rouge-score langdetect plotly torch tqdm
   ```

3. **Install Ollama in Colab**:
   To run the Dr. X Research Analyzer in Google Colab, you need to install and configure Ollama to serve local language models. The following steps guide you through setting up Ollama in a Colab environment, including starting the server and pulling the required models. These steps are designed to work in Colab's cloud-based environment, leveraging GPU acceleration if enabled.

   **Step 1: Install colab-xterm for Terminal Access**  
   Colab does not provide a native terminal, so we use `colab-xterm` to open a terminal session within the notebook. This allows us to run shell commands interactively.

   ```bash
   !pip install colab-xterm
   ```

   **Step 2: Load the colab-xterm Extension**  
   Load the `colab-xterm` extension to enable terminal functionality in Colab. This step prepares the notebook to open a terminal window.

   ```bash
   %load_ext colabxterm
   ```

   **Step 3: Open a Terminal Session**  
   Launch a terminal session in Colab. This will display an interactive terminal where you can execute commands to install and start Ollama.

   ```bash
   %xterm
   ```

   In the terminal that appears, run the following commands to install Ollama and start the server:

   ```bash
   curl https://ollama.ai/install.sh | sh
   ollama serve
   ```

   **Note**: After starting the Ollama server, keep the terminal session running in the background. Do not close the terminal window, as it will stop the server. You can minimize it and proceed with the next steps in the notebook.

   **Step 4: Pull the Required Models**  
   Download the necessary Ollama models (`nomic-embed-text`, `mixtral`, `llama3`, `mistral`) to your Colab environment. These models are used for embedding, translation, summarization, and question answering.

   ```bash
   !ollama pull nomic-embed-text
   !ollama pull mixtral
   !ollama pull llama3
   !ollama pull mistral
   ```

   **Step 5: Verify Ollama Server**  
   Confirm that the Ollama server is running and the models are available. This step ensures that the server is accessible at `http://localhost:11434`.

   ```bash
   !ollama list
   ```

   **Troubleshooting Tip**: If the `ollama list` command fails or shows no models, ensure the terminal session from Step 3 is still active and the `ollama serve` command is running. If the server has stopped, restart it in the terminal with `ollama serve`.

4. **Set Up Streamlit with Localtunnel**:
   To expose the Streamlit app in Colab, use `localtunnel` to create a public URL. This requires installing `localtunnel` and running Streamlit in the background.

   **Step 1: Get External IP Address**  
   Retrieve the external IP address of your Colab instance, which may be required for `localtunnel` authentication.

   ```bash
   !wget -q -O - ipv4.icanhazip.com
   ```

   **Step 2: Install Localtunnel**  
   Install the `localtunnel` package using npm to enable tunneling of the Streamlit app.

   ```bash
   !npm install localtunnel
   ```

   **Step 3: Run Streamlit in Background**  
   Start the Streamlit app in the background, redirecting logs to a file. Adjust the path to `app.py` if it’s stored elsewhere (e.g., `/content/app.py`).

   ```bash
   !streamlit run /content/drive/MyDrive/Osos_Project/app.py &>/content/logs.txt &
   ```

   **Step 4: Expose Streamlit with Localtunnel**  
   Use `localtunnel` to expose the Streamlit app on port 8501. When prompted, copy the external IP from Step 1 and paste it into the `localtunnel` interface.

   ```bash
   !npx localtunnel --port 8501
   ```

   **Note**: After running the `localtunnel` command, a URL (e.g., `https://<random>.loca.lt`) will be generated. Visit this URL, and when prompted, enter the external IP address from Step 1 to access the Streamlit app.

5. **Directory Setup**:
   - Create required directories for data and ChromaDB persistence:
     ```bash
     !mkdir -p data chroma_db
     ```

6. **Upload the Code**:
   - Save the provided code as `app.py` and upload it to Colab. If stored in Google Drive, ensure the path matches the one used in the Streamlit command (e.g., `/content/drive/MyDrive/Osos_Project/app.py`).
     ```python
     from google.colab import files
     files.upload()  # Upload app.py
     ```

## Model Selection and Comparison

The application uses Ollama to run local LLMs and embedding models. Below is the rationale for the chosen models and a comparison based on the `MODEL_CONFIG` in the code.

### Embedding Model
- **Chosen**: `nomic-embed-text`
- **Why**:
  - **Performance**: High score (0.92) for general-purpose embeddings.
  - **Speed**: Fast, suitable for real-time document processing.
  - **Dimension**: 768, balancing quality and efficiency.
- **Alternatives**:
  - `llama2` (4096 dim, 0.89 performance, medium speed): Higher dimensionality but slower and less performant.

**Comparison**:
| Model           | Description                     | Performance | Speed  |
|-----------------|---------------------------------|-------------|--------|
| **nomic-embed-text** ⭐ | General purpose embeddings | 0.92        | Fast   |
| llama2          | Llama 2 embeddings             | 0.89        | Medium |

### Translation Model
- **Chosen**: `mixtral`
- **Why**:
  - **Performance**: Best for multilingual tasks (0.95).
  - **Versatility**: Mixture of Experts model excels in preserving technical terms and formatting.
- **Alternatives**:
  - `llama3` (0.91, fast): Good balance but less accurate for complex translations.
  - `mistral` (0.88, fast): Efficient but lower performance.

**Comparison**:
| Model       | Description                          | Performance | Speed  |
|-------------|--------------------------------------|-------------|--------|
| **mixtral** ⭐ | Mixture of Experts, multilingual | 0.95        | Medium |
| llama3      | Meta's latest, balanced             | 0.91        | Fast   |
| mistral     | Efficient 7B model                 | 0.88        | Fast   |

### Summarization Model
- **Chosen**: `llama3`
- **Why**:
  - **Performance**: Best for abstractive summarization (0.94).
  - **Balance**: Medium speed with high-quality summaries.
- **Alternatives**:
  - `mistral` (0.89, fast): Better for extractive summarization but less accurate.
  - `gemma` (0.92, medium): Good for technical summaries but not the best overall.

**Comparison**:
| Model       | Description                          | Performance | Speed  |
|-------------|--------------------------------------|-------------|--------|
| **llama3** ⭐ | Best for abstractive summarization | 0.94        | Medium |
| mistral     | Good for extractive summarization   | 0.89        | Fast   |
| gemma       | Good for technical summaries        | 0.92        | Medium |

### Question-Answering Model
- **Chosen**: `llama3`
- **Why**:
  - **Performance**: Best for context-aware Q&A (0.93).
  - **Reliability**: Handles research-oriented questions well.
- **Alternatives**:
  - `command-r` (0.91, fast): Good for research but slightly less accurate.
  - `mixtral` (0.90, medium): Suitable for complex questions but slower.

**Comparison**:
| Model       | Description                          | Performance | Speed  |
|-------------|--------------------------------------|-------------|--------|
| **llama3** ⭐ | Best for question answering       | 0.93        | Medium |
| command-r   | Good for research-oriented QA      | 0.91        | Fast   |
| mixtral     | Good for complex questions         | 0.90        | Medium |

### Why These Models?
- **Local Compatibility**: All models run efficiently on CPU (local) or GPU (Colab) via Ollama.
- **Performance vs. Speed**: Chosen models balance high performance with reasonable speed, critical for real-time interaction in Streamlit.
- **Task-Specific Optimization**: Each model is selected based on its strengths for specific tasks (e.g., `mixtral` for translation, `llama3` for summarization and Q&A).

## Running the Application

### Running Locally
1. Ensure Ollama is running (`ollama serve`).
2. Place the code in a file named `app.py`.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Access the app at `http://localhost:8501`.

### Running on Google Colab
1. Follow the [Colab Setup](#google-colab-setup) steps to install dependencies, Ollama, and `localtunnel`.
2. Ensure `app.py` is uploaded to the correct path (e.g., `/content/drive/MyDrive/Osos_Project/app.py`).
3. Run the following commands to start the Streamlit app and expose it via `localtunnel`:

   **Step 1: Get External IP Address**  
   Retrieve the external IP address of your Colab instance, which may be required for `localtunnel` authentication.

   ```bash
   !wget -q -O - ipv4.icanhazip.com
   ```

   **Step 2: Install Localtunnel**  
   Install the `localtunnel` package using npm to enable tunneling of the Streamlit app.

   ```bash
   !npm install localtunnel
   ```

   **Step 3: Run Streamlit in Background**  
   Start the Streamlit app in the background, redirecting logs to a file. Adjust the path to `app.py` if it’s stored elsewhere.

   ```bash
   !streamlit run /content/drive/MyDrive/Osos_Project/app.py &>/content/logs.txt &
   ```

   **Step 4: Expose Streamlit with Localtunnel**  
   Use `localtunnel` to expose the Streamlit app on port 8501. When prompted, copy the external IP from Step 1 and paste it into the `localtunnel` interface.

   ```bash
   !npx localtunnel --port 8501
   ```

4. Visit the generated `localtunnel` URL (e.g., `https://<random>.loca.lt`). When prompted, enter the external IP address from Step 1 to access the Streamlit app.

**Note**: Colab's free tier has resource limits. For heavy processing, consider Colab Pro for better GPU access. If the `localtunnel` URL fails, ensure the Streamlit server is running (`!ps aux | grep streamlit`) and restart the tunneling command.

## Project Structure
The provided code is a single file (`app.py`) for Colab compatibility, as it simplifies uploading and running in a notebook environment. However, for better organization in a local or production setup, the project can be split into the following structure:

```
drx_research_analyzer/
├── app.py                    # Main Streamlit app
├── src/
│   ├── __init__.py
│   ├── processor.py         # ResearchProcessor and ModelManager classes
│   ├── config.py           # MODEL_CONFIG and other constants
│   ├── utils.py            # Helper functions (e.g., detect_language, chunk_text)
├── data/                    # Folder for uploaded files
├── chroma_db/               # ChromaDB persistence
├── requirements.txt         # Dependencies
├── research_processor.log   # Log file
└── README.md               # This file
```

### Why Single File for Colab?
- **Simplicity**: Uploading one file to Colab is faster and avoids managing multiple files in a notebook.
- **Portability**: A single file is easier to share and run without directory setup.
- **Colab Constraints**: Colab's file system is temporary, making a single file more practical.

### How to Split into Sub-files
To modularize the code:
1. Move `MODEL_CONFIG`, `PerformanceMetrics`, `RougeMetrics`, `TranslationLanguage`, and `SummaryStrategy` to `config.py`.
2. Move `ResearchProcessor` and `ModelManager` to `processor.py`.
3. Extract helper methods (e.g., `process_pdf`, `process_docx`) to `utils.py`.
4. Update `app.py` to import from `src`:
   ```python
   from src.processor import ResearchProcessor
   from src.config import MODEL_CONFIG, TranslationLanguage, SummaryStrategy
   from src.utils import process_pdf, process_docx
   ```

5. Create `requirements.txt`:
   ```text
   streamlit
   pandas
   numpy
   chromadb
   PyPDF2
   python-docx
   ollama
   tiktoken
   rouge-score
   langdetect
   plotly
   torch
   tqdm
   colab-xterm
   ```

## Optimization Notes
- **Code Modularization**: Splitting into sub-files improves maintainability and scalability.
- **Performance**: Use smaller models (e.g., `mistral`) for faster processing on CPU-only local setups.
- **Memory Management**: Clear ChromaDB periodically (`processor.clear_database()`) to manage storage.
- **Caching**: Add `@st.cache` to expensive functions in Streamlit for better performance.
- **Asynchronous Processing**: Use `asyncio` for I/O-bound tasks (e.g., file processing) to improve responsiveness.

## Downloading Files
To download the project files directly:
- **Single File (app.py)**: Download `app.py` from the repository or copy the provided code into a `.py` file.
- **Full Project Structure**: Clone the repository (if hosted) or create the structure manually:
  ```bash
  git clone <repository_url>
  ```
- **Colab Notebook**: Create a `.ipynb` file with the Colab setup commands and `app.py` content, or download a pre-configured notebook:
  ```python
  from google.colab import files
  files.download('app.py')
  ```

For a pre-configured notebook, check the repository (if available) or create one by combining the setup commands and code.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For issues or contributions, please open a pull request or contact the maintainer.