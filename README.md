# Medical Chatbot

This is a web-based chatbot that answers medical questions using a Retrieval-Augmented Generation (RAG) model. The chatbot uses "The GALE ENCYCLOPEDIA of MEDICINE SECOND" as its knowledge base.

## How it works

The application follows a Retrieval-Augmented Generation (RAG) architecture:

1.  **User Input:** The user asks a question through the web interface.
2.  **Embedding:** The user's question is converted into a vector embedding using a sentence transformer model.
3.  **Retrieval:** The embedding is used to search for relevant text chunks from the medical encyclopedia stored in a Pinecone vector store.
4.  **Augmentation:** The retrieved text chunks are then passed as context to a Large Language Model (LLM).
5.  **Generation:** The LLM (Llama 3.3 70B running on Groq) generates a comprehensive answer based on the provided context.

## Technology Stack

*   **Backend:** Flask
*   **LLM Framework:** LangChain
*   **LLM:** Llama 3.3 70B (via Groq)
*   **Vector Store:** Pinecone
*   **Embedding Model:** Hugging Face Sentence Transformers
*   **PDF Processing:** PyPDF

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add the following variables:
    ```
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    ```

5.  **Process the data and populate the vector store:**
    Run the `store_index.py` script to process the PDF file, create embeddings, and store them in your Pinecone index.
    ```bash
    python store_index.py
    ```
    *Note: Make sure your Pinecone index is created with the correct dimensions for the sentence transformer model.*

## Usage

To start the chatbot, run the Flask application:

```bash
python app.py
```

The application will be available at `http://localhost:8080`.

## Project Structure

```
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── store_index.py        # Script to process data and store in Pinecone
├── data/                 # Contains the PDF knowledge base
├── src/                  # Source code for helper functions and prompts
├── static/               # Static files (CSS)
├── templates/            # HTML templates
└── README.md             # This file
```
