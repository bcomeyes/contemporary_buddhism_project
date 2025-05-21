# %% [markdown]
# # PDF to Text Converter with OpenAI Embeddings
# 
# This notebook extracts text from PDF files (including large ones) and can generate embeddings using OpenAI's API.
# The extracted text is saved to a text file, and optionally embeddings can be generated
# for the content to enable semantic search or analysis.
# 
# ## Requirements
# You'll need to install these packages first:
# ```
# pip install PyPDF2 pymupdf pdf2image pytesseract pillow openai tqdm tiktoken numpy python-dotenv
# ```
# 
# Note: For OCR functionality (handling scanned PDFs), you'll also need to install Tesseract OCR:
# - Windows: https://github.com/UB-Mannheim/tesseract/wiki
# - Mac: `brew install tesseract`
# - Linux: `sudo apt install tesseract-ocr`

# %% [markdown]
# ## Import Libraries

# %%
import os
import re
import time
import json
import glob
from pathlib import Path
import PyPDF2
import openai
from tqdm.notebook import tqdm  # Using notebook version for better display in Jupyter
import numpy as np
import tiktoken
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Load environment variables from .env file in the current directory
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# %% [markdown]
# ## Constants

# %%
# Constants
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model to use
MAX_TOKENS = 8191  # Maximum tokens per chunk for embedding
CHUNK_OVERLAP = 200  # Number of tokens to overlap between chunks
DPI = 300  # DPI for OCR conversion (higher is better quality but slower)

# %% [markdown]
# ## Helper Functions for PDF Extraction

# %%
def extract_text_from_pdf(pdf_path, use_ocr=False, ocr_threshold=10):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        use_ocr: Whether to use OCR for text extraction (for scanned PDFs)
        ocr_threshold: Character count threshold below which to try OCR (per page)
    
    Returns:
        Extracted text as a string
    """
    total_text = []
    
    # Try extracting text with PyMuPDF (faster and better than PyPDF2 for most PDFs)
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"Extracting text from {total_pages} pages...")
        for page_num in tqdm(range(total_pages)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Check if page has little text and might need OCR
            if use_ocr and len(text.strip()) < ocr_threshold:
                print(f"Page {page_num+1} might be scanned. Attempting OCR...")
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                # Apply OCR
                text = pytesseract.image_to_string(img)
            
            total_text.append(text)
        
        doc.close()
        
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        print("Falling back to PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for page_num in tqdm(range(total_pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    
                    # If text extraction failed or returned minimal text, try OCR
                    if use_ocr and len(text.strip()) < ocr_threshold:
                        print(f"Page {page_num+1} might be scanned. Attempting OCR...")
                        # Convert PDF page to image using pdf2image
                        images = convert_from_path(pdf_path, dpi=DPI, first_page=page_num+1, last_page=page_num+1)
                        # Apply OCR to the image
                        text = pytesseract.image_to_string(images[0])
                    
                    total_text.append(text)
                    
        except Exception as e:
            print(f"Error with PyPDF2: {e}")
            # If everything fails, try full OCR if enabled
            if use_ocr:
                print("Attempting full document OCR...")
                try:
                    images = convert_from_path(pdf_path, dpi=DPI)
                    for i, img in enumerate(tqdm(images)):
                        text = pytesseract.image_to_string(img)
                        total_text.append(text)
                except Exception as e:
                    print(f"OCR failed: {e}")
    
    # Join all text with double newlines between pages
    return "\n\n".join(total_text)

# %%
def clean_text(text):
    """Clean the extracted text"""
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single one
    text = re.sub(r' +', ' ', text)
    # Fix any broken words that might have been split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()

# %% [markdown]
# ## Helper Functions for Embeddings

# %%
def get_token_count(text, encoding_name="cl100k_base"):
    """Count the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

# %%
def split_into_chunks(text, max_tokens=MAX_TOKENS, overlap=CHUNK_OVERLAP):
    """Split text into chunks respecting token limits with overlap"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # Get chunk of tokens (respecting max_tokens)
        chunk_tokens = tokens[i:i + max_tokens]
        # Decode chunk back to text
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        # Move forward by max_tokens - overlap
        i += max_tokens - overlap
    
    return chunks

# %%
def create_embeddings(chunks, api_key=None):
    """Create embeddings for text chunks using OpenAI API"""
    # Use provided API key or environment variable
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = OPENAI_API_KEY
        
    if not openai.api_key:
        print("Error: No OpenAI API key provided. Set OPENAI_API_KEY in your .env file.")
        return []
        
    embeddings = []
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(tqdm(chunks)):
        try:
            # Add a small delay to respect API rate limits
            if i > 0 and i % 10 == 0:
                time.sleep(1)
                
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk
            )
            embedding = response.data[0].embedding
            embeddings.append({
                "chunk": chunk,
                "embedding": embedding,
                "chunk_index": i
            })
        except Exception as e:
            print(f"Error creating embedding for chunk {i}: {e}")
    
    return embeddings

# %% [markdown]
# ## Semantic Search Function

# %%
def search_embeddings(query, embeddings, api_key=None, top_n=5):
    """Search embeddings for relevant text chunks based on a query"""
    # Get embedding for the query
    if api_key:
        openai.api_key = api_key
    else:
        # Use environment variable if no API key is provided
        openai.api_key = OPENAI_API_KEY
    
    if not openai.api_key:
        print("Error: No OpenAI API key provided. Set OPENAI_API_KEY in your .env file.")
        return []
    
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Convert query embedding to numpy array
        query_embedding_array = np.array(query_embedding)
        
        # Calculate similarity scores
        similarities = []
        for i, item in enumerate(embeddings):
            embed_array = np.array(item["embedding"])
            # Cosine similarity
            similarity = np.dot(query_embedding_array, embed_array) / (
                np.linalg.norm(query_embedding_array) * np.linalg.norm(embed_array)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N results
        results = []
        for idx, score in similarities[:top_n]:
            results.append({
                "chunk": embeddings[idx]["chunk"],
                "similarity": float(score),
                "chunk_index": embeddings[idx]["chunk_index"]
            })
        
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []

# %% [markdown]
# ## Interactive Notebook Version

# %%
# Interactive part - you can edit these parameters
pdf_pattern = "*.pdf"  # Glob pattern to match PDF files (e.g., "*.pdf", "documents/*.pdf")
output_dir = "."  # Directory to save output files
generate_embeddings = False  # Set to True if you want to generate embeddings
use_ocr = False  # Set to True for scanned PDFs that need OCR
batch_process = False  # Set to True to process multiple PDFs matching the pattern
single_file = "your_document.pdf"  # Path to a single PDF file if not batch processing

# Use API key from environment variable by default
# You can override it here if needed
openai_api_key = OPENAI_API_KEY

# Check if API key is available when embeddings are requested
if generate_embeddings and not openai_api_key:
    print("Warning: No OpenAI API key found in environment variables.")
    print("Please add OPENAI_API_KEY to your .env file or set it here manually.")

# %% [markdown]
# ## Process PDF Files

# %%
# Create output directory if it doesn't exist
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Function to process a single PDF file
def process_pdf(pdf_file):
    print(f"\nProcessing: {pdf_file}")
    
    # Get filename without extension
    filename = Path(pdf_file).stem
    
    # Extract text from PDF
    print(f"Extracting text from {pdf_file}...")
    text = extract_text_from_pdf(pdf_file, use_ocr=use_ocr)
    
    # Clean the text
    if text:
        print("Cleaning extracted text...")
        text = clean_text(text)
        
        # Save text content
        text_path = output_dir / f"{filename}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {text_path}")
        
        # Display a sample of the text
        print("\nSample of extracted text:")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("\nTotal characters:", len(text))
        token_count = get_token_count(text)
        print(f"Total tokens: {token_count}")
        
        # Create embeddings if requested
        if generate_embeddings:
            if not openai_api_key:
                print("Error: OpenAI API key is required for creating embeddings")
                print("Please set OPENAI_API_KEY in your .env file or provide it in the parameters cell.")
            else:
                # Split text into chunks
                chunks = split_into_chunks(text)
                print(f"Split into {len(chunks)} chunks")
                
                # Generate embeddings
                embeddings = create_embeddings(chunks, openai_api_key)
                
                # Save embeddings
                embeddings_path = output_dir / f"{filename}_embeddings.json"
                with open(embeddings_path, "w", encoding="utf-8") as f:
                    json.dump(embeddings, f, ensure_ascii=False, indent=2)
                print(f"Embeddings saved to {embeddings_path}")
                
                # Save a version with just the text chunks for reference
                chunks_path = output_dir / f"{filename}_chunks.json"
                chunks_data = [{"chunk_index": i, "chunk": chunk} for i, chunk in enumerate(chunks)]
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                print(f"Text chunks saved to {chunks_path}")
        
        return True
    else:
        print(f"Failed to extract text from {pdf_file}")
        return False

# %%
# Process files based on settings
if batch_process:
    # Find all PDF files matching the pattern
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found matching pattern: {pdf_pattern}")
    else:
        print(f"Found {len(pdf_files)} PDF files matching pattern: {pdf_pattern}")
        
        # Process each PDF file
        results = []
        for pdf_file in pdf_files:
            success = process_pdf(pdf_file)
            results.append((pdf_file, success))
        
        # Summary
        print("\n===== Processing Summary =====")
        print(f"Total files: {len(results)}")
        successful = sum(1 for _, success in results if success)
        print(f"Successfully processed: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if len(results) - successful > 0:
            print("\nFailed files:")
            for pdf_file, success in results:
                if not success:
                    print(f"- {pdf_file}")
else:
    # Process a single file
    process_pdf(single_file)

# %% [markdown]
# ## Example: Using the Semantic Search Feature
# 
# After generating embeddings, you can use this code to search within your PDF content.
# Uncomment and modify this code when you're ready to search your embeddings.

# %%
# Example code for semantic search - uncomment and modify when needed
# Load embeddings from file
# embedding_file = "your_document_embeddings.json"  # Replace with your actual embeddings file
# with open(embedding_file, "r", encoding="utf-8") as f:
#     embeddings = json.load(f)

# Search for relevant content - uses API key from .env by default
# query = "Enter your search query here"  # Replace with your actual search query
# results = search_embeddings(query, embeddings, top_n=3)

# Display results
# print(f"Search results for: {query}\n")
# for i, result in enumerate(results):
#     print(f"Result {i+1} (Similarity: {result['similarity']:.4f}):")
#     print("-" * 40)
#     print(result["chunk"][:300] + "...")  # Show first 300 chars
#     print()

# %% [markdown]
# ## Examples: Batch Processing Patterns
# 
# Here are examples of different glob patterns you can use for batch processing.
# Uncomment and modify these examples when needed.

# %%
# Example glob patterns for different scenarios - uncomment and modify when needed

# Process all PDFs in the current directory
# pdf_pattern = "*.pdf"

# Process PDFs with specific naming pattern
# pdf_pattern = "report_*.pdf"

# Process PDFs in a specific directory
# pdf_pattern = "documents/*.pdf"

# Process PDFs in a specific directory and subdirectories (recursive)
# pdf_pattern = "documents/**/*.pdf"  # Note: requires Python 3.5+

# Process PDFs from multiple directories
# pdf_files = []
# pdf_files.extend(glob.glob("reports/*.pdf"))
# pdf_files.extend(glob.glob("archives/*.pdf"))

# %% [markdown]
# ## Creating a .env File
# 
# Create a file named `.env` in the same directory as this notebook with the following content:
# 
# ```
# OPENAI_API_KEY=your_api_key_here
# ```
# 
# This file will be automatically loaded when you run the notebook, and the API key will be available
# without having to hardcode it.
