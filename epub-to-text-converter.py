# %% [markdown]
# # EPUB to Text Converter with OpenAI Embeddings
# 
# This notebook extracts text from EPUB files and can generate embeddings using OpenAI's API.
# The extracted text is saved to a text file, and optionally embeddings can be generated
# for the content to enable semantic search or analysis.
# 
# ## Requirements
# You'll need to install these packages first:
# ```
# pip install ebooklib openai tqdm tiktoken numpy beautifulsoup4 python-dotenv
# ```

# %% [markdown]
# ## Import Libraries

# %%
import os
import re
import time
import json
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import openai
from tqdm.notebook import tqdm  # Using notebook version for better display in Jupyter
import numpy as np
import tiktoken
from dotenv import load_dotenv

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

# %% [markdown]
# ## Helper Functions

# %%
def epub_to_text(epub_path):
    """Extract text content from an EPUB file"""
    try:
        book = epub.read_epub(epub_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Get content from each document item
                content = item.get_content().decode('utf-8')
                # Use BeautifulSoup to extract text from HTML
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                # Clean up the text
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    chapters.append(text)
        
        return "\n\n".join(chapters)
    except Exception as e:
        print(f"Error processing EPUB file: {e}")
        return None

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
# ## Interactive Notebook Version

# %%
# Interactive part - you can edit these parameters
epub_file = "your_book.epub"  # Path to your EPUB file
output_dir = "."  # Directory to save output files
generate_embeddings = False  # Set to True if you want to generate embeddings

# Use API key from environment variable by default
# You can override it here if needed
openai_api_key = OPENAI_API_KEY

# Check if API key is available when embeddings are requested
if generate_embeddings and not openai_api_key:
    print("Warning: No OpenAI API key found in environment variables.")
    print("Please add OPENAI_API_KEY to your .env file or set it here manually.")

# %% [markdown]
# ## Process the EPUB

# %%
# Create output directory if it doesn't exist
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Get filename without extension
filename = Path(epub_file).stem

# Convert EPUB to text
print(f"Extracting text from {epub_file}...")
text = epub_to_text(epub_file)

# %% 
if text:
    # Save text content
    text_path = output_dir / f"{filename}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved to {text_path}")
    
    # Display a sample of the text
    print("\nSample of extracted text:")
    print(text[:500] + "...\n")
    
    # Create embeddings if requested
    if generate_embeddings:
        if not openai_api_key:
            print("Error: OpenAI API key is required for creating embeddings")
            print("Please set OPENAI_API_KEY in your .env file or provide it in the parameters cell.")
        else:
            # Set API key
            openai.api_key = openai_api_key
            
            # Split text into chunks
            token_count = get_token_count(text)
            print(f"Total tokens: {token_count}")
            
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
else:
    print("Failed to extract text from the EPUB file.")

# %% [markdown]
# ## Add Semantic Search Functionality

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
# ## Example Usage of Search Functionality
# Uncomment and modify this code when you're ready to search your embeddings

# %%
# Example code for semantic search 
# Uncomment when you have embeddings to search
'''
# Load embeddings from file
embedding_file = "your_book_embeddings.json"
with open(embedding_file, "r", encoding="utf-8") as f:
    embeddings = json.load(f)

# Search for relevant content - uses API key from .env by default
query = "Enter your search query here"
results = search_embeddings(query, embeddings, top_n=3)

# Display results
print(f"Search results for: {query}\n")
for i, result in enumerate(results):
    print(f"Result {i+1} (Similarity: {result['similarity']:.4f}):")
    print("-" * 40)
    print(result["chunk"][:300] + "...")  # Show first 300 chars
    print()
'''

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

# %% [markdown]
# ## Optional: Command-line Version
# 
# This cell contains a version of the code that can be run as a command-line script.
# It's included here for reference, but is commented out since we're using the notebook interactively.

# %%
# Command-line script version - keep this commented out in the notebook
'''
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB to text and optionally create embeddings")
    parser.add_argument("epub_file", help="Path to the EPUB file")
    parser.add_argument("--embeddings", action="store_true", help="Generate OpenAI embeddings")
    parser.add_argument("--api-key", help="OpenAI API key (if not using .env file)")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get filename without extension
    filename = Path(args.epub_file).stem
    
    # Convert EPUB to text
    print(f"Extracting text from {args.epub_file}...")
    text = epub_to_text(args.epub_file)
    
    if text:
        # Save text content
        text_path = output_dir / f"{filename}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {text_path}")
        
        # Create embeddings if requested
        if args.embeddings:
            if not api_key:
                print("Error: OpenAI API key is required for creating embeddings")
                print("Set it with --api-key or add OPENAI_API_KEY to your .env file")
                return
            
            # Split text into chunks
            token_count = get_token_count(text)
            print(f"Total tokens: {token_count}")
            
            chunks = split_into_chunks(text)
            print(f"Split into {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = create_embeddings(chunks, api_key)
            
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
    else:
        print("Failed to extract text from the EPUB file.")

if __name__ == "__main__":
    main()
'''
