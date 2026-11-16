"""
Mini RAG tool: ask a question about a text file.

How it works (the whole idea in 5 steps):
  1. Read the file and split it into chunks of ~200(CHANGE THE CHUNK SIZE IF NEEDED) words.
  2. Turn each chunk into a vector (embedding) so "meaning" can be compared with math.
     These vectors are cached in a JSON file so we only pay for them once.
  3. Embed the question the same way.
  4. Rank the chunks by cosine similarity to the question and keep the top 3.
  5. If the best score clears THRESHOLD, send those 3 chunks plus the question
     to the chat model, which answers using ONLY those chunks.

Usage:  python3 code.py <file.txt> "<question>"
Needs:  OPENAI_API_KEY in a .env file next to this script.
"""

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import os
import sys
import hashlib
from pypdf import PdfReader
from pypdf.errors import PyPdfError
import time
import math
import io

# Minimum similarity for the best chunk. Below this, the question is treated
# as unrelated to the document. 0.30 came from testing on-topic questions
# (best scores 0.52+) vs off-topic ones (best scores 0.15 or lower).
# It is specific to the text-embedding-3-small model; re-measure if you change models.
THRESHOLD = 0.30
CHUNK_SIZE = 100 # change this as the size of the file changes, big files would require a higher chunnk size to reduct the time taken, if the chunk size changes then delete any old jSON file else there will be problem
EMBED_MODEL = "text-embedding-3-small" # change this only when you need more compute for embedding, models are expensive
EXTRACT_VERSION = "pypdf-v2"  # this is version control, change this when the pay text is pulled changes    
DEFAULT_EMBED_SIZE = 100

load_dotenv()      # Must run BEFORE OpenAI() so the key from .env is available.
client = OpenAI()  # Reads OPENAI_API_KEY from the environment automatically.

def read_bytes(path):
    try: 
        with open(path, "rb") as file:
            return file.read()
    except FileNotFoundError:
        raise DocumentError(f"{path} not found, check the file or  path")

class DocumentError(Exception):
    pass

def hashfingerprint(file_hash):
    fingerprint = hashlib.sha256(f"{file_hash}|{CHUNK_SIZE}|{EMBED_MODEL}|{EXTRACT_VERSION}".encode()).hexdigest()
    return fingerprint


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def read_file(data):
    try:
        return data.decode("UTF-8")
    except UnicodeDecodeError:
        raise DocumentError("this file is not readable as UTF-8 text")


def read_pdf(PdfBytes):
    try:

        stream =  io.BytesIO(PdfBytes)
        reader = PdfReader(stream)
        page_content = []

        for page in reader.pages:
            extracted = page.extract_text() or ""
            page_content.append(extracted)
        
    except PyPdfError:
        raise DocumentError("This PDF is damanged or password protected")

    full_text = "\n".join(page_content)

    if not full_text.strip():
        raise DocumentError("PDF has no extractable text")

    return full_text

def chunk_text(text, size):
    """Split text into chunks of `size` words."""

    if size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    words = text.split()
    chunks = []

    for j in range(0, len(words), size):
        chunk = " ".join(words[j:j + size])
        chunks.append(chunk)

    if len(chunks) == 0:
        raise DocumentError("file is empty")

    return chunks



def embed(text):
    """Turn a LIST of strings into a list of vectors (1536 numbers each).

    One API call handles the whole list, and the vectors come back in the same order.
    The print is a debug line: it shows how many texts were sent (i.e. how much we pay for).
    """

    
    print(f"embedding {len(text)} text(s)")
    embedding_vector = []
    j = int(math.ceil(len(text)/DEFAULT_EMBED_SIZE))
    i = 1
    for start in range(0, len(text), DEFAULT_EMBED_SIZE):
        print(f"{i}/{j} Embedding  now.....")
        chunk = text[start : start + DEFAULT_EMBED_SIZE]
        response = client.embeddings.create(
            model = EMBED_MODEL,
            input = chunk,
            encoding_format="float"
            )
        for item in response.data:  # response.data holds one item per input text
            embedding_vector.append(item.embedding)
        i+=1
    
    print(f"Embedding vector length : {len(embedding_vector)}")
    return embedding_vector



def answer_question(question, context):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a question-answering system.

                Answer using ONLY the information in the
                provided context.

                Carefully examine ALL the context passages
                before deciding whether the answer is available.

                If the context contains sufficient information,
                provide a clear and accurate answer.

                If the context does not contain sufficient
                information, respond:
                "The document doesn't contain the answer."

                Do not use external knowledge or invent facts.
                """
            },
            {
                "role": "user",
                "content": f"""
                <context>
                {context}
                </context>

                <question>
                {question}
                </question>
                """
            }
        ],
        temperature=0,
        max_tokens=300
    )

    return response.choices[0].message.content



def cosine_similarity(a, b):
    """Similarity of two vectors: 1 = same direction, 0 = unrelated, -1 = opposite.

    Dot product divided by the two lengths. Dividing by length means only direction
    matters, so a short chunk and a long chunk on the same topic still score high.
    """
    a = np.array(a)
    b = np.array(b)
    return (np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def save_index(chunks, vectors, path, contentHash):
    """Save chunks + vectors + a fingerprint of the source text to a JSON file (the cache).

    chunks[i] and vectors[i] belong together (same order).
    """
    data = dict()
    data["hashCode"] = contentHash  # fingerprint of the document this index was built from
    data["Chunks"] = chunks
    data["vectors"] = vectors

    with open(path, "w") as file:
        json.dump(data, file)


def load_index(path):
    """Read a saved index. Returns (chunks, vectors, saved_hash).

    The key names must match save_index EXACTLY. A one-letter typo causes a KeyError.
    """
    data = dict()
    with open(path, "r") as file:
        data = json.load(file)

    return data["Chunks"], data["vectors"], data["hashCode"]

def extract_text(data, filename):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return read_pdf(data)
    else:
        return read_file(data)

def build_index(data, filename, cache_path = None):
    file_hash = hash_bytes(data)
    fingerprint = hashfingerprint(file_hash)
    cache_ok = False
    try:
        if cache_path is not None and os.path.exists(cache_path):
            saved_chunks, saved_vectors, ex_fingerprint = load_index(cache_path)
            if fingerprint == ex_fingerprint:
                cache_ok = True
    except (json.JSONDecodeError, KeyError, TypeError):
        cache_ok = False

    if cache_ok:
        chunks = saved_chunks
        vectors = saved_vectors
        
    else:
      
        content = extract_text(data, filename)
        chunks = chunk_text(content, CHUNK_SIZE)
        vectors = embed(chunks)
        if cache_path is not None:    
            save_index(chunks, vectors, cache_path, fingerprint)
   
    return chunks, vectors, fingerprint



if __name__ == "__main__":
    # --- 1. Command-line input: the file to search and the question to ask ---
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Calling the parser for declaration")
    parser.add_argument("filename", help="enter the name of the file")
    parser.add_argument("question", help="Enter the question you would like the system to answer")
    args = parser.parse_args()

    # --- 2. Read the document and fingerprint it ---
    # The hash changes if even one character of the file changes. We use it to
    # detect that a saved index is stale (built from an older version of the file).
    try:
        byteData = read_bytes(args.filename)
        file_path = f"{os.path.basename(args.filename)}_indexes.json"


        result_file, vector_file, fingerprint = build_index(byteData, args.filename, file_path)

        if not args.question.strip():
            raise DocumentError("please enter a question")
        result_question = chunk_text(args.question, len(args.question))
        vector_question = embed(result_question)

        # --- 5. Score every chunk against the question, as [score, chunk_text] pairs ---
        score = []
        i = 0
        for vec in vector_file:
            s = cosine_similarity(vec, vector_question[0])
            score.append([s, result_file[i]])
            i += 1

        score = sorted(score, key=lambda x: x[0], reverse=True)  # best match first

        # # --- 6. Answer, but only if the best chunk is relevant enough ---
        # if score[0][0] >= THRESHOLD:
        #     # Join the top 3 chunk texts with blank lines so the model can see where each one ends.
        #     print(answer_question(args.question, "\n\n".join([item[1] for item in score[:5]])))
        # else:
        #     print("Sorry, I could not find anything related to the question you have asked")


        if score[0][0] >= THRESHOLD:

            selected_chunk = [item[1] for item in score[:10]]
            context = "\n\n".join(selected_chunk)
            print(answer_question(args.question, context))
        else:
            print("We Could not find any relevant information regarding the question")
        

        end_time = time.perf_counter()
        print(f"total time taken {end_time-start_time:.6f} seconds")

    except DocumentError as e:
        print(e)
        sys.exit(1)
