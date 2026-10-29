"""
Mini RAG tool: ask a question about a text file.

How it works (the whole idea in 5 steps):
  1. Read the file and split it into chunks of ~40 words.
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
from PyPDF2 import PdfReader

# Minimum similarity for the best chunk. Below this, the question is treated
# as unrelated to the document. 0.30 came from testing on-topic questions
# (best scores 0.52+) vs off-topic ones (best scores 0.15 or lower).
# It is specific to the text-embedding-3-small model; re-measure if you change models.
THRESHOLD = 0.30

load_dotenv()      # Must run BEFORE OpenAI() so the key from .env is available.
client = OpenAI()  # Reads OPENAI_API_KEY from the environment automatically.


def read_file(filename):
    """Return the whole file as one string. Exit with a clear message if missing."""
    try:
        with open(filename, "r", encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"{filename} not found in the current directory, please check the location")
        sys.exit(1)


def read_pdf(filepath):
    try:
        with open(filepath, "rb") as file:
            reader = PdfReader(file)
            page_content = []
            for i in range(len(reader.pages)):
                page_content.append(reader.pages[i].extract_text())  
                
            content =  "\n".join(page_content)
            if len(content.strip()) == 0:
                sys.exit(1)
            else:
                return content

    except FileNotFoundError:
        print(f"{filepath} not found in the current directory, please check the location")
        sys.exit(1)



def chunk_text(text, size):
    """Split text into chunks of `size` words. Returns a list of strings.

    Example: 7 words with size=4 -> ['w1 w2 w3 w4', 'w5 w6 w7'] (last chunk may be shorter).
    """
    lis = []
    words = text.split()
    j = 0  # start position of the current chunk
    for i in range(len(words)):  # 'i' is unused; the loop just guarantees enough iterations
        if j >= len(words):      # ran past the end, so we are done
            return lis
        else:
            lis.append(" ".join(words[j:size+j]))  # slice of words, joined back into a string
            j = size+j
    return lis


def embed(text):
    """Turn a LIST of strings into a list of vectors (1536 numbers each).

    One API call handles the whole list, and the vectors come back in the same order.
    The print is a debug line: it shows how many texts were sent (i.e. how much we pay for).
    """
    print(f"embedding {len(text)} text(s)")
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text,
        encoding_format="float"
    )

    embedding_vector = []
    for item in response.data:  # response.data holds one item per input text
        embedding_vector.append(item.embedding)
    return embedding_vector


def answer_question(question, chunks):
    """Ask the chat model to answer from the given chunks only (the 'G' in RAG).

    The system prompt forbids using outside knowledge, which is what makes the model say
    "not in the document" for questions the chunks can't answer.
    Low temperature (0.3) keeps answers focused. max_tokens caps the answer length.
    """
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role" : "system",
                "content" : "You are a question-answering assistant operating within a Retrieval-Augmented Generation (RAG) system. Your task is to answer the user's question using ONLY the information explicitly available in the provided context. Do not invent answer, if the answer to the question is not available in the data chunk then say clearly that the document doesn't contain the answer"
            },
            {
                "role" : "user",
                "content" : f"Here is the chunk of the data  : {chunks} \n this must be used to answer the following question : {question}"
            }
        ],
        temperature = 0.3,
        max_tokens = 300
    )
    return response.choices[0].message.content  # dig the answer text out of the response


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


if __name__ == "__main__":
    # --- 1. Command-line input: the file to search and the question to ask ---
    parser = argparse.ArgumentParser(description="Calling the parser for declaration")
    parser.add_argument("filename", help="enter the name of the file")
    parser.add_argument("question", help="Enter the question you would like the system to answer")
    args = parser.parse_args()

    # --- 2. Read the document and fingerprint it ---
    # The hash changes if even one character of the file changes. We use it to
    # detect that a saved index is stale (built from an older version of the file).
    content = read_file(args.filename)
    contentHash = hashlib.sha256(content.encode()).hexdigest()
    result_file = chunk_text(content, 40)  # chunks of the CURRENT file

    # One cache file per document, e.g. "Ocean.txt_indexes.json" (created in the folder you run from).
    file_path = f"{os.path.basename(args.filename)}_indexes.json"

    # --- 3. Get the chunk vectors: from the cache if it is still valid, otherwise rebuild ---
    if os.path.exists(file_path):
        saved_chunks, saved_vectors, hashCode = load_index(file_path)
        if hashCode != contentHash:
            # File was edited since the cache was built: re-embed the NEW chunks and overwrite.
            vector_file = embed(result_file)
            save_index(result_file, vector_file, file_path, contentHash)
        else:
            # Cache is valid: reuse it, with no embedding cost for the document.
            result_file = saved_chunks
            vector_file = saved_vectors
    else:
        # First run for this document: embed everything and save the cache.
        vector_file = embed(result_file)
        save_index(result_file, vector_file, file_path, contentHash)
    # NOTE: the hash covers only the file's text. If you change the chunk size (40)
    # or the embedding model, delete the *_indexes.json files by hand, or the old
    # vectors will be reused wrongly.

    # --- 4. Embed the question ---
    # chunk_text with size=len(question) forces the question into exactly one chunk
    # (it never has more words than characters). embed() wants a list, hence this.
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

    # --- 6. Answer, but only if the best chunk is relevant enough ---
    if score[0][0] >= THRESHOLD:
        # Join the top 3 chunk texts with blank lines so the model can see where each one ends.
        print(answer_question(args.question, "\n\n".join([item[1] for item in score[:3]])))
    else:
        print("Sorry, I could not find anything related to the question you have asked")


    print("Calling PDF reader")
    pdf_content  = read_pdf("/home/varun/Desktop/RAG/Introduction to Linear Algebra Fifth Edition by Gilbert Strang.pdf")
    print(pdf_content[:])