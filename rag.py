'''
RAG :  Retreival Augmented Tool

Primary role, take document as imoort and then break the content of the document into chunks, embed them through Open AI emebedding, compare it with the question vecton
using cosine similarity and then returning the final response with the help of the LLM model where the question the context are sent

'''


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
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


THRESHOLD = 0.30 #this will be used for comparison after cosine similarity function has returned, we will use threashold to decide which chunks can be used for context
CHUNK_SIZE = 100 # this is character limit of a single chunk, having a larger chunk size on a bigger file help with less number of calls to API improving the run time
EMBED_MODEL = "text-embedding-3-small" # embedding model being used, embedding will be used to convert the chunks to vectors which will be then be used to compare
EXTRACT_VERSION = "pypdf-v2"   # this is version history, not mandatory
DEFAULT_EMBED_SIZE = 100 # this will be used in the embed function to pass the maximum number of chunks at one go the embed mode
TOP_K = 10 # maximum context length sent to the LLM model for final response, these are top matched with the question as shown below
MAX_FILE = 10 # maximum file size
MAX_PAGES = 100 # maximum number of pages in the document

load_dotenv() # this will be used to load the .env file which contains the API key

_client = None # creating a temporary client variable
'''
whenever we would need to get the api key which will happen during the embedding model call and LLM model call, we will use the get_client() function
'''
def get_client():
    global _client 
    if _client is not None:  # this will check if the client is none i.e if the client does not have a key instance
        return _client

    key = os.environ.get('OPENAI_API_KEY') #calling the key stored in then .env file
    if not key:
        try:
            key = st.secrets['OPENAI_API_KEY'] # if the key does not exist in the .env file which will happen in the case of streamlit file, bevause 
            # .env is not sent to GitHub hence, streamlit never sees it so we need a way to call the secrets area of streamlit where the key is also stored
        except(KeyError,StreamlitSecretNotFoundError): # if the key is not found anywhere then ofcourse here is the error
            key = None
    if not key:
        raise DocumentError('OPENAI API KEY NOT FOUND') # if the key is nowhere then raise the documnent error 
    _client = OpenAI(api_key=key) # the key gets assigned to the client temp variable
    return _client #client variabnle with the key is then returned


'''
the role of read_bytes() function is to read the file as bytes using "rb" and then just return the content of the file which will be bytes
# if file is not found then raise an error

'''

def read_bytes(path):
    try: 
        with open(path, "rb") as file:
            return file.read()
    except FileNotFoundError:
        raise DocumentError(f"{path} not found, check the file or  path")

'''
Error class - Document Error, this will be called whenever there is an error and since this is the child class of Exception it will be able handle all of the exception
all we need to raise the Document Error where there is a potential issue and it will be handeled in one place
'''

class DocumentError(Exception): 
    pass


'''
hasfingerprint() : this is one of the most critical function as this will be used to generate a unique finger print for the file, which will tell us the folliwing

1. whethwer the file has been modified-  if the fingerprint changed then it means yes
2. whether the file already is in use - this is important to avoid multiple chunking and embedding, saving loads of time

how does the hashfingerprint work?

1. it takes file_hash as the first input which is basically hashed version of the entire data in the file coming from the hash_bytes() func
2. then it will called the hashlib library which is used for hashing in python
3. then sha256 will be selected as the hashing protocol
4. file_hash, chunksize, embedmodel and extract version are then combined in a string - this will help generate the unique fingerprint
5. then .encode function is called which will convert the string to bytes
6. bytes are passed to  hexdigest which will convert the bytes to hexadecimal giving the final hashcode as fingerprint
'''
def hashfingerprint(file_hash):
    fingerprint = hashlib.sha256(f"{file_hash}|{CHUNK_SIZE}|{EMBED_MODEL}|{EXTRACT_VERSION}".encode()).hexdigest() 
    return fingerprint

'''
hash_bytes() is used to generate data hash it also uses the hashlib.sha256 but does not use encode because the data is already in bytes
'''
def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


'''
read_file() function will be called when the file type is txt 
it will get the data which will be in bytes and has to decode the text into string for it to be read by the chunks function
if there is an issue with the bytes being passed to the function the it will raise a Document  Error as the content become unreadable
'''

def read_file(data):
    try:
        return data.decode("UTF-8")
    except UnicodeDecodeError:
        raise DocumentError("this file is not readable as UTF-8 text")



'''
read_pdf() will be called when the file type is pdf
it will take bytes as input and follow the following approach:

1. it will first generate a data streamm using the io.BytesIO which will be passed to pdfReader and the final version is then passed to a reader instance
BytesIO gets the bbytes and covert it into file like instance which is then passed to pdfReader which extracts the content and send to reader

2. reader then checks if the pages in the file are under the limit, if they are then the text is extracted and stored in a list (page_content)
3. the entire data from the list is then combined in full_text
4. then we perform a check to see if the file was not empty or only had blankspaces, if the file was empty then an error is raised
5. if the file was not empty then the entire text is returned
'''
def read_pdf(PdfBytes):
    try:
        stream =  io.BytesIO(PdfBytes)
        reader = PdfReader(stream)
        page_content = []

        if len(reader.pages) <= MAX_PAGES:
            for page in reader.pages:
                extracted = page.extract_text() or "" # this will decode the bytes to actual text or store "" if the page is blank
                page_content.append(extracted)
        else:
            raise DocumentError(f"The document contains {len(reader.pages)}pages, while the limit is {MAX_PAGES} pages")
    
    except PyPdfError:
        raise DocumentError("This PDF is damanged or password protected")

    full_text = "\n".join(page_content)

    if not full_text.strip():
        raise DocumentError("PDF has no extractable text")

    return full_text

'''
chunk_text is used to split the text which it recives either from read_pdf() or read_file() and then splits it into chunks as list and then return that list
1. size would mostly be the DEFAULT_CHUNK_SIZE called above 
2. the text is split into a list and then through slicing the chunks are created from the list
3. a check is needed to ensure the chunks are not empty

'''

def chunk_text(text, size):
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


'''
Emebedding is used to take the chunks rom chunk_text -> pass it to a open Ai embedding model - > get the vectors -> return those vectors

As we have alot of chunks in the list we cannot pass all of them at one as every model has a token input limit, hence, we use DEFAULT_EMBED_SIZE
the role of this to divide the chunks list into further chunks through slicing in order to avoid model overloading

then each new embeded chunk is passed over to the model through the following process

response.get_client(this is will get the api key).embeddings.creat() - this will pass the EMBED MODEL we want to use, input chunk and the return type "float"
that we will require

the response from the model is added in a list 

as response will have a lot of data we will only take response.data.embedding as the final ouput as this will include the vectors we need

list of all the vectors is then returned


'''
def embed(text):
    print(f"embedding {len(text)} text(s)")
    embedding_vector = []
    j = int(math.ceil(len(text)/DEFAULT_EMBED_SIZE))
    i = 1
    for start in range(0, len(text), DEFAULT_EMBED_SIZE):
        print(f"{i}/{j} Embedding  now.....")
        chunk = text[start : start + DEFAULT_EMBED_SIZE]
        response = get_client().embeddings.create(
            model = EMBED_MODEL,
            input = chunk,
            encoding_format="float"
            )
        for item in response.data:  # response.data holds one item per input text
            embedding_vector.append(item.embedding)
        i+=1
    
    print(f"Embedding vector length : {len(embedding_vector)}")
    return embedding_vector

'''
answer_question() wille bs used to take in the question and context which will be the selected chunks having the highest similarty with the question

All of those chunks will be passed on to the LLM model which will be assigned a role and will be asked to use the context to answer the question
the model is further stopped from using an external knowledge
response from the model is then return which will be printed

so either a answer to the question or a confirmation that the answer to the question does not exist inside the model
'''



def answer_question(question, context):

    response = get_client().chat.completions.create(
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

'''
cosine_similarity() uses a mathematical function which takes in 2 vectors 

1. the question vector 
2. the vector of the chunks from the document

each question and combination will have a similarity which is then used to find the best matching chunk with the question


returns a response which is 1, 0, -1 
Similarity of two vectors: 1 = same direction, 0 = unrelated, -1 = opposite.

here is how it works : 
    Dot product divided by the two lengths. Dividing by length means only direction
    matters, so a short chunk and a long chunk on the same topic still score high.

'''

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return (np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


'''
save_index is basic function which just does 2 things

1. create a dict which stores the fingerprint, chunks and vector
2. takes the path generated in the main()n and dumps the dict in the json format to that path
3., this function will only called when either the path does not exists or fingerint is not a match
'''


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


'''
load_index() will be called when we already have a file in cache stored as JSON, and the only role this play is returning the content stored
'''

def load_index(path):
    """Read a saved index. Returns (chunks, vectors, saved_hash).

    The key names must match save_index EXACTLY. A one-letter typo causes a KeyError.
    """
    data = dict()
    with open(path, "r") as file:
        data = json.load(file)

    return data["Chunks"], data["vectors"], data["hashCode"]

'''
extract_text() takes in the byte data and the file name 
it will check whether the filename is pdf or txt and the will call the correct function to get the content and return the data
'''

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


def retrieve(question, chunks, vectors, top_k=TOP_K):
    if question is None or not question.strip():
        raise DocumentError("Enter a question")

    vector_question  = embed([question])
    score = []
    for i, vec in enumerate(vectors):
        s = cosine_similarity(vec, vector_question[0])
        score.append([s, chunks[i]])

    score = sorted(score, key=lambda x: x[0], reverse=True)
   
    return score[:top_k]

def answer(question, chunks, vectors):
    top = retrieve(question, chunks, vectors)
    if top[0][0] < THRESHOLD:
        return {"answer": "no relevant information in the document", "sources": [], "best_score": top[0][0]}
    else:
        sources = [item[1] for item in top]
        context = "\n\n".join(sources)
        return {"answer": answer_question(question, context), "sources": sources, "best_score": top[0][0]}

if __name__ == "__main__":
    # --- 1. Command-line input: the file to search and the question to ask ---
    start_time = time.perf_counter() 
    parser = argparse.ArgumentParser(description="Calling the parser for declaration") # generating a argument parser, this will be used to get input from CLI
    parser.add_argument("filename", help="enter the name of the file") # used to call the file name, if the filname is not provided then help= will be printed
    parser.add_argument("question", help="Enter the question you would like the system to answer") # used to get the queastion
    args = parser.parse_args() # generating parser instance and passing it to args variable

    try:
        byteData = read_bytes(args.filename) 
        '''
        the first call is to the read_bytes function with the filename, because during streamlit the data sent will not be a file rather it will be stream
        of bytes hence, we cannot handel it normally and have to send it as bytes

        this functio is mainly used when we working with CLI because we need to use this function to read the file as byte while in streamlit this will be default
        '''
        file_path = f"{os.path.basename(args.filename)}_indexes.json"
        # generating filepath is very imoortant because this will store the cache file for us so that we dont have to load te file everytimg a question is asked
        # this will save the runtime of the code and memory
        

        '''
        Role of the build_index is to take in data, filename and the path

        Generate Finger print which will be act as unique identity for the cache

        1. check if the path exist(by matching the fingerprint), if yes, return the data
        2. if the path does not exist then open the file to read its content
        3. generate chunks
        4. send those chunks over to the embed code
        5. return chunks, vectors and fingerprint
        '''
        result_file, vector_file, fingerprint = build_index(byteData, args.filename, file_path)

        '''
        answer() is used to generating the final response, it will take input the output from the build_iondex function and then do the following:
        1. retreive the top 10 chunks by checkign the threshold
        2, then run the LLM model and pass those top 10 chunks alongside the question
        3. return the answer 

        '''

        result  = answer(args.question, result_file, vector_file)
        print(result["answer"])

        end_time = time.perf_counter()
        print(f"total time taken {end_time-start_time:.6f} seconds")

    except DocumentError as e:
        print(e)
        sys.exit(1)
