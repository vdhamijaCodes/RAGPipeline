
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import os
import sys
import hashlib

THRESHOLD = 0.30

load_dotenv()
client = OpenAI()

# api_key = os.environ.get("OPENAI_API_KEY")

def read_file(filename):
    try:
        with open(filename, "r", encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"{filename} not found in the current directory, please check the location")
        sys.exit(1)
    
def chunk_text(text, size):
    lis = []
    words = text.split()
    j = 0
    for i in range(len(words)):
        if j >= len(words):
            return lis
        else:
            lis.append(" ".join(words[j:size+j]))
            j = size+j

    return lis

def embed(text):
    print(f"embedding {len(text)} text(s)")
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text,
        encoding_format="float"
    )

    embedding_vector =[]
    for item in response.data:
        embedding_vector.append(item.embedding)

    return embedding_vector

def answer_question(question, chunks):
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
    return response.choices[0].message.content 

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return (np.dot(a,b))/(np.linalg.norm(a) * np.linalg.norm(b))

def save_index(chunks, vectors, path, contentHash):
    data = dict()
    data["hashCode"] = contentHash
    data["Chunks"] = chunks
    data["vectors"] = vectors

    with open(path,"w") as file:
        json.dump(data, file)


def load_index(path):
    data = dict()
    with open(path, "r") as file:
        data = json.load(file)
       
    return data["Chunks"], data["vectors"], data["hashCode"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calling the parser for declaration")
    parser.add_argument("filename",help="enter the name of the file")
    parser.add_argument("question", help= "Enter the question you would like the system to answer")
    args = parser.parse_args()
    content = read_file(args.filename)
    contentHash = hashlib.sha256(content.encode()).hexdigest()
    result_file = chunk_text(content, 40)
    file_path = f"{os.path.basename(args.filename)}_indexes.json"
    if os.path.exists(file_path):
        saved_chunks, saved_vectors, hashCode = load_index(file_path)
        if hashCode != contentHash:
            vector_file = embed(result_file)
            save_index(result_file, vector_file,file_path, contentHash)
        else:
            result_file = saved_chunks
            vector_file = saved_vectors
    else:
        vector_file = embed(result_file)
        save_index(result_file, vector_file,file_path, contentHash) 

    result_question = chunk_text(args.question, len(args.question))
    vector_question = embed(result_question)
    score = []
    i = 0
    for vec in vector_file:
        s = cosine_similarity(vec, vector_question[0])
        score.append([s,result_file[i]])
        i+=1

    score = sorted(score, key=lambda x:x[0], reverse=True)

    if score[0][0] >= THRESHOLD:
        # print(score[:3])
        print(answer_question(args.question, "\n\n".join([item[1] for item in score[:3]])))
    else:
        print("Sorry, I could not find anything related to the question you have asked")





