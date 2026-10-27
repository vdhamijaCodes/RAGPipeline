
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
client = OpenAI()

api_key = os.environ.get("OPENAI_API_KEY")

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
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text,
        encoding_format="float"
    )

    embedding_vector =[]
    for item in response.data:
        embedding_vector.append(item.embedding)

    return embedding_vector

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return (np.dot(a,b))/(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    text = "one two three four five six seven"
    result = chunk_text(text, 4)
    vector = embed(result)
    print(result)
    print(len(vector))
    print(len(vector[0]))   
    a = [1,0]
    b = [2,0]
    print(cosine_similarity(a,b))



