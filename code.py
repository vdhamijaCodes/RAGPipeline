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

if __name__ == "__main__":
    text = "one two three four five six seven"
    result = chunk_text(text, 4)
    print(result)

