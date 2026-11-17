import streamlit as st
from rag import build_index, DocumentError, answer

try:

    file = st.file_uploader("Attach a .pdf or .txt file", type=["pdf","txt"])
    if file is None:
        st.info("Upload a file to begin")
        st.stop()
    
    data = file.getvalue()
    filename = file.name
    with st.spinner("Reading your document..."):
        chunks, vectors, fingerprint = build_index(data, filename)
    
    question = st.text_input("Ask a question : ")
    if st.button("Ask") and question.strip():
        with st.spinner(text="looking for an answer..."):
            result = answer(question, chunks, vectors)
            st.write(result["answer"])
        with st.expander("Check the sources used : "):
            st.write(
                f"sources : {result['sources']}"
            )

except DocumentError as e:
    st.error(str(e))

