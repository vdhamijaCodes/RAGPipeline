import streamlit as st
from rag import build_index, DocumentError, answer


MAX_SIZE = 10
MAX_QUESTIONS = 10


@st.cache_resource(max_entries=5, ttl = "30m")
def wrapper(data, filename):
    return build_index(data, filename)

try:
    file = st.file_uploader("Attach a .pdf or .txt file", type=["pdf","txt"], max_upload_size=MAX_SIZE)
    if file is None:
        st.info("Upload a file to begin")
        st.stop()


    data = file.getvalue()
    filename = file.name
    with st.spinner("Reading your document..."):
        chunks, vectors, fingerprint = wrapper(data, filename)

    question = st.text_input("Ask a question : ")
    if "count" not in st.session_state:
        st.session_state.count = 1
    if st.button("Ask"):
        if st.session_state.count <= MAX_QUESTIONS:
            if question.strip():
                with st.spinner(text="looking for an answer..."):
                    result = answer(question, chunks, vectors)
                    st.write(result["answer"])
                with st.expander("Check the sources used : "):
                    st.write(
                        f"remaining question : {st.session_state['count']}\n"

                        f"sources : {result['sources']}"
                    )
            else:
                st.warning("Please Type a question")
            st.session_state.count +=1
        else:
            st.warning("You run out of the maximum question left, kindly restart a new session")

    
except DocumentError as e:
    st.error(str(e))

