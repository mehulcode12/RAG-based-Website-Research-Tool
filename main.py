import streamlit as st
from rag import generate_answers, process_urls
st.title("RAG based Website Research Tool")

url1 = st.text_input("Enter URL 1")
url2 = st.text_input("Enter URL 2 (optional)")
url3 = st.text_input("Enter URL 3 (optional)")

process_url_button = st.button("Process URLs")
if process_url_button:
    urls = [url1, url2, url3]
    urls = [url for url in urls if url]  # Filter out empty URLs
    if urls:
        # Here you would call the process_urls function
        process_urls(urls)
        for status in process_urls(urls):
            st.text(status)
        st.success("URLs processed successfully!")
    else:
        st.warning("Please enter at least one URL.")

input = st.text_input("Enter your query here")
if input:
    
    answer, sources = generate_answers(input)
    st.header("Answer")
    st.write(answer)

    st.header("Sources")
    st.write(sources)