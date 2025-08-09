import streamlit as st
import boto3
import os
import shutil
import tempfile
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate

# -------------------------
# Config / AWS / Globals
# -------------------------
s3 = boto3.client("s3")
BUCKET = os.environ.get("BUCKET_NAME")  # make sure this is set in your env / docker run
if not BUCKET:
    st.error("BUCKET_NAME environment variable is not set.")
    st.stop()

st.set_page_config(page_title="Chat with Indexed PDF", layout="wide")
st.title("Chat with Indexed PDF (load by Document ID)")

# -------------------------
# Session state
# -------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "question" not in st.session_state:
    st.session_state.question = ""

# -------------------------
# Helper: load index from S3 (expects <doc_id>.faiss and <doc_id>.pkl in bucket root)
# -------------------------
def load_vectorstore_from_s3(doc_id: str):
    tmp_dir = tempfile.mkdtemp()
    try:
        local_faiss = os.path.join(tmp_dir, "index.faiss")
        local_pkl = os.path.join(tmp_dir, "index.pkl")
        s3.download_file(BUCKET, f"{doc_id}.faiss", local_faiss)
        s3.download_file(BUCKET, f"{doc_id}.pkl", local_pkl)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    embeddings = BedrockEmbeddings()
    vectorstore = FAISS.load_local(tmp_dir, embeddings, allow_dangerous_deserialization=True)
    return vectorstore, tmp_dir

# -------------------------
# LLM and prompt setup
# -------------------------
def build_qa_chain_from_vectorstore(vectorstore):
    llm = BedrockChat(model_id="anthropic.claude-v2:1", model_kwargs={"temperature": 0.0, "max_tokens": 1024})
    prompt_template = """Use the provided context to answer the question concisely.
If the answer is not present in the context, say "I don't know."

<context>
{context}
</context>

Question: {question}

Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa

# -------------------------
# UI: Load index by Document ID
# -------------------------
st.subheader("Step 1 — Load an existing indexed document")
doc_id_input = st.text_input("Enter document ID (e.g., UUID shown by admin)", value="")

if st.button("Load Document"):
    doc_id = doc_id_input.strip()
    if not doc_id:
        st.error("Please enter a document ID.")
    else:
        with st.spinner(f"Downloading index for {doc_id} from S3..."):
            try:
                vectorstore, tmp_dir = load_vectorstore_from_s3(doc_id)
            except Exception as e:
                st.error(f"Failed to download/load index for `{doc_id}`: {e}")
            else:
                qa_chain = build_qa_chain_from_vectorstore(vectorstore)
                st.session_state.doc_id = doc_id
                st.session_state.qa_chain = qa_chain
                st.session_state._tmp_dir_for_doc = tmp_dir
                st.success(f"Loaded document `{doc_id}`. You can now ask questions.")

# -------------------------
# UI: Ask a question
# -------------------------
if st.session_state.qa_chain:
    st.subheader(f"Step 2 — Ask questions (Document ID: {st.session_state.doc_id})")

    with st.form(key="qa_form"):
        question = st.text_input("Enter your question", value=st.session_state.get("question", ""), key="question_input")
        submit = st.form_submit_button("Ask")

    if submit:
        question = question.strip()
        if not question:
            st.warning("Please type a non-empty question.")
        else:
            st.session_state.question = question
            with st.spinner("Querying..."):
                try:
                    answer = st.session_state.qa_chain.run(question)
                except Exception as e:
                    st.error(f"Error while querying the model: {e}")
                else:
                    st.markdown("### Answer")
                    st.write(answer)
                    st.success("Done")

                    # Clear question input AFTER answer shown
                    st.session_state.question = ""

                    # Force Streamlit to rerun so cleared input shows (avoid experimental_rerun)
                    if hasattr(st, "experimental_request_rerun"):
                        st.experimental_request_rerun()
                    else:
                        # fallback: do nothing; input will clear next time user interacts
                        pass

# -------------------------
# Optional: cleanup button
# -------------------------
st.write("---")
if st.button("Unload current document"):
    st.session_state.qa_chain = None
    if "_tmp_dir_for_doc" in st.session_state:
        shutil.rmtree(st.session_state._tmp_dir_for_doc, ignore_errors=True)
        del st.session_state._tmp_dir_for_doc
    st.session_state.doc_id = None
    st.session_state.question = ""
    if hasattr(st, "experimental_request_rerun"):
        st.experimental_request_rerun()

