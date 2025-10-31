import os
import streamlit as st
import openai
import numpy as np
from typing import List, Dict

# streamlit_app.py
# GitHub Copilot
# Streamlit app that incorporates semantic search into LLM responses.
#
# Requirements: streamlit, openai, numpy
# Install: pip install streamlit openai numpy


st.set_page_config(page_title="Langgraph-style Search Assistant", layout="wide")

# -----------------------
# Helpers
# -----------------------
def set_openai_key(key: str):
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    # Simple chunker that respects newline boundaries where possible
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = current + "\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) <= max_chars:
                current = p
            else:
                # break long paragraph
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[np.ndarray]:
    # Calls OpenAI embedding API for a list of texts, returns numpy arrays
    resp = openai.Embedding.create(model=model, input=texts)
    embeds = [np.array(d["embedding"], dtype=np.float32) for d in resp["data"]]
    return embeds

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, index: List[Dict], top_k: int = 3, embed_model: str = "text-embedding-3-small"):
    if not index:
        return []
    q_emb = embed_texts([query], model=embed_model)[0]
    sims = []
    for item in index:
        sims.append((cosine_similarity(q_emb, item["embedding"]), item))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]

def build_system_prompt():
    return (
        "You are a helpful assistant that uses provided context snippets (clearly marked) to answer user questions. "
        "If the answer is not contained in the snippets, say so and provide a best-effort answer, and clearly label speculation. "
        "Cite snippet titles when you can."
    )

def generate_answer(question: str, context_snippets: List[Dict], model: str = "gpt-3.5-turbo"):
    # Compose a prompt that includes top-k context snippets
    system = build_system_prompt()
    context_text = ""
    for i, sn in enumerate(context_snippets, 1):
        context_text += f"\n--- SNIPPET {i} (source: {sn['title']}) ---\n{sn['chunk']}\n"
    user = f"Question: {question}\n\nUse only the information in the snippets above for facts. If you add additional info, label it as your own inference."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context_text + "\n\n" + user},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2, max_tokens=800)
    return resp.choices[0].message.content.strip()

# -----------------------
# Session storage
# -----------------------
if "index" not in st.session_state:
    st.session_state["index"] = []  # list of {title, chunk, embedding (np.array)}

# -----------------------
# UI
# -----------------------
st.title("Langgraph-style Search + LLM Assistant")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        set_openai_key(api_key)
    embed_model = st.selectbox("Embedding model", ["text-embedding-3-small"], index=0)
    gen_model = st.selectbox("Generation model", ["gpt-3.5-turbo"], index=0)
    top_k = st.slider("Top-k retrieved snippets", 1, 6, 3)
    st.markdown("---")
    if st.button("Clear indexed documents"):
        st.session_state["index"] = []
        st.success("Cleared index.")

st.header("Add / Index Documents")
with st.form("add_doc"):
    title = st.text_input("Document title", value="Untitled")
    text = st.text_area("Paste text to index (plain text or markdown). Short docs recommended for fast demo.", height=220)
    submit_index = st.form_submit_button("Index document")
    if submit_index:
        if not text.strip():
            st.error("Provide text to index.")
        elif not openai.api_key:
            st.error("Set OpenAI API key in the sidebar first.")
        else:
            chunks = chunk_text(text)
            embeddings = embed_texts(chunks, model=embed_model)
            for c, e in zip(chunks, embeddings):
                st.session_state["index"].append({"title": title, "chunk": c, "embedding": e})
            st.success(f"Indexed {len(chunks)} chunks from '{title}'. Total indexed chunks: {len(st.session_state['index'])}")

st.markdown("---")
st.header("Ask a question (answers will incorporate search results)")

query = st.text_input("Your question", "")
if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question.")
    elif not openai.api_key:
        st.error("Set OpenAI API key in the sidebar first.")
    else:
        if not st.session_state["index"]:
            st.info("No documents indexed. Answering without search.")
            # Direct LLM call without context
            resp = openai.ChatCompletion.create(
                model=gen_model,
                messages=[{"role": "system", "content": build_system_prompt()},
                          {"role": "user", "content": query}],
                temperature=0.2,
                max_tokens=800,
            )
            answer = resp.choices[0].message.content.strip()
            st.subheader("Answer")
            st.write(answer)
        else:
            results = retrieve(query, st.session_state["index"], top_k=top_k, embed_model=embed_model)
            snippets = [item for score, item in results]
            st.subheader("Retrieved snippets (with similarity)")
            for score, item in results:
                st.markdown(f"- **{item['title']}** — similarity {score:.3f}")
                st.caption(item["chunk"][:400] + ("…" if len(item["chunk"]) > 400 else ""))
            # Generate answer using snippets
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, snippets, model=gen_model)
            st.subheader("Answer")
            st.write(answer)
            st.markdown("---")
            st.subheader("Sources (full snippets)")
            for i, s in enumerate(snippets, 1):
                st.markdown(f"**Snippet {i} — {s['title']}**")
                st.code(s["chunk"][:4000], language="text")

st.markdown("---")
st.write("Notes: This simple demo embeds pasted documents and uses vector search to provide context to the LLM. It is not a production retrieval system. For larger corpora, consider chunking more carefully, persistent vector stores (FAISS, Milvus), and stronger RAG pipelines.")