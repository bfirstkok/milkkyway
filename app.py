"""MilkLab RAG Chatbot (S3).

Run locally:
    streamlit run app.py
"""

import os
import re

import faiss
import numpy as np
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_index():
    """Load menu_kb.md, split into chunks, encode, and create FAISS index."""

    # TODO 1: โหลด knowledge base
    with open("menu_kb.md", "r", encoding="utf-8") as file:
        document = file.read()

    # TODO 2: แบ่งเอกสารเป็น chunk
    # แบ่งตามหัวข้อ/ย่อหน้า เพื่อให้แต่ละ chunk ยังมีความหมายครบ
    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", document)
        if chunk.strip()
    ]

    if not chunks:
        raise ValueError("ไม่พบข้อมูลใน menu_kb.md")

    # TODO 3: สร้าง embedding
    # multilingual model รองรับภาษาไทยและภาษาอื่น ๆ
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ใช้ Inner Product กับ normalized embeddings = cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return model, index, chunks


def retrieve_top_k(
    query: str,
    model,
    index,
    chunks: list[str],
    k: int = 3,
) -> list[str]:
    """Encode query, search FAISS index, and return top-k chunks."""

    # TODO 4
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ป้องกันกรณีจำนวน chunk น้อยกว่า k
    k = min(k, len(chunks))

    scores, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Send query + retrieved context to Gemini."""

    # TODO 5
    api_key = os.getenv("GEMINI_API_KEY")

    # รองรับ Streamlit secrets ตอน deploy
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass

    if not api_key:
        return (
            "ไม่พบ GEMINI_API_KEY กรุณาตั้งค่า API key "
            "ใน environment variable หรือ Streamlit Secrets"
        )

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
คุณคือผู้ช่วยของร้าน MilkLab

ตอบคำถามโดยใช้เฉพาะข้อมูลใน Context ที่ให้มาเท่านั้น
ห้ามแต่งข้อมูลเพิ่มเติมเอง

ถ้า Context ไม่มีข้อมูลเพียงพอสำหรับตอบคำถาม
ให้ตอบว่า "ขออภัย ไม่พบข้อมูลนี้ในฐานข้อมูลของ MilkLab"

ตอบเป็นภาษาเดียวกับที่ผู้ใช้ถาม
ตอบให้กระชับและเข้าใจง่าย

Context:
{context}

คำถามของลูกค้า:
{query}

คำตอบ:
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as exc:
        return f"เกิดข้อผิดพลาดในการเรียก Gemini: {exc}"


def main():
    st.set_page_config(
        page_title="MilkLab° RAG",
        page_icon="🥛",
    )

    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"ไม่สามารถโหลด Knowledge Base ได้: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(
                    prompt,
                    model,
                    index,
                    chunks,
                    k=3,
                )

                answer = generate_answer(
                    prompt,
                    context,
                )

            st.write(answer)

            with st.expander("Source chunks"):
                for i, chunk in enumerate(context, 1):
                    st.markdown(f"**[{i}]**")
                    st.write(chunk)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__ == "__main__":
    main()