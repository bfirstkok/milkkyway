"""Milkkyway Work & Chill Space RAG chatbot.

Run locally:
    streamlit run app.py
"""

import os
import re

import faiss
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


BUSINESS_NAME = "Milkkyway"
KNOWLEDGE_BASE_PATH = "menu_kb.md"


@st.cache_resource
def load_index():
    """Load, chunk, embed, and index the business knowledge base."""
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        document = file.read()

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", document)
        if chunk.strip()
    ]
    if not chunks:
        raise ValueError(f"ไม่พบข้อมูลใน {KNOWLEDGE_BASE_PATH}")

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 4) -> list[str]:
    """Return the knowledge-base chunks most relevant to a question."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    _, indices = index.search(query_embedding, min(k, len(chunks)))
    return [chunks[idx] for idx in indices[0] if idx != -1]


def get_api_key() -> str | None:
    """Read Gemini key from the environment or Streamlit Secrets."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        return None


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Generate a grounded answer from retrieved Milkkyway information."""
    api_key = get_api_key()
    if not api_key:
        return "ไม่พบ GEMINI_API_KEY กรุณาตั้งค่าใน Streamlit Secrets ก่อนใช้งาน"

    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
คุณชื่อ Nova เป็นผู้ช่วยประจำ Milkkyway – Work & Chill Space
พื้นที่เช่ารายชั่วโมงสำหรับนั่งทำงาน อ่านหนังสือ ประชุม และพักผ่อน

กติกาการตอบ:
- ใช้เฉพาะข้อมูลใน Context เท่านั้น ห้ามแต่งราคา เวลา หรือเงื่อนไขเอง
- ตอบเป็นภาษาเดียวกับผู้ใช้ ด้วยน้ำเสียงเป็นมิตร สุภาพ และกระชับ
- ช่วยเปรียบเทียบแพ็กเกจหรือคำนวณค่าใช้บริการได้เมื่อ Context มีข้อมูลครบ
- หากยังไม่มีระบบตรวจที่นั่งแบบเรียลไทม์ ให้แนะนำช่องทางยืนยันกับร้าน
- ถ้า Context ไม่มีคำตอบ ให้ตอบว่า "ขออภัย Nova ยังไม่พบข้อมูลนี้ในฐานความรู้ของ Milkkyway กรุณาติดต่อพนักงานเพื่อยืนยันค่ะ"

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
        return response.text or "ขออภัย ระบบไม่ได้รับคำตอบจาก Gemini"
    except Exception as exc:
        return f"เกิดข้อผิดพลาดในการเรียก Gemini: {exc}"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #f7f4ff 0%, #eef7ff 55%, #fff 100%); color: #172033; }
        header[data-testid="stHeader"] { background: rgba(247,244,255,.92); }
        [data-testid="stToolbar"] { color: #4b2e83; }
        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, rgba(238,247,255,0) 0%, #eef7ff 32%, #eef7ff 100%);
        }
        .block-container { max-width: 920px; padding-top: 2rem; }
        .stApp h1, .stApp h2, .stApp h3, .stApp p,
        .stApp label, .stApp [data-testid="stCaptionContainer"] {
            color: #172033;
        }
        [data-testid="stChatMessage"] { background: rgba(255,255,255,.88); border: 1px solid #e6e0f4; border-radius: 18px; color: #172033; }
        [data-testid="stChatMessage"] p { color: #172033; }
        [data-testid="stSidebar"] { background: #f1edfb; }
        [data-testid="stSidebar"] * { color: #222033; }
        [data-testid="stChatInput"] { background: white; color: #172033; border: 1px solid #d9d0ed; }
        [data-testid="stChatInput"] textarea { color: #172033; -webkit-text-fill-color: #172033; }
        [data-testid="stChatInput"] textarea::placeholder { color: #6b7280; opacity: 1; }
        .hero { padding: 1.35rem 1.5rem; border-radius: 24px; color: white; background: linear-gradient(120deg,#4b2e83,#7257b5 55%,#3f83c5); box-shadow: 0 14px 35px rgba(75,46,131,.18); margin-bottom: 1.2rem; }
        .hero h1 { margin: 0; font-size: 2.15rem; color: white; }
        .hero p { margin: .45rem 0 0; opacity: .92; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Milkkyway | Work & Chill Space", page_icon="🌌")
    apply_theme()

    st.markdown(
        """
        <div class="hero">
          <h1>🌌 Milkkyway</h1>
          <p>Work & Chill Space · จะทำงาน อ่านหนังสือ ประชุม หรือนั่งพัก ก็จ่ายตามเวลาที่ใช้</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Milkkyway")
        st.write("พื้นที่รายชั่วโมงสำหรับนักศึกษา ฟรีแลนซ์ และคนทำงานออนไลน์")
        st.markdown("**เปิดทุกวัน:** 09:00–24:00 น.")
        st.markdown("**เริ่มต้น:** 35 บาท/ชั่วโมง")
        st.divider()
        st.caption("ลองถาม Nova")
        st.markdown("• มีแพ็กเกจอะไรบ้าง\n\n• จะประชุม 4 คนควรเลือกโซนไหน\n\n• เปิดถึงกี่โมง\n\n• มีปลั๊กและ Wi-Fi ไหม")

    st.subheader("คุยกับ Nova ผู้ช่วยของ Milkkyway")
    st.caption("คำตอบอ้างอิงจากข้อมูลแพ็กเกจ โซน การจอง และกฎของร้าน")

    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"ไม่สามารถโหลดฐานความรู้ได้: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "สวัสดีค่ะ ฉันชื่อ Nova 🌟 วันนี้อยากหาที่นั่งทำงาน อ่านหนังสือ ประชุม หรือนั่งพักแบบไหนคะ?",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("ถามเรื่องราคา แพ็กเกจ โซน หรือการจอง..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Nova กำลังค้นข้อมูล..."):
                context = retrieve_top_k(query, model, index, chunks)
                answer = generate_answer(query, context)
            st.write(answer)
            with st.expander("ดูแหล่งข้อมูลที่ใช้อ้างอิง"):
                for number, chunk in enumerate(context, 1):
                    st.markdown(f"**แหล่งข้อมูล {number}**")
                    st.write(chunk)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
