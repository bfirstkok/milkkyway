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

    # แบ่งตามหัวข้อ Markdown เพื่อให้ชื่อหัวข้อและรายละเอียดอยู่ใน chunk เดียวกัน
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?=^##\s+)", document, flags=re.MULTILINE)
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


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 5) -> list[str]:
    """Return the knowledge-base chunks most relevant to a question."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    _, indices = index.search(query_embedding, min(k, len(chunks)))
    semantic_results = [chunks[idx] for idx in indices[0] if idx != -1]

    # คำถามโปรโมชันสั้น ๆ เช่น "มีโปรอะไร" มักมี semantic signal น้อย
    # จึงดัน chunk สรุปโปรโมชันขึ้นเป็นอันดับแรก แล้วค่อยตามด้วยผล semantic
    normalized_query = query.lower().strip()
    promotion_keywords = ("โปร", "promotion", "ส่วนลด", "คูปอง", "สิทธิ์")
    priority_results = []
    if any(keyword in normalized_query for keyword in promotion_keywords):
        priority_results = [
            chunk for chunk in chunks
            if "## สรุปโปรโมชันทั้งหมด" in chunk
        ]

    results = []
    for chunk in priority_results + semantic_results:
        if chunk not in results:
            results.append(chunk)
    return results[: max(k, 6 if priority_results else k)]


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
        .stApp { background: #f7f8fc; color: #182035; }
        header[data-testid="stHeader"] { background: rgba(247,248,252,.9); backdrop-filter: blur(12px); }
        [data-testid="stToolbar"] { color: #4b2e83; }
        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, rgba(247,248,252,0) 0%, #f7f8fc 35%, #f7f8fc 100%);
        }
        .block-container { max-width: 1040px; padding-top: 1.25rem; padding-bottom: 7rem; }
        .stApp h1, .stApp h2, .stApp h3, .stApp p,
        .stApp label, .stApp [data-testid="stCaptionContainer"] {
            color: #182035;
        }
        [data-testid="stChatMessage"] { background: #fff; border: 1px solid #e7e9f2; border-radius: 20px; color: #182035; box-shadow: 0 8px 28px rgba(34,38,70,.05); padding: .35rem .55rem; margin-bottom: .75rem; }
        [data-testid="stChatMessage"] p { color: #182035; line-height: 1.7; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #ececf4; }
        [data-testid="stSidebar"] * { color: #25233a; }
        [data-testid="stChatInput"] { background: white; color: #172033; border: 1px solid #dcd9eb; border-radius: 18px; box-shadow: 0 12px 35px rgba(54,44,99,.12); }
        [data-testid="stChatInput"] textarea { color: #172033; -webkit-text-fill-color: #172033; }
        [data-testid="stChatInput"] textarea::placeholder { color: #6b7280; opacity: 1; }
        .hero { position: relative; overflow: hidden; padding: 2rem 2.1rem; border-radius: 28px; color: white; background: linear-gradient(125deg,#3e2478 0%,#684bb1 50%,#348dd1 100%); box-shadow: 0 20px 50px rgba(73,53,139,.2); margin-bottom: 1rem; }
        .hero:after { content: ''; position: absolute; width: 230px; height: 230px; right: -70px; top: -105px; border-radius: 50%; background: rgba(255,255,255,.12); }
        .hero-badge { display: inline-flex; padding: .36rem .72rem; border-radius: 999px; background: rgba(255,255,255,.16); color: white; font-size: .82rem; font-weight: 700; margin-bottom: .75rem; }
        .hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.03em; color: white; }
        .hero p { margin: .55rem 0 0; opacity: .92; color: white; max-width: 680px; font-size: 1.02rem; }
        .stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: .75rem; margin: .9rem 0 1.55rem; }
        .stat-card { background: white; border: 1px solid #ececf4; border-radius: 18px; padding: 1rem; box-shadow: 0 7px 24px rgba(32,37,67,.04); }
        .stat-icon { font-size: 1.35rem; }
        .stat-label { color: #74788d; font-size: .78rem; margin-top: .4rem; }
        .stat-value { color: #20243a; font-weight: 800; font-size: 1rem; margin-top: .12rem; }
        .section-kicker { color: #684bb1; font-size: .8rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .15rem; }
        .section-title { font-size: 1.45rem; font-weight: 850; color: #20243a; margin-bottom: .25rem; }
        .section-subtitle { color: #74788d; font-size: .9rem; margin-bottom: .85rem; }
        .side-brand { background: linear-gradient(135deg,#442680,#6d55b8); border-radius: 18px; padding: 1.1rem; margin: .2rem 0 1rem; }
        .side-brand strong, .side-brand span { color: white !important; }
        .side-brand strong { display:block; font-size:1.18rem; }
        .side-brand span { display:block; opacity:.82; font-size:.8rem; margin-top:.18rem; }
        .side-info { background:#f7f5fc; border:1px solid #ebe6f5; border-radius:14px; padding:.75rem .85rem; margin:.55rem 0; font-size:.86rem; }
        .stButton > button { border: 1px solid #e6e1f1; background: #fff; color: #3f3563; border-radius: 14px; min-height: 54px; font-weight: 700; box-shadow: 0 5px 16px rgba(52,44,90,.04); transition: all .18s ease; }
        .stButton > button:hover { border-color: #7054b6; color: #563a9a; transform: translateY(-2px); box-shadow: 0 10px 24px rgba(85,61,151,.12); }
        div[data-testid="stExpander"] { border: 1px solid #ececf3; border-radius: 14px; background: #fafafd; }
        @media (max-width: 760px) { .stat-grid { grid-template-columns: repeat(2,1fr); } .hero { padding:1.4rem; } .hero h1 { font-size:1.9rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Milkkyway | Work & Chill Space",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-badge">OPEN DAILY · 09:00–24:00</div>
          <h1>พื้นที่ดี ๆ ให้งานไปได้ไกลกว่าเดิม</h1>
          <p>Milkkyway คือ Work & Chill Space สำหรับอ่านหนังสือ ทำงาน ประชุม หรือนั่งพัก พร้อม Wi-Fi และปลั๊กไฟ เริ่มเพียง 35 บาทต่อชั่วโมง</p>
        </div>
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-icon">⏱️</div><div class="stat-label">เริ่มต้น</div><div class="stat-value">35 บาท/ชั่วโมง</div></div>
          <div class="stat-card"><div class="stat-icon">📶</div><div class="stat-label">รวมในแพ็กเกจ</div><div class="stat-value">Wi-Fi + ปลั๊กไฟ</div></div>
          <div class="stat-card"><div class="stat-icon">👥</div><div class="stat-label">ห้องประชุม</div><div class="stat-value">รองรับ 2–6 คน</div></div>
          <div class="stat-card"><div class="stat-icon">🎓</div><div class="stat-label">สิทธิ์นักศึกษา</div><div class="stat-value">ลดสูงสุด 15%</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div class='side-brand'><strong>🌌 Milkkyway</strong><span>Work & Chill Space</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='side-info'>🕘 <b>เปิดทุกวัน</b><br>09:00–24:00 น.</div>", unsafe_allow_html=True)
        st.markdown("<div class='side-info'>📍 <b>ทำเล</b><br>บริเวณหน้ามหาวิทยาลัย</div>", unsafe_allow_html=True)
        st.markdown("<div class='side-info'>💬 <b>จองพื้นที่</b><br>ติดต่อผ่าน LINE OA</div>", unsafe_allow_html=True)
        st.divider()
        st.caption("NOVA CAN HELP")
        st.write("✓ เปรียบเทียบแพ็กเกจ\n\n✓ ค้นหาโปรโมชัน\n\n✓ แนะนำโซนและห้อง\n\n✓ ตอบกฎและการจอง")

    st.markdown("<div class='section-kicker'>QUICK START</div><div class='section-title'>อยากรู้อะไร เลือกถาม Nova ได้เลย</div><div class='section-subtitle'>กดหัวข้อด้านล่าง หรือพิมพ์คำถามของคุณในช่องแชต</div>", unsafe_allow_html=True)

    quick_query = None
    quick_columns = st.columns(4)
    quick_actions = [
        ("🎉 โปรโมชันวันนี้", "วันนี้มีโปรโมชันอะไรบ้าง และมีเงื่อนไขสำคัญอะไร"),
        ("🎓 สิทธิ์นักศึกษา", "ฉันเป็นนักศึกษา มีส่วนลดหรือแพ็กเกจอะไรที่คุ้มบ้าง"),
        ("👥 มาเป็นกลุ่ม", "มา 4 คน ใช้ 3 ชั่วโมง ควรเลือกแพ็กเกจหรือโปรโมชันอะไร"),
        ("⭐ สมาชิกและแต้ม", "สมาชิก Milkkyway มีสิทธิ์อะไร และสะสมดาวอย่างไร"),
    ]
    for column, (label, question) in zip(quick_columns, quick_actions):
        with column:
            if st.button(label, use_container_width=True):
                quick_query = question

    st.markdown("<br><div class='section-kicker'>ASK NOVA</div><div class='section-title'>ผู้ช่วยส่วนตัวของคุณที่ Milkkyway</div><div class='section-subtitle'>คำตอบอ้างอิงจากข้อมูลราคา โปรโมชัน โซน การจอง และกฎของร้าน</div>", unsafe_allow_html=True)

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

    typed_query = st.chat_input("ถาม Nova เรื่องราคา โปรโมชัน แพ็กเกจ โซน หรือการจอง...")
    query = quick_query or typed_query
    if query:
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
