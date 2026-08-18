"""Milkkyway Work & Chill Space RAG chatbot.

Run locally:
    streamlit run app.py
"""

import os
import re
import time

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
    priority_results = []
    knowledge_routes = [
        (("โปร", "promotion", "ส่วนลด", "คูปอง", "สิทธิ์"), "## สรุปโปรโมชันทั้งหมด"),
        (("เครดิต", "เวลาเหลือ", "หมดอายุ", "โอน", "คืนเงิน"), "## ระบบเครดิตเวลา"),
        (("ราคา", "กี่บาท", "แพ็ก", "ชั่วโมง", "ชม.", "ชม "), "## แพ็กเกจที่นั่งทั่วไป"),
        (("check-in", "check out", "check-out", "เช็คอิน", "เช็คเอาท์", "qr", "ประตู"), "## ขั้นตอน Check-in และ Check-out"),
        (("คอม", "computer", "สเปก", "เล่นเกม", "โปรแกรม"), "## คอมพิวเตอร์ให้เช่า"),
        (("ที่อยู่", "อยู่ไหน", "แผนที่", "เปิดกี่โมง", "24 ชั่วโมง", "กลางคืน"), "## ที่ตั้งและเวลาเปิดบริการ"),
        (("จอดรถ", "รถยนต์", "มอเตอร์ไซค์", "เดินทาง"), "## ที่จอดรถและการเดินทาง"),
        (("ห้องน้ำ", "รถเข็น", "ผู้พิการ", "ทางเข้า", "wi-fi", "wifi", "ปลั๊ก"), "## สิ่งอำนวยความสะดวก"),
        (("ล็อกเกอร์", "ของหาย", "กล้อง", "ของมีค่า", "ความปลอดภัย"), "## ความปลอดภัย ล็อกเกอร์ และของหาย"),
        (("ไฟดับ", "เน็ตล่ม", "ระบบล่ม", "ชดเชย", "คืนเครดิต"), "## กรณีไฟดับ อินเทอร์เน็ตล่ม และระบบขัดข้อง"),
    ]
    for keywords, heading in knowledge_routes:
        if any(keyword in normalized_query for keyword in keywords):
            priority_results.extend(
                chunk for chunk in chunks if heading in chunk
            )

    results = []
    for chunk in priority_results + semantic_results:
        if chunk not in results:
            results.append(chunk)
    return results[: max(k, 6 if priority_results else k)]


def build_retrieval_query(query: str, messages: list[dict]) -> str:
    """Expand short follow-up questions with the previous user question."""
    normalized = query.lower().strip()
    follow_up_markers = (
        "ถ้า", "แล้ว", "ล่ะ", "ละ", "อันนี้", "แบบนี้", "เท่าไหร่",
        "กี่บาท", "เพิ่ม", "เอา ", "มัน", "ได้ไหม", "ได้มั้ย",
    )
    is_follow_up = len(normalized) <= 60 and any(
        marker in normalized for marker in follow_up_markers
    )
    if not is_follow_up:
        return query

    previous_questions = [
        message["content"]
        for message in messages
        if message.get("role") == "user" and message.get("content")
    ]
    if not previous_questions:
        return query
    return f"{previous_questions[-1]}\nคำถามต่อเนื่อง: {query}"


def format_conversation_history(messages: list[dict]) -> str:
    """Format a small recent window so Gemini can resolve follow-up wording."""
    lines = []
    for message in messages[-6:]:
        role = "ลูกค้า" if message.get("role") == "user" else "Nova"
        content = str(message.get("content", ""))[:1200]
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "ยังไม่มีบทสนทนาก่อนหน้า"


def get_api_key() -> str | None:
    """Read Gemini key from the environment or Streamlit Secrets."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        return None


def build_quota_fallback(context_chunks: list[str]) -> str:
    """Return useful retrieved facts when the LLM quota is temporarily full."""
    useful_chunks = context_chunks[:3]
    context_text = "\n\n---\n\n".join(useful_chunks)
    if len(context_text) > 5000:
        context_text = context_text[:5000].rsplit("\n", 1)[0] + "\n…"
    return (
        "ระบบสรุปด้วย AI ไม่พร้อมชั่วคราว แต่ Nova ค้นข้อมูลที่เกี่ยวข้อง "
        "จากฐานความรู้ของ Milkkyway ให้แล้วค่ะ\n\n"
        f"{context_text}\n\n"
        "หากต้องการคำตอบแบบสรุป กรุณารอประมาณ 1 นาทีแล้วลองถามอีกครั้งค่ะ"
    )


@st.cache_data(ttl=1800, show_spinner=False)
def request_gemini(prompt: str, api_key: str) -> str:
    """Cache only successful Gemini responses; raised errors are not cached."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text or ""


def generate_answer(
    query: str,
    context_chunks: list[str],
    conversation_history: str = "",
) -> str:
    """Generate a grounded answer, falling back to retrieved facts on any outage."""
    api_key = get_api_key()
    if not api_key:
        return build_quota_fallback(context_chunks)

    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
คุณชื่อ Nova เป็นผู้ช่วยประจำ Milkkyway – Work & Chill Space
พื้นที่เช่ารายชั่วโมงสำหรับนั่งทำงาน อ่านหนังสือ ประชุม และพักผ่อน

กติกาการตอบ:
- ใช้เฉพาะข้อมูลใน Context เท่านั้น ห้ามแต่งราคา เวลา หรือเงื่อนไขเอง
- ตอบเป็นภาษาเดียวกับผู้ใช้ ด้วยน้ำเสียงเป็นมิตร สุภาพ และกระชับ
- ช่วยเปรียบเทียบแพ็กเกจหรือคำนวณค่าใช้บริการได้เมื่อ Context มีข้อมูลครบ
- หากยังไม่มีระบบตรวจที่นั่งแบบเรียลไทม์ ให้แนะนำช่องทางยืนยันกับร้าน
- ใช้บทสนทนาก่อนหน้าเพื่อทำความเข้าใจคำถามต่อเนื่อง เช่น "ถ้า 24 ชั่วโมงล่ะ"
- หากคำถามต่อเนื่องยังคลุมเครือจนระบุบริการไม่ได้ ให้ถามกลับสั้น ๆ แทนการเดา
- ถ้า Context ไม่มีคำตอบ ให้ตอบว่า "ขออภัย Nova ยังไม่พบข้อมูลนี้ในฐานความรู้ของ Milkkyway กรุณาติดต่อพนักงานเพื่อยืนยันค่ะ"

บทสนทนาก่อนหน้า:
{conversation_history}

Context:
{context}

คำถามของลูกค้า:
{query}

คำตอบ:
"""
    for attempt, delay in enumerate((0, 2.5, 5.0), start=1):
        if delay:
            time.sleep(delay)
        try:
            answer = request_gemini(prompt, api_key)
            return answer or build_quota_fallback(context_chunks)
        except Exception as exc:
            error_text = str(exc)
            is_quota_error = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
            if is_quota_error and attempt < 3:
                continue
            if is_quota_error:
                return build_quota_fallback(context_chunks)
            return build_quota_fallback(context_chunks)

    return build_quota_fallback(context_chunks)


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
          <div class="hero-badge">OPEN 24 HOURS · EVERY DAY</div>
          <h1>พื้นที่ดี ๆ ให้งานไปได้ไกลกว่าเดิม</h1>
          <p>Milkkyway คือ Work & Chill Space สำหรับอ่านหนังสือ ทำงาน ประชุม หรือนั่งพัก พร้อม Wi-Fi และปลั๊กไฟ เริ่มเพียง 35 บาทต่อชั่วโมง</p>
        </div>
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-icon">🌙</div><div class="stat-label">เวลาเปิดบริการ</div><div class="stat-value">24 ชั่วโมงทุกวัน</div></div>
          <div class="stat-card"><div class="stat-icon">📶</div><div class="stat-label">รวมในแพ็กเกจ</div><div class="stat-value">Wi-Fi + ปลั๊กไฟ</div></div>
          <div class="stat-card"><div class="stat-icon">👥</div><div class="stat-label">พื้นที่ทั้งหมด</div><div class="stat-value">รองรับ 50 คน</div></div>
          <div class="stat-card"><div class="stat-icon">🎓</div><div class="stat-label">สิทธิ์นักศึกษา</div><div class="stat-value">ลดสูงสุด 15%</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div class='side-brand'><strong>🌌 Milkkyway</strong><span>Work & Chill Space</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='side-info'>🌙 <b>เปิด 24 ชั่วโมง</b><br>ทุกวัน ไม่มีเวลาปิด</div>", unsafe_allow_html=True)
        st.markdown("<div class='side-info'>📍 <b>บึงแก่นนคร ขอนแก่น</b><br>99/95 ซอย 12 พิมานชล 2</div>", unsafe_allow_html=True)
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
        recent_messages = st.session_state.messages[-6:]
        retrieval_query = build_retrieval_query(query, recent_messages)
        conversation_history = format_conversation_history(recent_messages)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Nova กำลังค้นข้อมูล..."):
                context = retrieve_top_k(retrieval_query, model, index, chunks)
                answer = generate_answer(query, context, conversation_history)
            st.write(answer)
            with st.expander("ดูแหล่งข้อมูลที่ใช้อ้างอิง"):
                for number, chunk in enumerate(context, 1):
                    st.markdown(f"**แหล่งข้อมูล {number}**")
                    st.write(chunk)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
