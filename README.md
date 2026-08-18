# Milkkyway – Work & Chill Space

RAG chatbot สำหรับธุรกิจพื้นที่นั่งทำงาน อ่านหนังสือ ประชุม และพักผ่อนแบบรายชั่วโมง พัฒนาต่อยอดจาก MilkLab° ในรายวิชา AI for Solopreneurs

## Demo

[เปิด Milkkyway บน Streamlit](https://milkkyway-9otlppbuxvcbph2sdn5v4t.streamlit.app/)

## ความสามารถ

- ตอบข้อมูลแพ็กเกจรายชั่วโมงและ Day Pass
- แนะนำโซนตามลักษณะการใช้งาน
- ตอบข้อมูลห้องประชุม การจอง และการยกเลิก
- อธิบายสิ่งอำนวยความสะดวก บริการเสริม และกฎของร้าน
- แสดงส่วนของฐานความรู้ที่นำมาใช้อ้างอิง
- ตอบโดยยึดเฉพาะข้อมูลที่ค้นได้จากฐานความรู้

## RAG pipeline

1. โหลดฐานความรู้จาก `menu_kb.md`
2. แบ่งเอกสารเป็น chunks ตามหัวข้อและย่อหน้า
3. สร้าง embeddings ด้วย `paraphrase-multilingual-MiniLM-L12-v2`
4. ค้นหา chunks ที่เกี่ยวข้องด้วย FAISS cosine similarity
5. ส่ง context ให้ Gemini 2.5 Flash สร้างคำตอบ

## เริ่มต้นใช้งาน

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
streamlit run app.py
```

สำหรับ Streamlit Community Cloud ให้ตั้ง `GEMINI_API_KEY` ใน App settings → Secrets โดยไม่ใส่ API key ลงใน Repository

## ไฟล์หลัก

| ไฟล์ | รายละเอียด |
|---|---|
| `app.py` | หน้า Streamlit และ RAG chatbot |
| `menu_kb.md` | ฐานความรู้ธุรกิจ Milkkyway |
| `PIVOT.md` | เอกสารอธิบายการเปลี่ยน Domain |
| `requirements.txt` | Python dependencies |

## เทคโนโลยี

- Python 3.11
- Streamlit
- Gemini API (`google-genai`)
- Sentence Transformers
- FAISS
