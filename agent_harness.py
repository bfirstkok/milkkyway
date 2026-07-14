"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google import genai
import gspread
from google.oauth2.service_account import Credentials
from sales_logger import append_to_sheet, send_notification


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """ให้ Gemini เลือก tool และสร้าง arguments จากคำสั่งผู้ใช้"""

    if not cmd.strip():
        raise ValueError("คำสั่งห้ามว่าง")

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("ไม่พบ GOOGLE_API_KEY")

    client = genai.Client(api_key=key)

    prompt = f"""
คุณคือ Agent สำหรับร้าน MilkLab

หน้าที่ของคุณคือเลือกเครื่องมือที่เหมาะสมจาก TOOL_SCHEMA
แล้วตอบเป็น JSON เท่านั้น ห้ามใส่ Markdown หรือคำอธิบายอื่น

รูปแบบคำตอบ:
{{
  "tool": "ชื่อเครื่องมือ",
  "args": {{
    "ชื่อ argument": "ค่า"
  }}
}}

เครื่องมือที่ใช้งานได้:
{json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2)}

คำสั่งจากผู้ใช้:
{cmd}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"เรียก Gemini ไม่สำเร็จ: {exc}") from exc

    if not response.text:
        raise RuntimeError("Gemini ไม่ส่งคำตอบกลับมา")

    try:
        tool_call = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini ตอบกลับมาไม่ใช่ JSON") from exc

    # Guardrail: รูปแบบต้องมี tool และ args
    if not isinstance(tool_call, dict):
        raise RuntimeError("Tool call ต้องเป็น object")
    if not isinstance(tool_call.get("args"), dict):
        raise RuntimeError("args ต้องเป็น object")

    allowed_tools = {tool["name"]: tool for tool in TOOL_SCHEMA}
    tool_name = tool_call.get("tool")

    # Guardrail: ห้ามเรียก tool นอกเหนือจากที่อนุญาต
    if tool_name not in allowed_tools:
        raise RuntimeError(f"ไม่อนุญาตให้เรียก tool: {tool_name}")

    required_args = allowed_tools[tool_name]["parameters"]["required"]
    missing_args = [
        name for name in required_args
        if name not in tool_call["args"]
    ]

    # Guardrail: arguments ต้องมาครบ
    if missing_args:
        raise RuntimeError(
            f"arguments ไม่ครบ: {', '.join(missing_args)}"
        )

    return tool_call


def get_sales_worksheet():
    """เปิด Worksheet ชื่อ Sales ด้วย Service Account"""

    credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_json:
        raise RuntimeError("ไม่พบ GOOGLE_SHEETS_CREDENTIALS")
    if not sheet_id:
        raise RuntimeError("ไม่พบ GOOGLE_SHEET_ID")

    try:
        credentials_info = json.loads(credentials_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Service Account JSON ไม่ถูกต้อง") from exc

    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )

    client = gspread.authorize(credentials)
    return client.open_by_key(sheet_id).worksheet("Sales")


def dispatch_tool(tool_call: dict) -> str:
    """ตรวจสอบและเรียกใช้ tool ที่ Agent เลือก"""

    tool_name = tool_call.get("tool")
    args = tool_call.get("args")

    if not isinstance(args, dict):
        raise ValueError("args ต้องเป็น object")

    if tool_name == "log_sale":
        menu = args.get("menu")
        qty = args.get("qty")
        price = args.get("price")

        # Guardrails ก่อนเรียกเครื่องมือจริง
        if not isinstance(menu, str) or not menu.strip():
            raise ValueError("ชื่อเมนูไม่ถูกต้อง")
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            raise ValueError("จำนวนต้องเป็นจำนวนเต็มที่มากกว่า 0")
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            raise ValueError("ราคาต้องเป็นตัวเลข")
        if price < 0:
            raise ValueError("ราคาห้ามติดลบ")

        row = append_to_sheet(menu, qty, float(price))
        send_notification(
            f"Agent บันทึก {row['menu']} x{row['qty']} "
            f"= {row['total']} บาท"
        )

        return (
            f"บันทึกยอดขาย {row['menu']} จำนวน {row['qty']} "
            f"ยอดรวม {row['total']} บาทเรียบร้อย"
        )

    if tool_name == "query_sales":
        query_date = args.get("date")

        if not isinstance(query_date, str):
            raise ValueError("วันที่ต้องเป็นข้อความรูปแบบ YYYY-MM-DD")

        worksheet = get_sales_worksheet()
        records = worksheet.get_all_records()

        matched_records = [
            record
            for record in records
            if str(record.get("timestamp", "")).startswith(query_date)
        ]

        total = sum(
            float(record.get("total", 0) or 0)
            for record in matched_records
        )

        return (
            f"วันที่ {query_date} มี {len(matched_records)} รายการ "
            f"ยอดรวม {round(total, 2)} บาท"
        )

    if tool_name == "send_alert":
        message = args.get("message")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("ข้อความแจ้งเตือนห้ามว่าง")

        provider = send_notification(message.strip())
        return f"ส่งข้อความแจ้งเตือนผ่าน {provider} เรียบร้อย"

    raise RuntimeError(f"ไม่อนุญาตให้เรียก tool: {tool_name}")


TRACE_LOG_PATH = Path("agent_trace.log")
TRACE_EVENTS = {
    "user_input",
    "llm_response",
    "tool_result",
    "tool_error",
}


def write_trace(event_type: str, payload: object) -> None:
    """บันทึก Agent trace แบบหนึ่งเหตุการณ์ต่อหนึ่งบรรทัด"""

    if event_type not in TRACE_EVENTS:
        raise ValueError(f"Unknown trace event: {event_type}")

    if isinstance(payload, (dict, list)):
        message = json.dumps(payload, ensure_ascii=False)
    else:
        message = str(payload)

    message = message.replace("\r", " ").replace("\n", " ")
    timestamp = datetime.now(
        ZoneInfo("Asia/Bangkok")
    ).strftime("%Y-%m-%d %H:%M:%S")

    with TRACE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} | {event_type} | {message}\n")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")
    write_trace("user_input", args.cmd)

    try:
        tool_call = parse_command(args.cmd)
        print(f"[LLM]  tool={tool_call['tool']} args={tool_call['args']}")
        write_trace("llm_response", tool_call)

        result = dispatch_tool(tool_call)
        print(f"[TOOL] {tool_call['tool']} {result}")
        print(f"[USER] ← {result}")
        write_trace("tool_result", result)

        return 0

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {error_message}", file=sys.stderr)
        write_trace("tool_error", error_message)
        return 1
