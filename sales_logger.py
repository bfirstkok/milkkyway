"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.

นักศึกษาต้องเติม TODO ใน 4 จุดด้านล่างใน Session 2 Lab 1.3
"""

import argparse
import json
import os
import sys
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """บันทึกยอดขายหนึ่งรายการลง Google Sheets"""

    # Guardrails: ป้องกันข้อมูลผิดปกติ
    if not menu.strip():
        raise ValueError("ชื่อเมนูห้ามว่าง")
    if qty <= 0:
        raise ValueError("จำนวนสินค้าต้องมากกว่า 0")
    if price < 0:
        raise ValueError("ราคาห้ามติดลบ")

    credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_json:
        raise RuntimeError("ไม่พบ GOOGLE_SHEETS_CREDENTIALS")
    if not sheet_id:
        raise RuntimeError("ไม่พบ GOOGLE_SHEET_ID")

    try:
        credentials_info = json.loads(credentials_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SHEETS_CREDENTIALS ไม่ใช่ JSON ที่ถูกต้อง") from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=scopes,
    )

    client = gspread.authorize(credentials)
    worksheet = client.open_by_key(sheet_id).worksheet("Sales")

    timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(
        timespec="seconds"
    )
    total = round(qty * price, 2)

    row = {
        "timestamp": timestamp,
        "menu": menu.strip(),
        "qty": qty,
        "price": price,
        "total": total,
    }

    worksheet.append_row(
        [
            row["timestamp"],
            row["menu"],
            row["qty"],
            row["price"],
            row["total"],
        ],
        value_input_option="USER_ENTERED",
    )

    return row


def send_notification(message: str) -> str:
    """ส่งข้อความแจ้งเตือนผ่าน Telegram Bot"""

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        raise RuntimeError("ไม่พบ TELEGRAM_BOT_TOKEN")
    if not chat_id:
        raise RuntimeError("ไม่พบ TELEGRAM_CHAT_ID")
    if not message.strip():
        raise ValueError("ข้อความแจ้งเตือนห้ามว่าง")
    if len(message) > 4096:
        raise ValueError("ข้อความยาวเกินขีดจำกัดของ Telegram")

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": message,
        },
        timeout=20,
    )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Telegram ตอบกลับมาในรูปแบบที่ไม่ถูกต้อง") from exc

    if not response.ok or not result.get("ok"):
        error_message = result.get("description", "Unknown Telegram error")
        raise RuntimeError(f"ส่ง Telegram ไม่สำเร็จ: {error_message}")

    return "telegram"


def main() -> int:
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนขวด")
    parser.add_argument("--price", type=float,
                        required=True, help="ราคาต่อขวด")
    args = parser.parse_args()

    try:
        # TODO 3: เรียก append_to_sheet แล้ว extract total
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        # TODO 4: เรียก send_notification ด้วย message ที่บอกยอดที่บันทึก
        provider = send_notification(
            f"บันทึก {args.menu} x{args.qty} = {total} บาท")
    except Exception as exc:
        print(
            f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว: {exc}", file=sys.stderr)
        return 0

    print(f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())
