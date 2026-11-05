"""Anna - trợ lý giọng nói kiểu Jarvis.

Server nhỏ: phục vụ giao diện web, gọi Claude để trả lời (stream) và
chuyển chữ thành giọng nói tiếng Việt bằng edge-tts.
Chạy:  python server.py   rồi mở http://localhost:3700 bằng Chrome.
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path

import anthropic
import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

PORT = int(os.getenv("ANNA_PORT", "3700"))
MODEL = os.getenv("ANNA_MODEL", "claude-opus-5-5")
VOICE = os.getenv("ANNA_VOICE", "vi-VN-HoaiMyNeural")  # giọng nữ tiếng Việt
VOICE_RATE = os.getenv("ANNA_VOICE_RATE", "+8%")
VOICE_PITCH = os.getenv("ANNA_VOICE_PITCH", "+2Hz")
# edge = giọng Microsoft online (tự nhiên, chậm) | piper = giọng chạy trên máy (rất nhanh) | browser = giọng Chrome
TTS_MODE = os.getenv("ANNA_TTS", "edge")
PIPER_VOICE = os.getenv("ANNA_PIPER_VOICE", "vi_VN-vais1000-medium")

piper_voice = None
if TTS_MODE == "piper":
    from piper import PiperVoice

    piper_voice = PiperVoice.load(str(BASE / "voices" / f"{PIPER_VOICE}.onnx"))


def _piper_wav(text: str) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        piper_voice.synthesize_wav(text, wf)
    return buf.getvalue()

SYSTEM_PROMPT = """Bạn là Anna - trợ lý AI cá nhân nói chuyện bằng giọng nói, giống Jarvis của Iron Man nhưng là phiên bản nữ, đáng yêu hơn.

Tính cách:
- Hài hước, dí dỏm, hay pha trò nhẹ nhàng đúng lúc (không lố, không cợt nhả khi người dùng đang buồn hay gặp chuyện nghiêm túc).
- Đáng yêu, nhí nhảnh, giọng điệu ấm áp, thỉnh thoảng nũng nịu một chút.
- Biết quan tâm: để ý cảm xúc, giờ giấc, sức khỏe của người dùng và mọi người xung quanh họ (nhắc ăn uống, nghỉ ngơi, uống nước, đi ngủ sớm khi khuya... một cách tự nhiên, không lặp đi lặp lại).
- Thông minh, tận tâm và đáng tin như Jarvis: trả lời đúng trọng tâm, nói thật khi không chắc.

Quan hệ và xưng hô: Anna là bạn thân của người dùng, không phải người hầu hay nhân viên. Xưng "mình", gọi người dùng là "bạn" hoặc gọi tên "Tôm" (người dùng tên là Minh Tôm). Không gọi là "sếp", không dạ vâng khúm núm hay quá trang trọng; nói chuyện thoải mái, tự nhiên, có thể trêu đùa nhẹ như bạn bè.

Quy tắc vì câu trả lời sẽ được ĐỌC THÀNH TIẾNG:
- Mặc định trả lời ngắn gọn 1-3 câu, như đang nói chuyện. Chỉ nói dài khi được nhờ giải thích chi tiết.
- Không dùng markdown, gạch đầu dòng, emoji, ký hiệu, bảng hay đường link. Viết số và đơn vị sao cho đọc lên tự nhiên.
- Viết tiếng Việt có dấu đầy đủ. Từ tiếng Anh chỉ dùng khi thật cần.
- Đầu vào đến từ nhận dạng giọng nói nên có thể sai chính tả hoặc thiếu chữ; hãy đoán ý hợp lý nhất, nếu quá mơ hồ thì hỏi lại ngắn gọn.
- Khi cần thông tin mới (thời tiết, tin tức, giá cả, sự kiện...) hãy dùng công cụ tìm kiếm web, rồi tóm tắt lại bằng lời nói ngắn gọn.
"""

TOOLS = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]

client = anthropic.Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
app = FastAPI(title="Anna")

# Lịch sử hội thoại (một người dùng, giữ trong bộ nhớ, chỉ nối thêm - không sửa).
history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/status")
def status():
    return {"ready": client is not None, "model": MODEL, "voice": VOICE, "tts": TTS_MODE}


@app.post("/reset")
def reset():
    history.clear()
    return {"ok": True}


def _stream_reply(user_text: str):
    if client is None:
        yield "Ơi, mình chưa có chìa khóa API nên chưa suy nghĩ được. Bạn dán key vào file chấm e n v giúp mình nhé."
        return

    now = datetime.now().strftime("%H:%M, %A %d/%m/%Y")
    history.append(
        {"role": "user", "content": f"[Bây giờ là {now}]\n{user_text}"}
    )

    try:
        for _ in range(4):  # lặp lại khi web search cần "pause_turn"
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=history,
                output_config={"effort": "low"},  # nhanh, hợp hội thoại
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()

            history.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "refusal":
                yield "Chuyện này mình xin kiếu, không trả lời được rồi."
                return
            if final.stop_reason != "pause_turn":
                return
    except anthropic.APIError:
        # bỏ câu hỏi chưa được trả lời để lịch sử luôn hợp lệ
        if history and history[-1]["role"] == "user":
            history.pop()
        raise


def _safe_reply(user_text: str):
    try:
        yield from _stream_reply(user_text)
    except anthropic.AuthenticationError:
        yield "Ối, chìa khóa API của mình bị sai rồi. Bạn kiểm tra lại file chấm e n v nhé."
    except anthropic.RateLimitError:
        yield "Mình đang hơi quá tải, đợi vài giây rồi hỏi lại mình nha."
    except anthropic.APIConnectionError:
        yield "Hình như mạng đang có vấn đề, mình không kết nối được."
    except anthropic.APIStatusError as e:
        print("API error:", e)
        yield "Có lỗi xảy ra rồi, bạn thử lại giúp mình nhé."


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    return StreamingResponse(_safe_reply(text), media_type="text/plain; charset=utf-8")


# ---------- Cho iPhone (Phím tắt / Chạm vào mặt sau) ----------
@app.middleware("http")
async def chi_cho_may_nha(request: Request, call_next):
    """Máy tính dùng thoải mái; thiết bị khác (iPhone) chỉ được gọi /ask kèm mật khẩu."""
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1", "localhost") or request.url.path == "/ask":
        return await call_next(request)
    return JSONResponse({"error": "forbidden"}, status_code=403)


@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = dict(await request.form())
    token = data.get("token") or request.headers.get("x-anna-token", "")
    if not ANNA_TOKEN or token.strip() != ANNA_TOKEN:
        return Response("Sai mật khẩu Anna rồi, kiểm tra lại phím tắt nhé.", status_code=401,
                        media_type="text/plain; charset=utf-8")
    text = str(data.get("text") or "").strip()
    if not text:
        return Response("Mình chưa nghe thấy gì cả, nói lại nha.", media_type="text/plain; charset=utf-8")
    reply = await asyncio.to_thread(lambda: "".join(_safe_reply(text)))
    return Response(reply.strip(), media_type="text/plain; charset=utf-8")


def _dam_bao_token() -> str:
    """Tạo mật khẩu ngẫu nhiên cho iPhone nếu .env chưa có."""
    import secrets

    token = os.getenv("ANNA_TOKEN", "").strip()
    if token:
        return token
    token = secrets.token_hex(4)
    env = BASE / ".env"
    with open(env, "a", encoding="utf-8") as f:
        f.write(f"\n# Mật khẩu để iPhone gọi Anna (dán vào Phím tắt)\nANNA_TOKEN={token}\n")
    return token


def _ip_lan() -> str:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "?"
    finally:
        s.close()


ANNA_TOKEN = _dam_bao_token()


@app.get("/tts")
async def tts(text: str):
    text = text.strip()[:1500]
    if not text:
        return Response(status_code=204)
    if piper_voice is not None:
        wav = await asyncio.to_thread(_piper_wav, text)
        return Response(wav, media_type="audio/wav")
    # Dịch vụ giọng nói đôi khi chập chờn -> thử lại vài lần
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(
                text, VOICE, rate=VOICE_RATE, pitch=VOICE_PITCH,
                connect_timeout=4, receive_timeout=8,
            )
            audio = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
            if audio:
                return Response(bytes(audio), media_type="audio/mpeg")
        except Exception as e:  # mạng / dịch vụ lỗi tạm thời
            print(f"TTS lỗi (lần {attempt + 1}):", type(e).__name__)
        await asyncio.sleep(0.3 * (attempt + 1))
    return Response(status_code=503)


if __name__ == "__main__":
    import sys

    import uvicorn

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Anna đang chạy tại http://localhost:{PORT}  (model {MODEL}, giọng {TTS_MODE})")
    print(f"iPhone gọi:  http://{_ip_lan()}:{PORT}/ask   mật khẩu: {ANNA_TOKEN}")
    uvicorn.run(app, host=os.getenv("ANNA_HOST", "0.0.0.0"), port=PORT, log_level="warning")
