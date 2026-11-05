# Anna

Trợ lý giọng nói tiếng Việt kiểu Jarvis: gọi "Anna ơi" là Anna đáp lại. Anna hài hước, đáng yêu, biết quan tâm, nói chuyện như bạn bè và có trí nhớ lâu dài.

- Bộ não: Claude (Anthropic API), tự tìm web khi cần thông tin mới
- Nghe: Web Speech API của Chrome (vi-VN), từ khóa gọi "Anna"
- Nói: edge-tts (Microsoft, mặc định giọng nam trầm NamMinh), Piper (offline) hoặc giọng trình duyệt
- Trí nhớ: Anna tự ghi/quên vào `anna_memory.json`; xem và xóa bằng nút "Trí nhớ"
- iPhone: gõ hai lần vào mặt lưng (Chạm vào mặt sau) → Phím tắt → `POST /ask`

## Cài đặt

```
pip install -r requirements.txt
copy .env.example .env      # rồi dán ANTHROPIC_API_KEY vào .env
python server.py            # hoặc bấm "Bat Anna.bat"
```

Mở http://localhost:3700 bằng Chrome, bấm KÍCH HOẠT, rồi gọi "Anna ơi".

Giọng Piper (tùy chọn, chạy offline): `python -m piper.download_voices --download-dir voices vi_VN-vais1000-medium`, rồi đặt `ANNA_TTS=piper` trong `.env`.

## iPhone

Lần chạy đầu, server tự tạo `ANNA_TOKEN` trong `.env` và in ra địa chỉ `/ask`. Tạo Phím tắt gồm: Đọc chính tả văn bản → Nhận nội dung của URL (POST, JSON: `text`, `token`) → Đọc văn bản; rồi gán vào Cài đặt → Trợ năng → Cảm ứng → Chạm vào mặt sau. iPhone phải dùng chung Wi-Fi với máy tính.

`.env` (API key, mật khẩu) và `anna_memory.json` (trí nhớ) không bao giờ được đưa lên git.
