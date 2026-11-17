# Face Detection High-Performance C++ Service

Service nhận diện khuôn mặt thời gian thực sử dụng mô hình **YuNet** (ONNX), viết hoàn toàn bằng **C++17** và OpenCV `FaceDetectorYN` nhằm tối ưu hóa tối đa hiệu năng, độ trễ và tài nguyên (RAM < 50MB, CPU inference ~5-15ms). Đóng gói hoàn chỉnh bằng **Docker**, hỗ trợ đa kiến trúc (`linux/amd64`, `linux/arm64` cho Raspberry Pi, VPS, server...) và sẵn sàng kết nối qua **ngrok**.

---

## 🚀 Luồng hoạt động (Workflow)

```
[ Client ] 
   │  Gửi ảnh (Multipart / Base64 JSON / Raw Binary)
   ▼
[ Ngrok Public HTTPS / Local Port 8085 ]
   │
   ▼
[ C++ HTTP Service (httplib) ]
   │
   ├─► 1. Tiền xử lý ảnh (Decode JPEG/PNG/WebP, chuẩn hóa kích thước)
   ├─► 2. YuNet Model (`cv::FaceDetectorYN` suy luận)
   └─► 3. Đóng gói kết quả tọa độ Bounding Box & 5 Landmarks
   │
   ▼
[ JSON Response ]
```

---

## 📋 Cấu trúc phản hồi JSON

### 1. Khi phát hiện khuôn mặt (`success: true`)
```json
{
  "success": true,
  "count": 2,
  "image_width": 1280,
  "image_height": 720,
  "faces": [
    {
      "confidence": 0.982,
      "bbox": {
        "x": 120.5,
        "y": 80.2,
        "width": 160.4,
        "height": 185.7
      },
      "landmarks": [
        [158.2, 140.1],
        [226.4, 139.8],
        [193.1, 174.5],
        [166.8, 205.2],
        [219.1, 204.8]
      ]
    },
    {
      "confidence": 0.945,
      "bbox": {
        "x": 520.3,
        "y": 110.2,
        "width": 145.0,
        "height": 170.0
      },
      "landmarks": [
        [550.0, 165.0],
        [610.0, 164.0],
        [580.0, 195.0],
        [558.0, 225.0],
        [603.0, 224.0]
      ]
    }
  ]
}
```

*Ghi chú 5 điểm landmarks theo chuẩn:*
1. `[x, y]` mắt phải (Right eye)
2. `[x, y]` mắt trái (Left eye)
3. `[x, y]` chóp mũi (Nose tip)
4. `[x, y]` khóe miệng phải (Right mouth corner)
5. `[x, y]` khóe miệng trái (Left mouth corner)

### 2. Khi không phát hiện khuôn mặt (`success: false`)
```json
{
  "success": false,
  "count": 0,
  "image_width": 1280,
  "image_height": 720,
  "faces": []
}
```

---

## 🐳 Hướng dẫn triển khai Docker

### Cách 1: Chạy trực tiếp với Docker (Khuyên dùng)

1. **Build Docker image:**
   ```bash
   cd FaceDetect_service
   docker compose build yunet-service 
   hoặc
   docker compose build --no-cache yunet-service
   ```

2. **Chạy container:**
   ```bash
   docker compose up -d yunet-service
   ```

3. **Build và chạy:**
   ```bash
   docker compose up -d --build yunet-service
   ```

4. **Kiểm tra trạng thái:**
   ```bash
   curl.exe http://localhost:8085/health
   ```
   Kết quả trả về:
   ```json
   {
     "status": "ok",
     "service": "yunet-face-detection",
     "engine": "C++ OpenCV FaceDetectorYN"
   }
   ```
5. **Chạy thử một ảnh**
  ```Bash
  curl.exe -X POST "http://localhost:8085/detect" -F "file=@author.jpg"
  ```
---

### Cách 2: Chạy ngrok cài trên máy trỏ vào Docker

Nếu bạn đã có sẵn ngrok trên máy tính:
1. Khởi động Docker service trên cổng 8085:
   ```bash
   docker compose up -d yunet-service
   ```
2. Mở terminal chạy lệnh:
   ```bash
   ngrok http 8085
   ```
3. Ngrok sẽ cấp đường dẫn dạng:
   `Forwarding https://abc1-23-45.ngrok-free.app -> http://localhost:8085`

---

## 📡 Hướng dẫn gọi API từ Client

Service hỗ trợ linh hoạt 3 định dạng dữ liệu gửi lên:

### 1. Gửi qua Form Data (`multipart/form-data`) - Khuyên dùng

**Bằng cURL:**
```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/detect" \
  -F "file=@/duong/dan/anh.jpg"
```

**Bằng Python:**
```python
import requests

url = "https://your-ngrok-url.ngrok-free.app/detect"
with open("photo.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

### 2. Gửi qua JSON Base64 (`application/json`)

**Bằng Python:**
```python
import base64
import requests

with open("photo.jpg", "rb") as f:
    b64_str = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://your-ngrok-url.ngrok-free.app/detect",
    json={"image": b64_str}
)
print(response.json())
```

### 3. Gửi trực tiếp nhị phân (Binary Body)

```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/detect" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@/duong/dan/anh.jpg"
```

### Tuỳ chỉnh ngưỡng nhạy (Query Parameters)
Client có thể truyền thêm ngưỡng phát hiện nếu muốn:
- `score_threshold` (mặc định: `0.6`): Ngưỡng tin cậy (0.1 -> 1.0).
- `nms_threshold` (mặc định: `0.3`): Ngưỡng lọc trùng khuôn mặt.

Ví dụ:
```bash
curl -X POST "http://localhost:8085/detect?score_threshold=0.7&nms_threshold=0.3" \
  -F "file=@test.jpg"
```

---

## 🧪 Script kiểm tra thử nghiệm

Một script Python tiện ích đã được tạo sẵn tại `test_client.py`. Script này gửi ảnh, in kết quả JSON và tự động vẽ Bounding Box cùng 5 Landmarks ra file `output_result.jpg`:

```bash
python test_client.py path/to/image.jpg https://your-ngrok-url.ngrok-free.app/detect
```

---

## ⚙️ Biến môi trường tuỳ chỉnh (Docker / Environment)

| Biến | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `PORT` | `8085` | Cổng lắng nghe của HTTP service |
| `HOST` | `0.0.0.0` | IP interface lắng nghe |
| `MODEL_PATH` | `/app/models/face_detection_yunet_2023mar.onnx` | Đường dẫn file mô hình ONNX |
| `SCORE_THRESHOLD`| `0.6` | Ngưỡng tin cậy mặc định |
| `NMS_THRESHOLD`  | `0.3` | Ngưỡng Non-Maximum Suppression |

