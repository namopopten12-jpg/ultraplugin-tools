# UltraPlugin Tools — ultra-matte

ตัวตัดคนออกจากวิดีโออัตโนมัติ (ใช้โดยปลั๊กอิน UltraPlugin ของ NamoVfxs) · repo นี้มีเฉพาะตัวตัดคน ไม่มีโค้ดปลั๊กอิน

- โมเดล: [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (GPL-3.0) · รันด้วย ONNX Runtime (DirectML / CoreML / CPU)
- ffmpeg: [BtbN FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (GPL)

```
ultra-matte --in clip.mov --out matte.mp4 --size 720
```
ความคืบหน้าพิมพ์เป็น JSON ทีละบรรทัด (`{"p":0.42,...}` → `{"done":true,...}`)

ไฟล์ .zip สำหรับ Windows/Mac สร้างอัตโนมัติด้วย GitHub Actions ทุกครั้งที่แก้ matte.py → อยู่ในหน้า Releases (ล่าสุดเสมอ)
