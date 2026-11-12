"""UltraPlugin Matte — ตัดคนออกจากวิดีโออัตโนมัติ (Robust Video Matting, ONNX)
ใช้: ultra-matte --in clip.mov --out matte.mp4 [--size 720] [--fps 30] [--start 0] [--dur 0] [--ffmpeg path]
พิมพ์ความคืบหน้าเป็น JSON ทีละบรรทัด: {"p":0.42,"f":120,"n":300,"fps":4.1} · จบ: {"done":true,"out":...}
GPU: DirectML (Windows ทุกยี่ห้อ) / CoreML (Mac) → ไม่มีก็ใช้ CPU
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, onnxruntime as ort

def here(*p):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *p)

def emit(**k): print(json.dumps(k), flush=True)

def probe(ff, src):
    import shutil
    fp = os.path.join(os.path.dirname(ff), os.path.basename(ff).replace('ffmpeg', 'ffprobe')) if os.path.dirname(ff) else 'ffprobe'
    if not os.path.exists(fp): fp = shutil.which('ffprobe') or 'ffprobe'
    o = subprocess.run([fp, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate:stream_side_data=rotation:format=duration', '-of', 'json', src], capture_output=True, text=True)
    j = json.loads(o.stdout or '{}'); s = (j.get('streams') or [{}])[0]
    w, h = int(s.get('width', 0)), int(s.get('height', 0)); rot = 0
    for sd in s.get('side_data_list', []) or []:
        if 'rotation' in sd: rot = int(sd['rotation'])
    if abs(rot) % 180 == 90: w, h = h, w
    num, den = (s.get('r_frame_rate', '30/1').split('/') + ['1'])[:2]
    return w, h, float(num) / float(den or 1), float((j.get('format') or {}).get('duration', 0) or 0)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--in', dest='src', required=True); a.add_argument('--out', required=True)
    a.add_argument('--size', type=int, default=960); a.add_argument('--fps', type=float, default=0)
    a.add_argument('--start', type=float, default=0); a.add_argument('--dur', type=float, default=0)
    a.add_argument('--ffmpeg', default=here('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'))
    a.add_argument('--model', default=here('rvm_mobilenetv3_fp32.onnx'))
    x = a.parse_args(); ff = x.ffmpeg if os.path.exists(x.ffmpeg) else 'ffmpeg'
    W0, H0, fps0, dur0 = probe(ff, x.src)
    if not W0: emit(error='อ่านไฟล์วิดีโอไม่ได้'); sys.exit(2)
    sc = x.size / max(W0, H0); W = int(round(W0 * sc / 2) * 2); H = int(round(H0 * sc / 2) * 2)
    fps = x.fps or fps0; dur = x.dur or max(0.0, dur0 - x.start); N = max(1, int(dur * fps))
    prov = [p for p in ('DmlExecutionProvider', 'CoreMLExecutionProvider') if p in ort.get_available_providers()] + ['CPUExecutionProvider']
    try: sess = ort.InferenceSession(x.model, providers=prov)
    except Exception: sess = ort.InferenceSession(x.model, providers=['CPUExecutionProvider'])
    emit(info=True, w=W, h=H, fps=fps, n=N, provider=sess.get_providers()[0])
    rd = subprocess.Popen([ff, '-v', 'error', '-ss', str(x.start)] + (['-t', str(dur)] if x.dur else []) + ['-i', x.src, '-vf', f'scale={W}:{H},fps={fps}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    wr = subprocess.Popen([ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', x.out], stdin=subprocess.PIPE)
    rec = [np.zeros([1, 1, 1, 1], np.float32)] * 4; ds = np.array([min(1.0, 512 / max(W, H) * 1.0)], np.float32)
    t0 = time.time(); f = 0; fsz = W * H * 3
    while True:
        buf = rd.stdout.read(fsz)
        if len(buf) < fsz: break
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32).transpose(2, 0, 1)[None] / 255.
        _, pha, *rec = sess.run(None, {'src': img, 'r1i': rec[0], 'r2i': rec[1], 'r3i': rec[2], 'r4i': rec[3], 'downsample_ratio': ds})
        wr.stdin.write((pha[0, 0] * 255).clip(0, 255).astype(np.uint8).tobytes()); f += 1
        if f % 5 == 0: el = time.time() - t0; emit(p=round(min(1, f / N), 3), f=f, n=N, fps=round(f / max(el, 1e-3), 2))
    wr.stdin.close(); wr.wait(); rd.wait()
    emit(done=True, out=x.out, frames=f, sec=round(time.time() - t0, 1), w=W, h=H, fps=fps)

if __name__ == '__main__': main()
