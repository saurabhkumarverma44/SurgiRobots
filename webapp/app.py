# import io
# import os
# import json
# from pathlib import Path
# from datetime import datetime

# import cv2
# import numpy as np
# from fastapi import FastAPI, UploadFile, File, Request, Response, Body
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from PIL import Image
# import torch
# import serial
# import time

# BASE_DIR = Path(__file__).resolve().parent
# WEIGHTS_PATH = BASE_DIR.parent / "results" / "surgical_memsafe" / "weights" / "best.pt"
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
# IMG_SIZE = 640

# # --- Serial config: update COM port to your Arduino ---
# SERIAL_PORT = os.getenv("ARM_PORT", "COM3")   # e.g., COM3/COM5 on Windows; check Device Manager
# SERIAL_BAUD = int(os.getenv("ARM_BAUD", "115200"))

# ser = None
# def serial_connect():
#     global ser
#     try:
#         ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
#         time.sleep(2.0)  # Arduino reset wait
#         print(f"[SERIAL] Connected {SERIAL_PORT} @ {SERIAL_BAUD}")
#     except Exception as e:
#         print(f"[SERIAL] Connect failed: {e}")
#         ser = None

# def serial_send(cmd: str):
#     if ser is None or not ser.is_open:
#         serial_connect()
#     if ser:
#         ser.write((cmd + "\n").encode("utf-8"))
#         # Optional: read a line
#         try:
#             resp = ser.readline().decode("utf-8", errors="ignore").strip()
#             if resp:
#                 print("[SERIAL<-]", resp)
#         except:
#             pass
#         print("[SERIAL->]", cmd)

# # --- Homography (image pixels -> tray XY meters)
# # Fill with your calibration values (3x3). This maps [u,v,1]^T to [X,Y,w]^T.
# H = np.array([
#     [1.23e-3, 0.00e+0, -0.200],   # PLACEHOLDER — replace with your numbers
#     [0.00e+0, 1.23e-3, -0.150],
#     [0.00e+0, 0.00e+0, 1.000]
# ], dtype=float)

# def pixels_to_xy(u, v):
#     p = np.array([u, v, 1.0])
#     q = H @ p
#     X, Y = q[0]/q[2], q[1]/q[2]
#     return float(X), float(Y)

# # Heights: conceptual for 3‑servo hobby arm (use two levels)
# Z_SAFE = float(os.getenv("Z_SAFE", "0.250"))
# Z_PICK = float(os.getenv("Z_PICK", "0.120"))

# app = FastAPI(title="SurgiVision - Instrument Detection + Pick")
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# # Load YOLOv5
# model = torch.hub.load("ultralytics/yolov5", "custom", path=str(WEIGHTS_PATH), device=DEVICE)
# model.conf = float(os.getenv("CONF_THRES", "0.25"))
# model.iou = float(os.getenv("IOU_THRES", "0.45"))
# model.max_det = int(os.getenv("MAX_DET", "300"))

# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# def draw_boxes(image: np.ndarray, detections) -> np.ndarray:
#     if detections is None or len(detections.xyxy) == 0:
#         return image
#     det = detections.xyxy[0].cpu().numpy() if torch.is_tensor(detections.xyxy[0]) else detections.xyxy[0]
#     names = model.names
#     for x1, y1, x2, y2, conf, cls_id in det:
#         cls_id = int(cls_id)
#         label = f"{names[cls_id]} {conf:.2f}"
#         color = (0, 200, 255)
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         cv2.rectangle(image, (int(x1), int(y1)-th-8), (int(x1)+tw+6, int(y1)), color, -1)
#         cv2.putText(image, label, (int(x1)+3, int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
#     return image

# @app.post("/infer_frame")
# async def infer_frame(file: UploadFile = File(...)):
#     data = await file.read()
#     img = Image.open(io.BytesIO(data)).convert("RGB")
#     frame = np.array(img)
#     results = model(frame, size=IMG_SIZE)
#     out = draw_boxes(frame.copy(), results)

#     det_list = []
#     if results is not None and len(results.xyxy):
#         arr = results.xyxy[0].cpu().numpy() if torch.is_tensor(results.xyxy[0]) else results.xyxy[0]
#         names = model.names
#         for x1, y1, x2, y2, conf, cls_id in arr:
#             det_list.append({
#                 "x1": float(x1), "y1": float(y1),
#                 "x2": float(x2), "y2": float(y2),
#                 "conf": float(conf),
#                 "cls": int(cls_id),
#                 "name": names[int(cls_id)]
#             })

#     _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
#     return Response(
#         content=jpeg.tobytes(),
#         headers={"X-Detections": json.dumps(det_list)},
#         media_type="image/jpeg"
#     )

# # Optional server-side logging
# LOG_PATH = BASE_DIR / "events.csv"
# @app.post("/log_event")
# async def log_event(payload: dict = Body(...)):
#     line = f"{datetime.utcnow().isoformat()},{payload.get('class')},{payload.get('conf')},{payload.get('box')}\n"
#     with open(LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(line)
#     return {"ok": True}

# # --- Arduino pick endpoint (3‑DOF planar) ---
# @app.post("/grasp3dof")
# async def grasp3dof(payload: dict = Body(...)):
#     """
#     Payload: { "class": "...", "conf": 0.92, "center": [u,v] }
#     """
#     cls = payload.get("class", "")
#     conf = float(payload.get("conf", 0))
#     center = payload.get("center", [0,0])
#     u, v = float(center[0]), float(center[1])

#     if conf < 0.35:
#         return {"ok": False, "reason": "low_conf"}

#     X, Y = pixels_to_xy(u, v)

#     # Bounds (meters) — set to your tray region
#     # if not (0.05 < X < 0.30 and 0.05 < Y < 0.25):
#     #     return {"ok": False, "reason": "out_of_bounds"}

#     # Sequence: HOME -> move safe -> move pick -> grip -> move safe
#     serial_send("HOME")
#     time.sleep(0.3)
#     serial_send(fmt_goto(X, Y, Z_SAFE))
#     time.sleep(0.3)
#     serial_send(fmt_goto(X, Y, Z_PICK))
#     time.sleep(0.2)
#     serial_send("GRIP CLOSE")
#     time.sleep(0.3)
#     serial_send(fmt_goto(X, Y, Z_SAFE))
#     time.sleep(0.3)

#     return {"ok": True, "picked": cls, "X": X, "Y": Y}

# def fmt_goto(x, y, z):
#     return f"GOTO X={x:.3f} Y={y:.3f} Z={z:.3f}"
import io, os, json, time
from pathlib import Path
import cv2, numpy as np, torch, serial
from fastapi import FastAPI, UploadFile, File, Request, Response, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR.parent / "results" / "surgical_memsafe" / "weights" / "best.pt"

# CHANGE THIS TO YOUR ARDUINO COM PORT
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

print("Loading model...")
model = torch.hub.load("ultralytics/yolov5", "custom", path=str(WEIGHTS_PATH))
model.conf = 0.25
print("Model loaded")

ser = None

def connect_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2)
        time.sleep(2)
        print(f"Connected to {SERIAL_PORT}")
        return True
    except Exception as e:
        print(f"Serial connection failed: {e}")
        return False

def send_command(cmd):
    global ser
    if ser is None and not connect_serial():
        return False
    try:
        ser.write(f"{cmd}\n".encode())
        response = ser.readline().decode().strip()
        print(f"SENT: {cmd}")
        if response:
            print(f"RECV: {response}")
        return True
    except Exception as e:
        print(f"Command failed: {e}")
        ser = None
        return False

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/infer_frame")
async def infer_frame(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    frame = np.array(image)
    
    results = model(frame, size=640)
    
    # Draw detections
    detections = []
    if len(results.xyxy[0]) > 0:
        for x1, y1, x2, y2, conf, cls_id in results.xyxy[0].cpu().numpy():
            cls_id = int(cls_id)
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            detections.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "conf": float(conf), "cls": cls_id, "name": model.names[cls_id]
            })
    
    # Encode image
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Detections": json.dumps(detections)}
    )

@app.post("/grasp3dof")
def grasp_object(payload: dict = Body(...)):
    try:
        conf = float(payload.get("conf", 0))
        if conf < 0.25:
            return {"ok": False, "reason": "Low confidence"}
        
        center = payload.get("center", [320, 240])
        u, v = float(center[0]), float(center[1])
        
        # Convert pixel coordinates to robot workspace
        # Adjust these ranges based on your camera setup
        x = 0.05 + (u / 640.0) * 0.25  # X range: 0.05 to 0.30 meters
        y = -0.15 + (v / 480.0) * 0.30  # Y range: -0.15 to 0.15 meters
        z_safe = 0.25
        z_pick = 0.12
        
        print(f"Picking object at pixel ({u:.0f}, {v:.0f}) -> world ({x:.3f}, {y:.3f})")
        
        # Execute pick sequence
        commands = [
            "HOME",
            f"GOTO X={x:.3f} Y={y:.3f} Z={z_safe:.3f}",
            f"GOTO X={x:.3f} Y={y:.3f} Z={z_pick:.3f}",
            "GRIP CLOSE",
            f"GOTO X={x:.3f} Y={y:.3f} Z={z_safe:.3f}"
        ]
        
        for cmd in commands:
            if not send_command(cmd):
                return {"ok": False, "reason": "Robot communication error"}
            time.sleep(1)
        
        return {"ok": True, "x": x, "y": y}
        
    except Exception as e:
        print(f"Grasp error: {e}")
        return {"ok": False, "reason": str(e)}

@app.post("/robot_home")
def robot_home():
    if send_command("HOME"):
        return {"ok": True}
    return {"ok": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
