# ACDT-Rasberrypi-project-G13
1. Camera Frame Acquisition and Sharing
```
def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        with frame_lock:
            latest_frame = frame
        frame_event.set()
```
2. YOLO-Based Cigarette Butt Detection
```
def yolo_thread(model):
    global latest_dets_all
    last_run = 0.0

    while running:
        frame_event.wait(timeout=0.2)

        with state_lock:
            s = state
        if s == STATE_PICKING:
            time.sleep(0.02)
            continue

        now = time.monotonic()
        if now - last_run < YOLO_INTERVAL_S:
            time.sleep(0.005)
            continue
        last_run = now

        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            continue

        results = model(frame[..., ::-1], size=YOLO_IMGSZ)
        det = results.xyxy[0]

        with det_lock:
            if det is not None and len(det) > 0:
                det_sorted = det[det[:, 4].argsort(descending=True)]
                latest_dets_all = det_sorted[:, :6].tolist()
            else:
                latest_dets_all = None
```



