# RCcar Mechanism
1. Camera Frame Acquisition and Sharing
```python

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
```python

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
3. Selection of the Closest Target
```python
def select_frontmost_target(dets_all):
    if dets_all is None or len(dets_all) == 0:
        return None

    dets = np.array(dets_all, dtype=np.float32)
    ys_center = (dets[:, 1] + dets[:, 3]) * 0.5
    idx = np.argsort(ys_center)[::-1]
    dets = dets[idx]

    return dets[0].tolist()
```
4. Motion Decision: Search, Align, or Stop
```python
with det_lock:
    dets_all = latest_dets_all

target = select_frontmost_target(dets_all)

if target is None:
    steer.angle = STEER_CENTER_ANGLE
    drive_forward(SEARCH_SPEED)
else:
    x1, y1, x2, y2, conf, cls = target
    h, w = frame.shape[:2]
    cy_n = ((y1 + y2) * 0.5) / h
    cx_n = ((x1 + x2) * 0.5) / w

    if cy_n < STOP_CY:
        if cx_n < CENTER_LEFT:
            steer.angle = 40
        elif cx_n > CENTER_RIGHT:
            steer.angle = 140
        else:
            steer.angle = STEER_CENTER_ANGLE
        drive_forward(APPROACH_SPEED)
    else:
        stop()
        steer.angle = STEER_CENTER_ANGLE
        threading.Thread(
            target=pickup_sequence,
            args=(target, frame.shape),
            daemon=True
        ).start()
```
5. Pickup Initialization and Heatmap Logging
```python
def pickup_sequence(target_det, frame_shape):
    global state, last_pick_time

    with state_lock:
        state = STATE_PICKING

    stop()
    steer.angle = STEER_CENTER_ANGLE

    h, w = frame_shape[:2]
    x1, y1, x2, y2, conf, cls = target_det
    cx_n = ((x1 + x2) * 0.5) / w
    cy_n = ((y1 + y2) * 0.5) / h

    HEAT_SCALE = 4.0
    forward_m = max(0.0, (1.0 - cy_n)) * (1.30 * HEAT_SCALE)
    lateral_m = (cx_n - 0.5) * (1.80 * HEAT_SCALE)

    wx = car_x + forward_m * np.cos(heading) - lateral_m * np.sin(heading)
    wy = car_y + forward_m * np.sin(heading) + lateral_m * np.cos(heading)

    add_heat_point_world(wx, wy)

```
6. Robotic Arm and Bucket Pickup Motion
```python
move_servo(arm_pitch, ARM_PITCH_PICK, 0.03)
move_servo(arm_yaw, ARM_YAW_PICK, 0.03)
time.sleep(0.25)

move_servo(arm_pitch, ARM_PITCH_SWIPE, 0.03)
move_servo(arm_yaw, ARM_YAW_SWIPE, 0.02)
time.sleep(0.35)

move_servo(arm_yaw, ARM_YAW_CENTER, 0.02)
move_servo(arm_pitch, ARM_PITCH_CENTER, 0.03)

extra_servo_sweep()

```



