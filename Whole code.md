# Whole code
1. AI Learning Whole Code
```python

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# 2. Clone YOLOv5 and install requirements
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt


# 3. Create YOLO data configuration file
%%writefile /content/drive/MyDrive/data.yaml
train: /content/drive/MyDrive/cigarette_dataset/images/train
val: /content/drive/MyDrive/cigarette_dataset/images/val
test: /content/drive/MyDrive/cigarette_dataset/images/test

nc: 1
names: ['cigarette_butt']


# 4. Verify dataset and config file
!ls /content/drive/MyDrive/cigarette_dataset/images/train
!cat /content/drive/MyDrive/data.yaml


# 5. Train YOLOv5 model
%cd /content/yolov5

!python train.py \
    --img 640 \
    --batch 16 \
    --epochs 50 \
    --data /content/drive/MyDrive/data.yaml \
    --weights yolov5s.pt \
    --name cigarette_model


# 6. Save trained model to Google Drive
!cp /content/yolov5/runs/train/cigarette_model/weights/best.pt \
    /content/drive/MyDrive/cigarette_best.pt


# 7. (Optional) Download trained model to local machine
from google.colab import files
files.download('/content/drive/MyDrive/cigarette_best.pt')

```
2. RCcar Driving Mechanism Whole Code
```python
import time
import cv2
import torch
import threading
import numpy as np
from threading import Event, Lock

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo
from gpiozero import DistanceSensor

BASE_DIR = "/home/sex/yolov5"
YOLO_REPO = BASE_DIR
WEIGHT_PATH = f"{BASE_DIR}/cigarette_best.pt"

MOTOR_LEFT_IN1 = 15
MOTOR_LEFT_IN2 = 14
MOTOR_RIGHT_IN1 = 12
MOTOR_RIGHT_IN2 = 13
STEER_CHANNEL = 0

STEER_CENTER_ANGLE = 90
STEER_LEFT_ANGLE = 0
STEER_RIGHT_ANGLE = 180

ARM_YAW_CHANNEL = 2
ARM_PITCH_CHANNEL = 3
EXTRA_SERVO_CHANNEL = 10

ARM_YAW_CENTER = 0
ARM_YAW_PICK = 40
ARM_YAW_SWIPE = 35

ARM_PITCH_CENTER = 180
ARM_PITCH_PICK = 110
ARM_PITCH_SWIPE = 15

TRIG_PIN = 23
ECHO_PIN = 24
MAX_DISTANCE_M = 2

CENTER_LEFT = 0.45
CENTER_RIGHT = 0.55

SEARCH_SPEED = 8.6
APPROACH_SPEED = 7.5
BACK_SPEED = 9

STOP_CY = 0.85
TOO_CLOSE_CY = 0.92
OBSTACLE_DISTANCE_CM = 20.0

CAM_W = 640
CAM_H = 480
CAM_FPS = 13

YOLO_IMGSZ = 640
YOLO_INTERVAL_S = 0.12
ULTRA_INTERVAL_S = 0.08
UI_INTERVAL_S = 0.03

MAP_SIZE = 700
MAP_CENTER = MAP_SIZE // 2
MAP_PX_PER_M = 120.0

SPEED_TO_MPS = 0.22
TURN_RATE = 1.20

STATE_MOVING = 0
STATE_PICKING = 1

latest_frame = None
latest_dets_all = None

running = True
state = STATE_MOVING

frame_event = Event()
frame_lock = Lock()
det_lock = Lock()
state_lock = Lock()

heatmap = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

car_x = 0.0
car_y = 0.0
heading = 0.0
cmd_speed = 0.0
cmd_steer = STEER_CENTER_ANGLE

last_pick_time = 0.0
PICK_COOLDOWN_S = 1.2

def map_value(x, a, b, c, d):
    return (x - a) / (b - a) * (d - c) + c

i2c = busio.I2C(SCL, SDA)
pwm = PCA9685(i2c, address=0x5f)
pwm.frequency = 50

motor_left = motor.DCMotor(pwm.channels[MOTOR_LEFT_IN1], pwm.channels[MOTOR_LEFT_IN2])
motor_right = motor.DCMotor(pwm.channels[MOTOR_RIGHT_IN1], pwm.channels[MOTOR_RIGHT_IN2])

steer = servo.Servo(pwm.channels[STEER_CHANNEL])
arm_yaw = servo.Servo(pwm.channels[ARM_YAW_CHANNEL])
arm_pitch = servo.Servo(pwm.channels[ARM_PITCH_CHANNEL])
extra_servo = servo.Servo(pwm.channels[EXTRA_SERVO_CHANNEL])

sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=MAX_DISTANCE_M)

def set_motor(m, d, s):
    s = max(0.0, min(100.0, float(s)))
    if d == 0 or s <= 0.0:
        m.throttle = 0.0
        return
    v = map_value(s, 0.0, 100.0, 0.0, 1.0)
    m.throttle = v if d > 0 else -v

def apply_drive(speed_percent, steer_angle):
    global cmd_speed, cmd_steer
    cmd_speed = float(speed_percent)
    cmd_steer = float(steer_angle)

def drive_forward(s):
    set_motor(motor_left, 1, s)
    set_motor(motor_right, 1, s)
    a = steer.angle if steer.angle is not None else STEER_CENTER_ANGLE
    apply_drive(s, a)

def drive_backward(s):
    set_motor(motor_left, -1, s)
    set_motor(motor_right, -1, s)
    a = steer.angle if steer.angle is not None else STEER_CENTER_ANGLE
    apply_drive(-s, a)

def stop():
    global cmd_speed
    motor_left.throttle = 0.0
    motor_right.throttle = 0.0
    cmd_speed = 0.0

def move_servo(servo_obj, target, speed):
    now = servo_obj.angle
    if now is None:
        servo_obj.angle = target
        return
    step = 1 if target > now else -1
    for a in range(int(now), int(target), step):
        servo_obj.angle = a
        time.sleep(speed)
    servo_obj.angle = target

def extra_servo_sweep():
    move_servo(extra_servo, 180, speed=0.001)
    time.sleep(0.2)
    move_servo(extra_servo, 40, speed=0.02)
    time.sleep(0.2)

def integrate_odometry(dt):
    global car_x, car_y, heading
    with state_lock:
        s = state
    if s != STATE_MOVING:
        return
    v_mps = (cmd_speed / 100.0) * SPEED_TO_MPS
    if abs(v_mps) < 1e-6:
        return
    if cmd_steer == STEER_LEFT_ANGLE:
        heading += TURN_RATE * dt
    elif cmd_steer == STEER_RIGHT_ANGLE:
        heading -= TURN_RATE * dt
    car_x += v_mps * np.cos(heading) * dt
    car_y += v_mps * np.sin(heading) * dt

def world_to_map(x, y):
    px = int(MAP_CENTER + x * MAP_PX_PER_M)
    py = int(MAP_CENTER - y * MAP_PX_PER_M)
    px = int(np.clip(px, 0, MAP_SIZE - 1))
    py = int(np.clip(py, 0, MAP_SIZE - 1))
    return px, py

def add_heat_point_world(wx, wy):
    px, py = world_to_map(wx, wy)
    cv2.circle(heatmap, (px, py), 6, (0, 0, 255), -1)
    label = f"({wx:+.2f},{wy:+.2f})"
    tx = min(px + 8, MAP_SIZE - 180)
    ty = max(py - 8, 18)
    cv2.putText(heatmap, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def select_frontmost_target(dets_all):
    if dets_all is None or len(dets_all) == 0:
        return None
    dets = np.array(dets_all, dtype=np.float32)

    ys_center = (dets[:, 1] + dets[:, 3]) * 0.5
    idx = np.argsort(ys_center)[::-1]
    dets = dets[idx]

    return dets[0].tolist()

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

    extra_servo.angle=0

    move_servo(arm_pitch, ARM_PITCH_PICK, 0.03)
    move_servo(arm_yaw, ARM_YAW_PICK, 0.03)
    time.sleep(0.25)

    move_servo(arm_pitch, ARM_PITCH_SWIPE, 0.03)
    move_servo(arm_yaw, ARM_YAW_SWIPE, 0.02)
    time.sleep(0.35)

    move_servo(arm_yaw, ARM_YAW_CENTER, 0.02)
    move_servo(arm_pitch, ARM_PITCH_CENTER, 0.03)

    extra_servo_sweep()

    time.sleep(1)

    last_pick_time = time.monotonic()

    with state_lock:
        state = STATE_MOVING
        
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
    cap.release()

def yolo_thread(model):
    global latest_dets_all
    last_run = 0.0
    while running:
        frame_event.wait(timeout=0.2)
        if not running:
            break

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
            time.sleep(0.01)
            continue

        results = model(frame[..., ::-1], size=YOLO_IMGSZ)
        det = results.xyxy[0]

        with det_lock:
            if det is not None and len(det) > 0:
                det_sorted = det[det[:, 4].argsort(descending=True)]
                latest_dets_all = det_sorted[:, :6].tolist()
            else:
                latest_dets_all = None

def start():
    global running

    torch.set_num_threads(2)
    model = torch.hub.load(YOLO_REPO, "custom", path=WEIGHT_PATH, source="local")
    model.conf = 0.25

    steer.angle = STEER_CENTER_ANGLE
    arm_yaw.angle = ARM_YAW_CENTER
    arm_pitch.angle = ARM_PITCH_CENTER
    extra_servo.angle = 40

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=yolo_thread, args=(model,), daemon=True).start()

    last_ultra = 0.0
    cached_dist = 999.0

    last_t = time.monotonic()

    while running:
        now_t = time.monotonic()
        dt = now_t - last_t
        last_t = now_t

        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.02)
            continue

        integrate_odometry(dt)

        with state_lock:
            s = state

        if s == STATE_PICKING:
            cv2.imshow("YOLO Cigarette", frame)
            cv2.imshow("Heatmap", heatmap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            time.sleep(UI_INTERVAL_S)
            continue

        if now_t - last_ultra >= ULTRA_INTERVAL_S:
            last_ultra = now_t
            try:
                cached_dist = sensor.distance * 100.0
            except Exception:
                cached_dist = 999.0

        if cached_dist < OBSTACLE_DISTANCE_CM:
            stop()
            steer.angle = STEER_CENTER_ANGLE
            time.sleep(0.3)
            drive_backward(BACK_SPEED)
            time.sleep(1.0)
            stop()
            steer.angle = STEER_RIGHT_ANGLE
            time.sleep(0.4)
            drive_forward(9)
            time.sleep(2.0)
            stop()
            steer.angle = STEER_CENTER_ANGLE

            cv2.imshow("YOLO Cigarette", frame)
            cv2.imshow("Heatmap", heatmap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            time.sleep(UI_INTERVAL_S)
            continue

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
                if last_pick_time == 0.0 or (now_t - last_pick_time) >= PICK_COOLDOWN_S:
                    threading.Thread(target=pickup_sequence, args=(target, frame.shape), daemon=True).start()

        cv2.imshow("YOLO Cigarette", frame)
        cv2.imshow("Heatmap", heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        time.sleep(UI_INTERVAL_S)

    stop()
    pwm.deinit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()
```
3. Heat Map Analysis Whole Code
```python
import cv2
import numpy as np


MAP_W_M = 1.2
MAP_H_M = 2.0
BOOTH_M = 0.3


img = cv2.imread("heatmap.png")
h, w, _ = img.shape


booth_w = int(BOOTH_M / MAP_W_M * w)
booth_h = int(BOOTH_M / MAP_H_M * h)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])


mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)


best_count = 0
best_score = float("inf")
best_xy = (0, 0)


step = 5


for x in range(0, w - booth_w, step):
    for y in range(0, h - booth_h, step):
        roi = mask[y:y + booth_h, x:x + booth_w]
        count = cv2.countNonZero(roi)


        if count == 0:
            continue


        ys, xs = np.where(roi > 0)
        xs = xs + x
        ys = ys + y


        cx = x + booth_w / 2
        cy = y + booth_h / 2


        mx = np.mean(xs)
        my = np.mean(ys)


        score = (cx - mx) ** 2 + (cy - my) ** 2


        if count > best_count or (count == best_count and score < best_score):
            best_count = count
            best_score = score
            best_xy = (x, y)


result = img.copy()
x, y = best_xy
cv2.rectangle(result, (x, y), (x + booth_w, y + booth_h), (0, 255, 0), 2)


cv2.imwrite("result_with_booth.png", result)
```
