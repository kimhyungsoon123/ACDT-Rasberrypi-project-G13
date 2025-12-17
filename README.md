# ACDT-Rasberrypi-project-G13
Our ACDT work log


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


