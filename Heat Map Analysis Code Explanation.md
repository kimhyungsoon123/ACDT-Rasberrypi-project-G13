#Heat Map Analysis
1. Heatmap Loading and Real-World Scaling
```python
MAP_W_M = 1.2
MAP_H_M = 2.0
BOOTH_M = 0.3

img = cv2.imread("heatmap.png")
h, w, _ = img.shape

booth_w = int(BOOTH_M / MAP_W_M * w)
booth_h = int(BOOTH_M / MAP_H_M * h)

```
2. Smoking Activity Extraction and Optimal Placement Search
```python
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
```
3. Result Visualization and Output
```python
result = img.copy()
x, y = best_xy
cv2.rectangle(result, (x, y), (x + booth_w, y + booth_h), (0, 255, 0), 2)

cv2.imwrite("result_with_booth.png", result)

```
