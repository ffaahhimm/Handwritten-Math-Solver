import cv2
import numpy as np
from tensorflow import keras
from solver import solve_expression

print("Loading model...")
model = keras.models.load_model('model/math_model.keras')
CLASSES = np.load('model/classes.npy')
print(f"Model loaded! Classes: {CLASSES}")

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None

def preprocess(img):
    img = cv2.resize(img, (28, 28))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)
    return img

def predict(img):
    pred = model.predict(preprocess(img), verbose=0)
    return CLASSES[np.argmax(pred)]

def draw(event, x, y, flags, param):
    global canvas, prev_x, prev_y
    if event == cv2.EVENT_LBUTTONDOWN:
        prev_x, prev_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255,255,255), 15)
        prev_x, prev_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        prev_x, prev_y = None, None

def find_symbols(c):
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # Dilate to connect broken strokes within same symbol
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    symbols = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            # Add padding around ROI
            pad = 8
            x1 = max(0, x-pad)
            y1 = max(0, y-pad)
            x2 = min(c.shape[1], x+w+pad)
            y2 = min(c.shape[0], y+h+pad)
            roi = c[y1:y2, x1:x2]
            symbol = predict(roi)
            symbols.append(symbol)
            cv2.rectangle(c, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(c, symbol, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return ''.join(symbols)

print("Canvas ready! Draw with mouse.")
print("S=Solve | C=Clear | Q=Quit")

cv2.namedWindow('Math Solver')
cv2.setMouseCallback('Math Solver', draw)

result = ""
expression = ""

while True:
    display = canvas.copy()
    cv2.putText(display, f"Expr: {expression}", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(display, f"Result: {result}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(display, "S=Solve  C=Clear  Q=Quit", (10,460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.imshow('Math Solver', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        expression = find_symbols(canvas)
        result = solve_expression(expression)
        print(f"{expression} = {result}")
    elif key == ord('c'):
        canvas = np.zeros((480,640,3), dtype=np.uint8)
        expression = ""
        result = ""
    elif key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
print("Done!")
