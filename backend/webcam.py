import cv2
from ultralytics import YOLO

# Load your RoadSense model
model = YOLO("yolo12s_RDD2022_best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, conf=0.35)

    potholes = 0
    cracks = 0

    annotated = frame.copy()

    for r in results:
        annotated = r.plot()

        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls].lower()

            if "pothole" in name or "d40" in name:
                potholes += 1

            elif (
                "crack" in name
                or "d00" in name
                or "d10" in name
                or "d20" in name
            ):
                cracks += 1

    # Road Health Score
    score = max(0, 100 - potholes * 20 - cracks * 5)

    if score >= 80:
        color = (0, 255, 0)
    elif score >= 50:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(
        annotated,
        f"Potholes: {potholes}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        annotated,
        f"Cracks: {cracks}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"Road Health: {score}%",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.imshow("RoadSense AI", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()