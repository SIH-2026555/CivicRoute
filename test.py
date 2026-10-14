


from inference_sdk import InferenceHTTPClient
import cv2

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="MWqj6vMjtATmcMjpX1uh"
)

image_path = "road1.jpg"
result = CLIENT.infer(image_path, model_id="rdd2022-zw12e/2")

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Could not find road1.jpg")

pothole_number = 0

print("\n===== ROAD DAMAGE REPORT =====")

for p in result.get("predictions", []):

    # D40 is the pothole class in RDD2022
    if p.get("class", "").upper() != "D40":
        continue

    pothole_number += 1

    x = p["x"]
    y = p["y"]
    w = p["width"]
    h = p["height"]
    confidence = p["confidence"]

    # Bounding-box area in pixels squared
    area = w * h

    # Example thresholds only; calibrate for your dataset
    if area < 3000:
        severity = "LOW"
        color = (0, 255, 0)
    elif area < 10000:
        severity = "MODERATE"
        color = (0, 165, 255)
    else:
        severity = "HIGH"
        color = (0, 0, 255)

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # Mark the pothole
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    # Show number and confidence on the image
    text = f"Pothole {pothole_number}: {confidence:.1%}"
    cv2.putText(
        image, text, (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
    )

    print(f"\nPothole {pothole_number}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Bounding-box area: {area:.0f} pixels squared")
    print(f"Estimated severity: {severity}")

if pothole_number == 0:
    print("No potholes detected.")

cv2.imwrite("road_damage_output.jpg", image)

print(f"\nTotal potholes detected: {pothole_number}")
print("Saved image: road_damage_output.jpg")

cv2.imshow("Pothole Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
