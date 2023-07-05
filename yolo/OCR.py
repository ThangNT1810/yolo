import cv2
import easyocr
from ultralytics import YOLO

image = cv2.imread("img.jpg")
model = YOLO("best.pt")
results = model.predict(show=True, source=image)

for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].astype(int)

        cropped_image = image[y1:y2, x1:x2]

        reader = easyocr.Reader(['en'])

        ocr_results = reader.readtext(cropped_image)

        for (bbox, text, score) in ocr_results:

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox

            x1 += x1
            y1 += y1
            x2 += x1
            y2 += y1

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print("Text:", text)
            print("Confidence:", score)
            print("----------------------")

        cv2.imshow("Number Plate", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
