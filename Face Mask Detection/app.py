import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("mask_detector_model.h5")

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)



labels = ['Mask', 'No Mask']

while True:
    ret, frame = cap.read()
    if not ret:
        break


    img = cv2.resize(frame, (128, 128))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 


    prediction = model.predict(img)
    prob = prediction[0][0]

    if prob < 0.5:
        label = labels[0]   # Mask
        color = (0, 255, 0)
    else:
        label = labels[1]   # No Mask
        color = (0, 0, 255)


    cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Mask Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
