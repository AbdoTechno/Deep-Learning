````markdown
````
# 😷 Mask Detection using Deep Learning (With & Without Mask)

## 📌 Overview
This project aims to detect whether a person is wearing a **face mask** or not using **Deep Learning** and **Computer Vision** techniques.  
It supports **real-time detection** using either a **webcam** or a **mobile camera (via IP stream)**.

---

## 🧠 Models Used
### 1. Custom CNN Model  
A convolutional neural network built from scratch using:
- `Conv2D`
- `MaxPooling2D`
- `Dropout`
- `Dense` layers  
This helps in understanding the inner workings of CNN architectures.

### 2. Pretrained Model (Transfer Learning)  
Used **MobileNetV2**, a lightweight and efficient model pretrained on **ImageNet**, fine-tuned for binary mask classification.  
This approach achieves higher accuracy in less training time.

---

## 🧾 Dataset  
- Dataset: Face Mask Detection Dataset (by Omkargurav) — contains ~7,553 images of faces with masks and without masks. :contentReference[oaicite:3]{index=3}  
- Link: [Kaggle Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- Classes: `With Mask`, `Without Mask`  
- Images resized to `(224, 224)` for MobileNetV2, and `(128, 128)` for custom-CNN (depending on your setup).  
- Data augmentation applied using `ImageDataGenerator` for better generalization.

---

## 🧰 Technologies Used  
- **Python 3.x**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **NumPy**  
- (Optional) **Matplotlib / Seaborn** for result plotting  
- MobileNetV2 pretrained weights (ImageNet)

---

## ⚙️ How to Run

### 1. Clone the repository  
```bash
git clone https://github.com/yourusername/mask-detection.git
cd mask-detection
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

You can train either the custom CNN or the MobileNetV2 version:

```bash
python train_cnn.py        # For custom CNN
python train_mobilenet.py  # For MobileNetV2
```

### 4. Real-time detection

```bash
python detect_mask.py
```

> To use your **mobile camera**, replace the webcam source:

```python
cap = cv2.VideoCapture("http://<your_ip_address>:8080/video")
```

---

## 📈 Results

| Model       | Accuracy (Example) | Type               |
| ----------- | ------------------ | ------------------ |
| Custom CNN  | ~90% (approx.)     | Built from scratch |
| MobileNetV2 | ~97% (approx.)     | Transfer Learning  |

> *Note: These values are indicative—actual results depend on your training, data splits, and hyper-parameters.*

---

## 🖼️ Example

Real-time detection with bounding boxes and labels:

* ✅ Mask (Green)
* ❌ No Mask (Red)

---

## 💡 Future Improvements

* Integrate **face detection** (e.g., Haar Cascades, MTCNN) before mask classification for more accuracy.
* Extend to multi-face detection in live video streams.
* Deploy as a **web application** or a **mobile app**.
* Expand to include class for **incorrectly worn mask** (i.e., mask not covering nose/ chin).

---

## 👤 Author

**Your Name**
📧 [abdoalsenawework@gmail.com]
🔗 [[Your LinkedIn](https://www.linkedin.com/in/abdotech/)]

---


