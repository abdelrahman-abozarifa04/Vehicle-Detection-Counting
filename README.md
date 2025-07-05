# 🚗 Vehicle Detection & Counting
---

A computer vision project designed to detect and count vehicles on a highway using object detection and tracking techniques. The system processes video frames to identify vehicles, tracks them across frames, and counts the total number of vehicles crossing a specified line. This project aims to support traffic analysis, congestion management, and intelligent transportation systems.

---

## 📋 Dataset / Input Overview

This project uses video footage of a highway as input for real-time vehicle detection and counting.

* **Video Source:** Sample highway surveillance footage.
* **Classes Detected:** Cars, trucks, buses, and motorcycles.

---

## 🚀 Features

* **Real-Time Detection & Tracking:** Detects vehicles in each frame and assigns unique IDs.
* **Counting Line:** Counts vehicles as they cross a virtual line.
* **Annotated Output:** Draws bounding boxes, IDs, and displays the total vehicle count.
* **Recording:** Saves processed video with annotations.

---

## 🛠️ Tools and Libraries

* **Programming Language:** Python
* **Computer Vision:** OpenCV
* **Deep Learning:** YOLO / other pre-trained models (can be updated)
* **Tracking:** Built-in tracking algorithms (e.g., SORT or custom tracking logic)
* **Development Environment:** Jupyter Notebook, Visual Studio Code

---

## 📈 Methodology

1. **Detection:** Uses an object detection model to detect vehicles in each frame.
2. **Tracking:** Assigns unique IDs to detected vehicles and tracks them across frames.
3. **Counting:** Checks if a tracked vehicle crosses the counting line and increments the count.
4. **Visualization:** Draws bounding boxes, IDs, the counting line, and displays the total count.

---

## 🎯 Usage

### Clone the repository:

```bash
git clone https://github.com/abdelrahman-abozarifa04/Vehicle-Detection-Counting.git
cd Vehicle-Detection-Counting
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the notebook:

```bash
jupyter notebook car_detection_project.ipynb
```

### Adjust settings:

* Set `line_position` in the notebook to adjust where vehicles are counted.
* Change video source path for different input videos.

---

## 📊 Results

* **Sample Performance:** Detects and tracks vehicles with high accuracy on highway video.
* **Output:** Displays processed video with bounding boxes, unique IDs, and total vehicle count.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📧 Contact

**Abdelrahman Abozarifa**

* GitHub: [@abdelrahman-abozarifa04](https://github.com/abdelrahman-abozarifa04)
* Email: [as0144549@gmail.com](mailto:as0144549@gmail.com)

---

## ⭐ Acknowledgements

Special thanks to the open-source community for providing powerful computer vision libraries and pre-trained models.

---

✅ Let me know if you’d like me to:

* Generate a `requirements.txt` based on your notebook.
* Create a `.gitignore` file for you.
* Or help write a `main.py` script for easier CLI execution.

Just tell me! 🚀
