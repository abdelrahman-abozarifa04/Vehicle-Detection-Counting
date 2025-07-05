import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚗 Car Detection and Tracking Project\n",
    "\n",
    "This notebook implements a real-time car detection and tracking system using:\n",
    "- YOLOv8 for object detection\n",
    "- DeepSORT for object tracking\n",
    "- OpenCV for video processing\n",
    "\n",
    "The system can:\n",
    "1. Detect vehicles in video frames\n",
    "2. Track vehicles across frames\n",
    "3. Count vehicles crossing a specified line\n",
    "4. Generate a processed video with tracking information"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📚 Import Required Libraries\n",
    "\n",
    "First, we import all necessary libraries for the project:\n",
    "- OpenCV for video processing\n",
    "- YOLO for object detection\n",
    "- DeepSORT for object tracking\n",
    "- NumPy for numerical operations\n",
    "- Tkinter for file dialog"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import cv2\n",
    "from ultralytics import YOLO\n",
    "from deep_sort_realtime.deepsort_tracker import DeepSort\n",
    "import numpy as np\n",
    "import tkinter as tk\n",
    "from tkinter import filedialog"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Video Selection Function\n",
    "\n",
    "This function opens a file dialog to select a video file for processing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "def select_video():\n",
    "    \"\"\"Open file dialog to select video file.\"\"\"\n",
    "    root = tk.Tk()\n",
    "    root.withdraw()\n",
    "    video_path = filedialog.askopenfilename(\n",
    "        title=\"Select Video File\",\n",
    "        filetypes=[(\"Video files\", \"*.mp4 *.avi *.mov *.mkv\"), (\"All files\", \"*.*\")]\n",
    "    )\n",
    "    if not video_path:\n",
    "        raise ValueError(\"No video file selected\")\n",
    "    return video_path"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📹 Video Writer Setup Function\n",
    "\n",
    "This function sets up the video writer with the same properties as the input video."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "def setup_video_writer(video_path, cap):\n",
    "    \"\"\"Setup video writer with same properties as input video.\"\"\"\n",
    "    output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'\n",
    "    fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n",
    "    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n",
    "    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n",
    "    fps = int(cap.get(cv2.CAP_PROP_FPS))\n",
    "    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)), output_path"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎨 Tracking Visualization Function\n",
    "\n",
    "This function draws tracking boxes and IDs on the video frames."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "def draw_tracking_info(frame, track, track_id, box_color=(0, 255, 0)):\n",
    "    \"\"\"Draw tracking box and ID on frame.\"\"\"\n",
    "    ltrb = track.to_ltrb()\n",
    "    x1, y1, x2, y2 = map(int, ltrb)\n",
    "    \n",
    "    # Draw box\n",
    "    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)\n",
    "    \n",
    "    # Draw ID\n",
    "    text = f'ID: {track_id}'\n",
    "    cv2.putText(frame, text, (x1, y1 - 5), \n",
    "                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n",
    "    return (x1, y1, x2, y2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔄 Main Processing Function\n",
    "\n",
    "This function contains the main processing loop that:\n",
    "1. Initializes the YOLO model and DeepSORT tracker\n",
    "2. Processes each frame of the video\n",
    "3. Detects and tracks vehicles\n",
    "4. Counts vehicles crossing the line\n",
    "5. Displays and saves the processed video"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "def process_video():\n",
    "    # Initialize\n",
    "    video_path = select_video()\n",
    "    print(f\"Processing video: {video_path}\")\n",
    "    \n",
    "    model = YOLO(\"yolov8m.pt\")\n",
    "    tracker = DeepSort(max_age=30)\n",
    "    cap = cv2.VideoCapture(video_path)\n",
    "    out, output_path = setup_video_writer(video_path, cap)\n",
    "    \n",
    "    # Tracking parameters\n",
    "    line_position = 300\n",
    "    offset = 10\n",
    "    counted_ids = set()\n",
    "    total_count = 0\n",
    "    \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            break\n",
    "            \n",
    "        # Detect objects\n",
    "        results = model(frame)[0]\n",
    "        detections = []\n",
    "        \n",
    "        for result in results.boxes.data.tolist():\n",
    "            x1, y1, x2, y2, score, class_id = result\n",
    "            if int(class_id) in [2, 3, 5, 7] and score > 0.4:  # Vehicle classes\n",
    "                detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(class_id)))\n",
    "        \n",
    "        # Update tracks\n",
    "        tracks = tracker.update_tracks(detections, frame=frame)\n",
    "        \n",
    "        # Process each track\n",
    "        for track in tracks:\n",
    "            if not track.is_confirmed():\n",
    "                continue\n",
    "                \n",
    "            track_id = track.track_id\n",
    "            x1, y1, x2, y2 = draw_tracking_info(frame, track, track_id)\n",
    "            cy = int((y1 + y2) / 2)\n",
    "            \n",
    "            # Count vehicles crossing the line\n",
    "            if (line_position - offset) < cy < (line_position + offset):\n",
    "                if track_id not in counted_ids:\n",
    "                    counted_ids.add(track_id)\n",
    "                    total_count += 1\n",
    "        \n",
    "        # Draw counting line and display count\n",
    "        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)\n",
    "        cv2.putText(frame, f'Total Vehicles: {total_count}', (20, 50),\n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)\n",
    "        \n",
    "        # Display and save frame\n",
    "        cv2.imshow(\"Vehicle Tracking\", frame)\n",
    "        out.write(frame)\n",
    "        \n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "    \n",
    "    # Cleanup\n",
    "    cap.release()\n",
    "    out.release()\n",
    "    cv2.destroyAllWindows()\n",
    "    print(f\"Processing complete! Total vehicles: {total_count}\")\n",
    "    print(f\"Output saved: {output_path}\")\n",
    "    return total_count, output_path"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 Run the Processing\n",
    "\n",
    "Execute the main processing function to start vehicle detection and tracking. The function will:\n",
    "1. Open a file dialog to select your video\n",
    "2. Process the video in real-time\n",
    "3. Show the tracking visualization\n",
    "4. Save the processed video\n",
    "5. Return the total count and output path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "total_count, output_path = process_video()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Write the notebook content to a file
with open('car_detection_project.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)