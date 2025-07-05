# 🚗 Car Detection and Tracking Project


# 📚 Import Required Libraries
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 🎯 Video Selection Function
def select_video():
    """Open file dialog to select video file."""
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not video_path:
        raise ValueError("No video file selected")
    return video_path

# 📹 Video Writer Setup Function
def setup_video_writer(video_path, cap):
    """Setup video writer with same properties as input video."""
    output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)), output_path

# 🎨 Tracking Visualization Function
def draw_tracking_info(frame, track, track_id, box_color=(0, 0, 255)):
    """Draw tracking box and ID on frame."""
    x1, y1, x2, y2 = map(int, track.to_ltrb())
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    return (x1, y1, x2, y2)

# 🔄 Main Processing Function
def process_video():
    try:
        # Initialize
        print("Opening file dialog to select video...")
        video_path = select_video()
        print(f"Selected video: {video_path}")
        
        print("Loading YOLO model...")
        model = YOLO("yolov8m.pt")
        print("YOLO model loaded successfully!")
        
        print("Initializing DeepSORT tracker...")
        tracker = DeepSort(max_age=30)
        
        print("Opening video capture...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        print("Setting up video writer...")
        out, output_path = setup_video_writer(video_path, cap)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video dimensions: {frame_width}x{frame_height}")
        
        # Tracking parameters
        line_position = frame_height // 2  # Set line to middle of frame
        offset = 10
        counted_ids = set()
        total_count = 0
        
        frame_count = 0
        print("Starting video processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frame_count}")
            
            # Detect objects
            results = model(frame)[0]
            detections = []
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if int(class_id) in [2, 3, 5, 7] and score > 0.4:  # Vehicle classes
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(class_id)))
            
            # Update tracks
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                x1, y1, x2, y2 = draw_tracking_info(frame, track, track_id)
                cy = int((y1 + y2) / 2)
                
                # Count vehicles crossing the line
                if (line_position - offset) < cy < (line_position + offset):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_count += 1
                        print(f"Vehicle {track_id} counted! Total: {total_count}")
            
            # Draw counting line and display count
            cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)
            cv2.putText(frame, f'Total Vehicles: {total_count}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display and save frame
            cv2.imshow("Vehicle Tracking", frame)
            out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user")
                break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete! Total vehicles: {total_count}")
        print(f"Output saved: {output_path}")
        return total_count, output_path
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Cleanup in case of error
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        raise

# 🚀 Run the Processing
if __name__ == "__main__":
    total_count, output_path = process_video() 