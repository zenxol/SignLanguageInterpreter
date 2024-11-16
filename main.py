import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import ttk

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First joint
    b = np.array(b)  # Mid joint
    c = np.array(c)  # End joint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize angle to be between 0 and 180 degrees
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to check if a landmark is visible
def landmark_check(landmark):
    return 0 <= landmark[0] <= 1 and 0 <= landmark[1] <= 1

# Function to start the exercise mode
def start_exercise(mode):
    root.destroy()  # Close the start screen
    run_exercise_detection(mode)

# Create start screen
root = tk.Tk()
root.title("Exercise Selection")
root.geometry("300x200")

label = ttk.Label(root, text="Select exercise mode:", font=("Arial", 14))
label.pack(pady=20)

pushup_button = ttk.Button(root, text="Push-ups", command=lambda: start_exercise('pushup'))
pushup_button.pack(pady=10)

squat_button = ttk.Button(root, text="Squats", command=lambda: start_exercise('squat'))
squat_button.pack(pady=10)

plank_button = ttk.Button(root, text="Plank", command=lambda: start_exercise('plank'))
plank_button.pack(pady=10)

# Function to run the exercise detection
def run_exercise_detection(mode):
    cap = cv2.VideoCapture(0)
    
    # Initialize variables
    rep_count = 0
    exercise_in_progress = False
    plank_start_time = None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get coordinates of relevant landmarks for exercise analysis
                landmarks = results.pose_landmarks.landmark
                
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                feedback_text = ""

                # Check if all relevant joints are visible
                joints_visible = all(map(landmark_check, [
                    shoulder_left, shoulder_right, elbow_left, elbow_right,
                    wrist_left, wrist_right, hip_left, hip_right, knee_left, knee_right
                ]))

                if mode == 'pushup':
                    if not joints_visible:
                        feedback_text += "Ensure all joints are visible for push-ups."
                    else:
                        left_arm_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
                        right_arm_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)

                        if left_arm_angle <= 90 and right_arm_angle <= 90:
                            feedback_text += "Good push-up form!"
                            if not exercise_in_progress:  
                                exercise_in_progress = True  
                        else:
                            if exercise_in_progress:  
                                rep_count += 1  
                                feedback_text += "Push-up complete!"
                            exercise_in_progress = False 
                            feedback_text += "Lower your body until elbows are at 90 degrees."

                elif mode == 'squat':
                    # Implement squat detection logic here
                    pass

                elif mode == 'plank':
                    # Implement plank detection logic here
                    pass

                # Display feedback and rep count on the frame
                cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if mode != 'plank':
                    cv2.putText(frame, f"Reps: {rep_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame with predictions and feedback
            cv2.imshow('Exercise Posture Detection', frame)

            # Check for key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Start the Tkinter event loop
root.mainloop()
