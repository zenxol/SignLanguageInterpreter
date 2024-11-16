import cv2
import numpy as np
import mediapipe as mp
import time
import customtkinter as ctk

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set appearance mode and color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def landmark_check(landmark):
    return 0 <= landmark[0] <= 1 and 0 <= landmark[1] <= 1

def run_exercise_detection(initial_mode):
    cap = cv2.VideoCapture(0)
    
    # Get screen dimensions
    cv2.namedWindow('Exercise Posture Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Exercise Posture Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize variables
    feedback_timer = 0
    exercise_state = "up"
    rep_count = 0
    last_rep_time = 0
    mode = initial_mode
    feedback_text = "Starting exercise detection..."

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize frame to fit screen while maintaining aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            if screen_width / screen_height > aspect_ratio:
                new_width = int(screen_height * aspect_ratio)
                new_height = screen_height
            else:
                new_width = screen_width
                new_height = int(screen_width / aspect_ratio)
            
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Create a black background image
            background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # Calculate position to center the resized frame
            y_offset = (screen_height - new_height) // 2
            x_offset = (screen_width - new_width) // 2
            
            # Place the resized frame on the background
            background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(background[y_offset:y_offset+new_height, x_offset:x_offset+new_width], 
                                          results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                
                # Extract relevant landmarks
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
                ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                joints_visible = all(map(landmark_check, [
                    shoulder_left, shoulder_right, elbow_left, elbow_right,
                    wrist_left, wrist_right, hip_left, hip_right, knee_left, knee_right
                ]))

                if not joints_visible:
                    feedback_text = "Ensure all joints are visible."
                elif mode == 'pushup':
                    left_arm_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    right_arm_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)
                    avg_arm_angle = (left_arm_angle + right_arm_angle) / 2

                    current_time = time.time()
                    if avg_arm_angle <= 90 and exercise_state == "up":
                        exercise_state = "down"
                        rep_count += 1
                        feedback_text = "PUSH-UP COMPLETE, GOOD FORM!"
                        feedback_timer = current_time + 0.5
                    elif current_time <= feedback_timer:
                        feedback_text = "PUSH-UP COMPLETE, GOOD FORM!"
                    else:
                        if avg_arm_angle >= 160:
                            exercise_state = "up"
                        feedback_text = "MAINTAIN PROPER FORM"

                elif mode == 'squat':
                    left_knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
                    right_knee_angle = calculate_angle(hip_right, knee_right, ankle_right)
                    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                    current_time = time.time()
                    if avg_knee_angle <= 100 and exercise_state == "up":
                        exercise_state = "down"
                        rep_count += 1
                        feedback_text = "SQUAT COMPLETE, GOOD FORM!"
                        feedback_timer = current_time + 0.5
                    elif current_time <= feedback_timer:
                        feedback_text = "SQUAT COMPLETE, GOOD FORM!"
                    else:
                        if avg_knee_angle >= 160:
                            exercise_state = "up"
                        feedback_text = "MAINTAIN PROPER FORM"
            else:
                feedback_text = "No pose detected. Please ensure you're in frame."

            # Display feedback and rep count
            cv2.putText(background, f"Mode: {mode.capitalize()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(background, feedback_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(background, f"Reps: {rep_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(background, "Press 'q' to quit, 's' to switch mode", (10, screen_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Exercise Posture Detection', background)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                switch_mode()
                return

    cap.release()
    cv2.destroyAllWindows()
    show_exercise_selection()

def switch_mode():
    root = ctk.CTk()
    root.title("Switch Mode")
    root.geometry("400x300")

    frame = ctk.CTkFrame(master=root)
    frame.pack(pady=20, padx=60, fill="both", expand=True)

    label = ctk.CTkLabel(master=frame, text="Select new mode:", font=("Roboto", 24))
    label.pack(pady=12, padx=10)

    pushup_button = ctk.CTkButton(master=frame, text="Push-ups", command=lambda: start_exercise(root, 'pushup'))
    pushup_button.pack(pady=12, padx=10)

    squat_button = ctk.CTkButton(master=frame, text="Squats", command=lambda: start_exercise(root, 'squat'))
    squat_button.pack(pady=12, padx=10)

    exit_button = ctk.CTkButton(master=frame, text="Exit", command=root.destroy)
    exit_button.pack(pady=12, padx=10)

    root.mainloop()

def show_exercise_selection():
    root = ctk.CTk()
    root.title("Exercise Selection")
    root.geometry("400x300")

    frame = ctk.CTkFrame(master=root)
    frame.pack(pady=20, padx=60, fill="both", expand=True)

    label = ctk.CTkLabel(master=frame, text="Select exercise mode:", font=("Roboto", 24))
    label.pack(pady=12, padx=10)

    pushup_button = ctk.CTkButton(master=frame, text="Push-ups", command=lambda: start_exercise(root, 'pushup'))
    pushup_button.pack(pady=12, padx=10)

    squat_button = ctk.CTkButton(master=frame, text="Squats", command=lambda: start_exercise(root, 'squat'))
    squat_button.pack(pady=12, padx=10)

    root.mainloop()

def start_exercise(root, mode):
    root.destroy()
    run_exercise_detection(mode)

if __name__ == "__main__":
    try:
        show_exercise_selection()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure all Tkinter windows are closed
        for widget in ctk.CTk().winfo_children():
            if isinstance(widget, ctk.CTk):
                widget.destroy()
