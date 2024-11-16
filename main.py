import cv2
import numpy as np
import mediapipe as mp

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

# Initialize video capture
cap = cv2.VideoCapture(0)

mode = None  # Variable to store the current exercise mode

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

            # Get coordinates of relevant landmarks for exercise analysis (shoulders, elbows, knees)
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

            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            feedback_text = ""

            # Example feedback based on mode selected
            if mode == 'pushup':
                left_arm_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
                right_arm_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)

                # Check for good push-up form: elbows should be less than 90 degrees
                if left_arm_angle < 90 and right_arm_angle < 90:
                    feedback_text = "Good push-up form!"
                else:
                    feedback_text += "Keep your elbows past 90 degrees!"

            elif mode == 'squat':
                left_knee_angle = calculate_angle(hip_left, knee_left, elbow_left)  # Using hip and elbow for reference
                right_knee_angle = calculate_angle(hip_right, knee_right, elbow_right)

                # Check if user is standing first (hips above knees)
                if hip_left[1] > knee_left[1] and hip_right[1] > knee_right[1]:
                    feedback_text += "Good squat form."
                elif left_knee_angle < 100 and right_knee_angle < 100:  # Knees bent at less than or equal to 100 degrees
                    feedback_text += "Go lower in your squat/"
                else:
                    feedback_text += "Good squat form"

            elif mode == 'plank':
                left_shoulder_position = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder_position = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip_position = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip_position = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Check if arms are straight (wrist and shoulder positions)
                arms_straight = (left_shoulder_position.visibility > 0.5 and 
                                 right_shoulder_position.visibility > 0.5)

                # Check if back is straight by comparing shoulder and hip positions
                back_straight_condition = (abs(left_shoulder_position.y - left_hip_position.y) < 0.15 and 
                                           abs(right_shoulder_position.y - right_hip_position.y) < 0.15)

                if arms_straight and back_straight_condition:
                    feedback_text += "Good plank position!"
                else:
                    feedback_text += "Keep your arms straight and back aligned!"

            else:
                feedback_text += "Select a mode (0 for push-up, 1 for squat, 2 for plank)"

            # Display feedback on the frame
            cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with predictions and feedback
        cv2.imshow('Exercise Posture Detection', frame)

        # Check for key presses to change modes
        key = cv2.waitKey(1)
        if key == ord('0'):
            mode = 'pushup'
        elif key == ord('1'):
            mode = 'squat'
        elif key == ord('2'):
            mode = 'plank'
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
