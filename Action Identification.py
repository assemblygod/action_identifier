import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def calculate_angle_3d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Resize
        frame = cv2.resize(frame, (640, 360))
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add padding for text
        padding_width = 200  # Adjust this value as needed

        # Get the dimensions of the original image
        height, width, channels = image.shape

        # Create a blank image with the desired dimensions
        padded_image = np.zeros((height, width + padding_width, channels), dtype=np.uint8)

        # Copy the original image onto the padded image
        padded_image[:, :width] = image
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            

            # Draw the rectangle
            # cv2.rectangle(image, top_right_corner, bottom_left_corner, (245, 117, 16), -1)
            
            # Get coordinates
            keypoints = [
                "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", 
                "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
                "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", 
                "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"
            ]

            landmark_coordinates = {}

            # Extract landmark coordinates
            for point in keypoints:
                landmark = mp_pose.PoseLandmark[point].value
                landmark_coordinates[point] = [
                    landmarks[landmark].x, 
                    landmarks[landmark].y, 
                    landmarks[landmark].z
                ]

            # Calculate angles
            angles = {}
            for side in ["LEFT", "RIGHT"]:
                for joint1, joint2, joint3 in [("WRIST", "ELBOW", "SHOULDER"), 
                                               ("ELBOW", "SHOULDER", "HIP"),
                                               ("SHOULDER", "HIP", "KNEE"), 
                                               ("HIP", "KNEE", "ANKLE")]:
                    angle_name = f"{side}_{joint1.lower()}_{joint2.lower()}_{joint3.lower()}"
                    angles[angle_name] = calculate_angle(
                        landmark_coordinates[f"{side}_{joint1}"], 
                        landmark_coordinates[f"{side}_{joint2}"], 
                        landmark_coordinates[f"{side}_{joint3}"]
                    )

            # Extract individual angles
            angle_left_elbow = angles["LEFT_wrist_elbow_shoulder"]
            angle_left_shoulder = angles["LEFT_elbow_shoulder_hip"]
            angle_left_hip = angles["LEFT_shoulder_hip_knee"]
            angle_left_knee = angles["LEFT_hip_knee_ankle"]

            angle_right_elbow = angles["RIGHT_wrist_elbow_shoulder"]
            angle_right_shoulder = angles["RIGHT_elbow_shoulder_hip"]
            angle_right_hip = angles["RIGHT_shoulder_hip_knee"]
            angle_right_knee = angles["RIGHT_hip_knee_ankle"]

            # Determine the action performed
            new_text = None
            if (int(angle_left_knee) > 150 or int(angle_right_knee) > 150) and (int(angle_left_hip) > 150 or int(angle_right_hip) > 150):
                new_text = 'Standing'
            elif int(angle_left_hip) in range(60, 120) or int(angle_right_hip) in range(60, 120):
                new_text = 'Sitting'

            text = []
            if 60 < int(angle_left_shoulder) < 120 and 60 < int(angle_right_shoulder) < 150:
                if int(angle_right_elbow) > 150 and int(angle_left_elbow) > 150:
                    text.append('Both Hands Horizontal')
                elif int(angle_right_elbow) > 150:
                    text.append('Right Hand Horizontal')
                elif int(angle_left_elbow) > 150:
                    text.append('Left Hand Horizontal')

            if int(angle_left_shoulder) >= 150 and int(angle_right_shoulder) >= 150:
                text.append('Both Arms Up')
            elif int(angle_left_shoulder) >= 150:
                text.append('Left Arm Up')
            elif int(angle_right_shoulder) >= 150:
                text.append('Right Arm Up')

            if int(angle_left_shoulder) <= 30 and int(angle_right_shoulder) <= 30:
                text.append('Both Arms Down')
            elif int(angle_left_shoulder) <= 30:
                text.append('Left Arm Down')
            elif int(angle_right_shoulder) <= 30:
                text.append('Right Arm Down')

            # Define text positions for the padded area
            text_x = width + 10  # X-coordinate for text
            text_y = 30  # Initial Y-coordinate for text
            
            # Add "Action" text to padded area
            cv2.putText(padded_image, 'Action', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add new_text to padded area
            if new_text:
                text_y += 30
                cv2.putText(padded_image, new_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Add additional text if any
            text_y += 40  # Adjust space between action and position text
            cv2.putText(padded_image, 'Position', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for line in text:
                text_y += 30
                cv2.putText(padded_image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            pass
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=3), 
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Resize frame
        frame = cv2.resize(frame, (640, 360))
        
        # Display the padded image with the original image and text
        cv2.imshow('Mediapipe Feed', padded_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
