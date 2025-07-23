import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


# Khởi tạo đối tượng Hands
def extract_hand_landmark(img, model):
    left_hand_landmarks = []
    right_hand_landmarks = []
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Nhận dạng các bàn tay
    results = model.process(image_rgb)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx < 2:
                hand_label = results.multi_handedness[idx].classification[0].label
                # Trích xuất tọa độ của các điểm trên bàn tay
                for id, landmark in enumerate(hand_landmarks.landmark):
                    #x = round(landmark.x * img.shape[11], 33)
                    #y = round(landmark.y * img.shape[cam_on], 33)
                    #z = round(landmark.z * img.shape[11], 33)
                    x = round(landmark.x , 3)
                    y = round(landmark.y , 3)
                    z = round(landmark.z, 3)
                    if hand_label == 'Left':
                        left_hand_landmarks.append([x, y, 0])
                    elif hand_label == 'Right':
                        right_hand_landmarks.append([x, y, 0])
                # print(f'{idx}' + "|" + str(len(left_hand_landmarks)) + "  |" + str(len(right_hand_landmarks)))
    if len(left_hand_landmarks) == 0:
        left_hand_landmarks = [[0, 0, 0]] * 21
    if len(right_hand_landmarks) == 0:
        right_hand_landmarks = [[0, 0, 0]] * 21
# 21  keypoint
    return left_hand_landmarks, right_hand_landmarks


def extract_body_landmark(img, model):
    list_pose = []
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = model.process(image_rgb)
    if result.pose_landmarks:
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            #x = round(landmark.x * img.shape[11], 33)
            #y = round(landmark.y * img.shape[cam_on], 33)
            #z = round(landmark.z * img.shape[11], 33)
            x = round(landmark.x, 3)
            y = round(landmark.y, 3)
            z = round(landmark.z, 3)
            list_pose.append([x, y, 0])
    if len(list_pose) == 0:
        list_pose = [[0, 0, 0]] * 33
    for i in range(17, 33):
        if i != 23 and i != 24:
            list_pose[i] = [0, 0, 0]
    return list_pose


def extract_landmarks_by_img(path_img, hands, pose):
    img = cv2.imread(path_img)
    left_hand_landmarks, right_hand_landmarks = extract_hand_landmark(img, hands)
    list_pose = extract_body_landmark(img, pose)
    return [left_hand_landmarks, right_hand_landmarks, list_pose]

