import os
import pickle
import mediapipe as mp
import keypoint_extract as md
import cv2
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Đường dẫn tuyệt đối
DATA_DIR = os.path.abspath('VSL_DATA')

data_left_hand = []
data_right_hand = []
data_pose_body = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"Lỗi: Thư mục {DATA_DIR} không tồn tại. Vui lòng kiểm tra lại.")
    exit()


def augment_image(image):
    """Custom image augmentation using OpenCV"""
    height, width = image.shape[:2]

    # Random scaling
    scale = random.uniform(0.8, 1.2)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height))

    # Random rotation
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (new_width, new_height))

    # Random shear
    shear_angle = random.uniform(-0.2, 0.2)  # Góc shear trong khoảng -0.2 đến 0.2
    shear_matrix = np.float32([
        [1, shear_angle, 0],
        [0, 1, 0]
    ])
    image = cv2.warpAffine(image, shear_matrix, (width, height))

    # Random translation
    tx = random.randint(-width // 10, width // 10)
    ty = random.randint(-height // 10, height // 10)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, matrix, (width, height))

    return image


# Kích thước chuẩn cho từng loại đặc trưng
MAX_LEN_LEFT_HAND = 21 * 3  # 63
MAX_LEN_RIGHT_HAND = 21 * 3  # 63
MAX_LEN_POSE_BODY = 33 * 3  # 99

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        print(f"Xử lý thư mục con: {dir_}")
        for img_path in os.listdir(dir_path):
            full_img_path = os.path.join(dir_path, img_path)
            if os.path.isfile(full_img_path) and full_img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = cv2.imread(full_img_path)
                    if img is None:
                        print(f"Lỗi: Không thể đọc file {full_img_path}. Bỏ qua.")
                        continue

                    num_augmentations_per_image = 5
                    for _ in range(num_augmentations_per_image):
                        image_augmented = augment_image(img.copy())

                        left_hand_landmarks, right_hand_landmarks = md.extract_hand_landmark(image_augmented, hands)
                        list_pose = md.extract_body_landmark(image_augmented, pose)

                        # Chuyển landmark từ list of lists thành flat list
                        flat_left_hand = [coord for lm in left_hand_landmarks for coord in lm]
                        flat_right_hand = [coord for lm in right_hand_landmarks for coord in lm]
                        flat_pose_body = [coord for lm in list_pose for coord in lm]

                        # Padding cho từng loại đặc trưng
                        flat_left_hand.extend([0.0] * (MAX_LEN_LEFT_HAND - len(flat_left_hand)))
                        flat_right_hand.extend([0.0] * (MAX_LEN_RIGHT_HAND - len(flat_right_hand)))
                        flat_pose_body.extend([0.0] * (MAX_LEN_POSE_BODY - len(flat_pose_body)))

                        # Cắt bớt nếu vượt quá kích thước tối đa
                        flat_left_hand = flat_left_hand[:MAX_LEN_LEFT_HAND]
                        flat_right_hand = flat_right_hand[:MAX_LEN_RIGHT_HAND]
                        flat_pose_body = flat_pose_body[:MAX_LEN_POSE_BODY]

                        # Chuyển đổi sang numpy array
                        np_left_hand = np.array(flat_left_hand, dtype=np.float32)
                        np_right_hand = np.array(flat_right_hand, dtype=np.float32)
                        np_pose_body = np.array(flat_pose_body, dtype=np.float32)

                        # Thêm vào các danh sách
                        data_left_hand.append(np_left_hand)
                        data_right_hand.append(np_right_hand)
                        data_pose_body.append(np_pose_body)
                        labels.append(dir_)

                except Exception as e:
                    print(f"Lỗi xử lý {full_img_path}: {e}. Bỏ qua file này.")
                    continue
            else:
                print(f"Bỏ qua {full_img_path}: Không phải file ảnh (.jpg, .jpeg, .png).")

# Ghi dữ liệu vào file pickle
try:
    with open('data_separate.pickle', 'wb') as f:
        pickle.dump({
            'data_left_hand': data_left_hand,
            'data_right_hand': data_right_hand,
            'data_pose_body': data_pose_body,
            'labels': labels
        }, f)
    print("Dữ liệu đã được lưu vào data_separate.pickle thành công.")
    print(f"Tổng số mẫu dữ liệu được tạo: {len(labels)}")
except Exception as e:
    print(f"Lỗi khi ghi file data_separate.pickle: {e}")
finally:
    hands.close()
    pose.close()