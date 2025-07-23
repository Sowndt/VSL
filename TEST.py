import pickle
import cv2
import numpy as np
import mediapipe as mp
import keypoint_extract as md
from collections import defaultdict, Counter
import time
import logging
import tensorflow as tf  # Thêm import tensorflow
from tensorflow.keras.models import load_model
# Thiết lập logging
logging.basicConfig(filename='prediction_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Tải mô hình từ file h5 thay vì pickle
model = tf.keras.models.load_model('Final_model.h5')

# Tải label binarizer
with open('label_binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)
classes = lb.classes_

# Kích thước chuẩn cho từng loại đặc trưng
MAX_LEN_LEFT_HAND = 21 * 3  # 63
MAX_LEN_RIGHT_HAND = 21 * 3  # 63
MAX_LEN_POSE_BODY = 33 * 3  # 99


# Khởi tạo Mediapipe Hands và Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3,
                       min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mở webcam
cap = cv2.VideoCapture(0)

# Danh sách lưu trữ các phần của từ ghép
word_parts = defaultdict(list)

# Bộ đệm lưu trữ từ được detect trong 33 giây
word_buffer = []  # Lưu tuple (word, timestamp)

# Đặt ngưỡng độ tin cậy
CONFIDENCE_THRESHOLD = 0.999

# Dictionary cho các từ ghép
word_compositions = {
    'Buoi sang': ['buoi_sang#1', 'buoi_sang#2'],
    'Quen': ['quen#1', 'quen#2'],
    'Chao':['chao#1', 'chao#2'],
    'khu_vuon':['khu_vuon#1', 'khu_vuon#2'],
    'don dep':['don_dep#1', 'don_dep#2'],
    'cam on' :['can_on#1', 'cam_on#2'],
    'benh':['benh#1', 'benh#2'],
    'lam viec':['lam_viec#1', 'lam_viec#2'],
}

# Danh sách để lưu trữ các từ đã hiển thị ở dòng trên (theo thứ tự thời gian, cũ đến mới)
displayed_words = []

# Biến đếm frame để giảm tần suất dự đoán
frame_counter = 0
PREDICTION_INTERVAL = 3  # Dự đoán mỗi 33 frame


def extract_and_process_landmarks(frame):
    """Trích xuất và xử lý landmark theo định dạng của mô hình được train"""
    left_hand_landmarks, right_hand_landmarks = md.extract_hand_landmark(frame, hands)
    list_pose = md.extract_body_landmark(frame, pose)

    # Chuyển landmark từ list of lists thành flat list
    flat_left_hand = [coord for lm in left_hand_landmarks for coord in lm]
    flat_right_hand = [coord for lm in right_hand_landmarks for coord in lm]
    flat_pose_body = [coord for lm in list_pose for coord in lm]

    # Padding cho từng loại đặc trưng
    if len(flat_left_hand) < MAX_LEN_LEFT_HAND:
        flat_left_hand.extend([0.0] * (MAX_LEN_LEFT_HAND - len(flat_left_hand)))
    elif len(flat_left_hand) > MAX_LEN_LEFT_HAND:
        flat_left_hand = flat_left_hand[:MAX_LEN_LEFT_HAND]

    if len(flat_right_hand) < MAX_LEN_RIGHT_HAND:
        flat_right_hand.extend([0.0] * (MAX_LEN_RIGHT_HAND - len(flat_right_hand)))
    elif len(flat_right_hand) > MAX_LEN_RIGHT_HAND:
        flat_right_hand = flat_right_hand[:MAX_LEN_RIGHT_HAND]

    if len(flat_pose_body) < MAX_LEN_POSE_BODY:
        flat_pose_body.extend([0.0] * (MAX_LEN_POSE_BODY - len(flat_pose_body)))
    elif len(flat_pose_body) > MAX_LEN_POSE_BODY:
        flat_pose_body = flat_pose_body[:MAX_LEN_POSE_BODY]

    # Ghép nối các đặc trưng
    X_combined = np.concatenate((
        np.array(flat_left_hand, dtype=np.float32),
        np.array(flat_right_hand, dtype=np.float32),
        np.array(flat_pose_body, dtype=np.float32)
    )).reshape(1, -1)

    return X_combined


def predict_word(data_combined, model, lb):
    """Dự đoán từ với mô hình đã train"""
    prediction = model.predict(data_combined, verbose=0)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]

    if confidence >= CONFIDENCE_THRESHOLD:
        return classes[predicted_index]
    return 'Unknown'


def display_words(frame, displayed_words):
    """Hiển thị từ có tần suất cao nhất và danh sách từ đã hiển thị"""
    # Hiển thị từ có tần suất cao nhất (lấy từ cuối cùng của danh sách)
    if displayed_words:
        cv2.putText(frame, f"Most frequent: {str(displayed_words[-1])}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Most frequent: Unknown", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # # Hiển thị các từ đã xuất hiện ở dòng dưới (10 từ mới nhất)
    # if displayed_words:
    #     # Chuyển đổi tất cả các phần tử thành chuỗi trước khi join
    #     displayed_text = ' '.join(str(word) for word in displayed_words[-10:])
    #     cv2.putText(frame, displayed_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # else:
    #     cv2.putText(frame, "", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def process_composed_words(word_buffer, word_compositions, displayed_words, current_time):
    """Kiểm tra và ghép từ ghép, loại bỏ các từ thành phần"""
    for composed_word, components in word_compositions.items():
        recent_components = [w for w, t in word_buffer if w in components and current_time - t <= 2]
        if set(components).issubset(set(recent_components)):
            # Thêm từ ghép vào danh sách hiển thị
            if composed_word not in displayed_words:  # Chỉ thêm nếu chưa có
                displayed_words.append(composed_word)
            logging.info(f"Composed word detected: {composed_word}")
            # Xóa các từ thành phần khỏi bộ đệm
            word_buffer[:] = [(w, t) for w, t in word_buffer if w not in components]


def main():
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Lỗi: Không thể đọc khung hình.")
            break

        current_time = time.time()
        word_buffer[:] = [(w, t) for w, t in word_buffer if current_time - t <= 2]

        # Xử lý frame hiện tại mỗi PREDICTION_INTERVAL frame
        global frame_counter
        if frame_counter % PREDICTION_INTERVAL == 0:
            # Trích xuất và xử lý landmark theo định dạng mới
            data_combined = extract_and_process_landmarks(frame)

            if data_combined is not None:
                predicted_label = predict_word(data_combined, model, lb)
                if predicted_label != 'Unknown':
                    word_buffer.append((predicted_label, current_time))
                    logging.info(f"Predicted: {predicted_label}")
                    if not any(predicted_label in components for components in word_compositions.values()):
                        if predicted_label not in displayed_words:
                            displayed_words.append(predicted_label)

        frame_counter += 1

        # Kiểm tra và ghép từ ghép
        process_composed_words(word_buffer, word_compositions, displayed_words, current_time)

        # Hiển thị từ có tần suất cao nhất
        if word_buffer:
            most_frequent_word = Counter([w for w, _ in word_buffer]).most_common(1)[0][0]
            # Nếu từ có tần suất cao nhất chưa có trong displayed_words, thêm vào
            if not any(most_frequent_word in components for components in word_compositions.values()):
                displayed_words.append(most_frequent_word)
        else:
            most_frequent_word = 'Unknown'

        # Hiển thị kết quả
        display_words(frame, displayed_words)

        # Hiển thị frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()