import dlib
from imutils import face_utils
import numpy as np

def get_landmark(image_tensor):
    # Load face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'SelfBlendedImage/src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    # Convert tensor to numpy array
    frame = np.transpose(image_tensor.numpy(), (1, 2, 0))

    # Detect faces
    faces = face_detector(frame, 1)

    if len(faces) == 0:
        print('No faces detected in the image.')
        return None

    # Initialize landmarks list
    landmarks = []

    for face_idx, face in enumerate(faces):
        # Predict landmarks
        landmark = face_predictor(frame, face)
        landmark = face_utils.shape_to_np(landmark)
        # Append landmarks to the list
        landmarks.append(landmark)

    return np.array(landmarks)

if __name__ == '__main__':
    # Example usage
    import torch

    # Example input tensor (RGB image)
    input_image_tensor = torch.randn(3, 224, 224)

    # Get landmarks
    landmarks = get_landmark(input_image_tensor)

    if landmarks is not None:
        print('Landmarks shape:', landmarks.shape)
    else:
        print('No landmarks detected.')
