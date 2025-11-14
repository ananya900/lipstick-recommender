import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

mp_face = mp.solutions.face_mesh

def extract_skin(image_bgr):
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for lm in results.multi_face_landmarks[0].landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(mask, (x, y), 1, 255, -1)

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        skin = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        return skin

def get_dominant_rgb(image_bgr, k=1):
    pixels = image_bgr.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(pixels) == 0:
        return None

    pixels_rgb = pixels[:, ::-1]
    k = min(k, len(pixels_rgb))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels_rgb)

    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)

    dominant = centers[np.argmax(counts)]
    return tuple(map(int, dominant))
