import cv2
import numpy as np
from scipy.interpolate import Rbf
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

REGIONS = {
    "head": [0, 1, 2, 3, 4, 5, 6],
    "neck": [11, 12],
    "shoulders": [11, 12, 13, 14],
    "chest": [11, 12, 23, 24],
    "waist": [23, 24, 25, 26],
    "hips": [23, 24, 25, 26],
    "upper_arms": [13, 14, 15, 16],
    "forearms": [15, 16, 17, 18],
    "hands": [19, 20, 21, 22],
    "thighs": [25, 26, 27, 28],
    "shins": [27, 28, 29, 30],
    "feet": [31, 32]
}

def get_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
        return points

def get_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return []
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return points

def deform_region(image, points, scale_dict):
    if points is None:
        print("Vücut noktaları bulunamadı!")
        return image

    h, w, c = image.shape
    src = np.array(points, dtype=np.float32)
    dst = src.copy()

    mask = np.zeros((h, w), dtype=np.uint8)

    for region, indices in REGIONS.items():
        if region in scale_dict:
            region_points = src[indices]
            center = np.mean(region_points, axis=0)
            dst[indices] = (region_points - center) * scale_dict[region] + center

            # Bölge maskesi oluştur (konveks hull ile)
            hull = cv2.convexHull(region_points.astype(np.int32))
            cv2.drawContours(mask, [hull], -1, 255, -1)

    # Tüm vücut noktalarını ve maskeyi kullanarak deformasyon uygula
    src_unique, indices = np.unique(src, axis=0, return_index=True)
    dst_unique = dst[indices]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    rbf_x = Rbf(src_unique[:, 0], src_unique[:, 1], dst_unique[:, 0], function='multiquadric', epsilon=2)
    rbf_y = Rbf(src_unique[:, 0], src_unique[:, 1], dst_unique[:, 1], function='multiquadric', epsilon=2)

    map_x = rbf_x(grid_x_flat, grid_y_flat).reshape((h, w))
    map_y = rbf_y(grid_x_flat, grid_y_flat).reshape((h, w))

    map_x = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y = np.clip(map_y, 0, h-1).astype(np.float32)

    # Sadece maske içini deforme et, dışı orijinal kalsın
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_norm = mask_blur.astype(np.float32) / 255.0
    output = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    blended = (output * mask_norm[..., None] + image * (1 - mask_norm[..., None])).astype(np.uint8)

    # Geçişleri yumuşat
    blended = cv2.GaussianBlur(blended, (3, 3), 0)

    return blended
