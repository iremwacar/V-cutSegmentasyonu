import cv2
import numpy as np
from scipy.interpolate import Rbf
import mediapipe as mp

mp_pose = mp.solutions.pose

# Bölge indeksleri (MediaPipe 33 noktası)
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
    """MediaPipe Pose ile 33 vücut noktası al"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return points

def deform_region(image, points, scale_dict):
    """
    Her hassas bölge için scale uygular.
    scale_dict örn: {"chest":0.8, "waist":0.75, "thighs":0.85}
    """
    if points is None:
        print("Vücut noktaları bulunamadı!")
        return image

    h, w, c = image.shape
    src = np.array(points, dtype=np.float32)
    dst = src.copy()

    # Her bölge için scale uygula
    for region, indices in REGIONS.items():
        if region in scale_dict:
            region_points = src[indices]
            center_x = np.mean(region_points[:,0])
            center_y = np.mean(region_points[:,1])
            dst[indices] = (region_points - [center_x, center_y]) * scale_dict[region] + [center_x, center_y]

    # Grid oluştur
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    # TPS benzeri RBF
    rbf_x = Rbf(src[:,0], src[:,1], dst[:,0], function='thin_plate')
    rbf_y = Rbf(src[:,0], src[:,1], dst[:,1], function='thin_plate')

    map_x = np.clip(rbf_x(grid_x_flat, grid_y_flat).reshape((h,w)), 0, w-1).astype(np.float32)
    map_y = np.clip(rbf_y(grid_x_flat, grid_y_flat).reshape((h,w)), 0, h-1).astype(np.float32)

    output = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return output
