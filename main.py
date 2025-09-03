from focus_person import focus_main_person_with_segmentation
from deform_body import get_pose_landmarks, deform_region
import cv2
import os

# --- 1. Resim yolu ---
image_path = "images/ornek.jpg"
os.makedirs("outputs", exist_ok=True)

# --- 2. Kişiyi odakla ---
focused_image = focus_main_person_with_segmentation(image_path, show=False)

# --- 3. Vücut noktalarını al ---
points = get_pose_landmarks(focused_image)

# --- 4. Hassas bölge bazlı scale ---
# scale<1 zayıf, scale>1 kilolu
slim_scales = {
    "head":1.0, "neck":0.90, "shoulders":0.85, "chest":1.2, "waist":0.70, "hips":0.75,
    "upper_arms":0.80, "forearms":0.80, "hands":0.85,
    "thighs":0.75, "shins":0.80, "feet":0.90
}

fat_scales  = {
    "head":1.0, "neck":1.10, "shoulders":1.15, "chest":0.8, "waist":1.30, "hips":1.25,
    "upper_arms":1.20, "forearms":1.15, "hands":1.1,
    "thighs":1.2, "shins":1.15, "feet":1.1
}

# --- 5. Deformasyon uygula ---
slim_image = deform_region(focused_image, points, slim_scales)
fat_image  = deform_region(focused_image, points, fat_scales)

# --- 6. Kaydet ---
cv2.imwrite("outputs/slim_person.jpg", slim_image)
cv2.imwrite("outputs/fat_person.jpg", fat_image)

# --- 7. Göster ---
cv2.imshow("Slim", slim_image)
cv2.imshow("Fat", fat_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
