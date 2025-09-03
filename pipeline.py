import uuid
import base64
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from deform_body import get_pose_landmarks, deform_region


class ModelRegistry:
	"""Singleton-like registry to hold heavy models loaded once."""
	_yolo_model: Optional[YOLO] = None

	@classmethod
	def get_yolo(cls) -> YOLO:
		if cls._yolo_model is None:
			cls._yolo_model = YOLO("yolov8n.pt")
		return cls._yolo_model


def read_image_to_bgr(image_bytes: bytes) -> np.ndarray:
	arr = np.frombuffer(image_bytes, dtype=np.uint8)
	image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError("Invalid image data")
	return image


def encode_bgr_image_to_base64_jpeg(image: np.ndarray, quality: int = 90) -> str:
	encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
	success, buf = cv2.imencode('.jpg', image, encode_params)
	if not success:
		raise RuntimeError("Failed to encode image")
	return base64.b64encode(buf.tobytes()).decode('ascii')


def focus_main_person_with_segmentation_bgr(image_bgr: np.ndarray, blur_strength: int = 55, mask_blur: int = 15) -> np.ndarray:
	"""
	Find the most prominent person with YOLO, build a refined foreground mask with MediaPipe Selfie Segmentation,
	and blur background while keeping the main person sharp.
	"""
	model = ModelRegistry.get_yolo()
	image = image_bgr
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	h, w, _ = image.shape

	results = model(image)[0]

	person_boxes: list[Tuple[int, Tuple[int, int, int, int]]] = []
	for box in results.boxes:
		cls = int(box.cls[0])
		if cls == 0:
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			area = (x2 - x1) * (y2 - y1)
			person_boxes.append((area, (x1, y1, x2, y2)))

	if not person_boxes:
		return image

	person_boxes.sort(reverse=True, key=lambda x: x[0])
	x1, y1, x2, y2 = person_boxes[0][1]

	import mediapipe as mp
	mp_selfie_segmentation = mp.solutions.selfie_segmentation
	with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
		seg_results = selfie_segmentation.process(image_rgb)
		mask = seg_results.segmentation_mask

	person_mask = np.zeros_like(mask)
	person_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
	condition = person_mask > 0.5

	person_mask_uint8 = (condition.astype(np.uint8) * 255)
	kernel = np.ones((5, 5), np.uint8)
	person_mask_uint8 = cv2.morphologyEx(person_mask_uint8, cv2.MORPH_CLOSE, kernel)
	person_mask_uint8 = cv2.morphologyEx(person_mask_uint8, cv2.MORPH_OPEN, kernel)
	person_mask_uint8 = cv2.GaussianBlur(person_mask_uint8, (mask_blur, mask_blur), 0)

	refined_mask = person_mask_uint8.astype(np.float32) / 255.0
	refined_mask = refined_mask[..., None]

	blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
	output = (refined_mask * image + (1 - refined_mask) * blurred).astype(np.uint8)
	return output


SLIM_SCALES: Dict[str, float] = {
	"head": 1.0, "neck": 0.95, "shoulders": 0.9, "chest": 0.8, "waist": 0.75, "hips": 0.8,
	"upper_arms": 0.85, "forearms": 0.85, "hands": 0.9,
	"thighs": 0.8, "shins": 0.85, "feet": 0.95
}

FAT_SCALES: Dict[str, float] = {
	"head": 1.0, "neck": 1.05, "shoulders": 1.1, "chest": 1.2, "waist": 1.25, "hips": 1.2,
	"upper_arms": 1.15, "forearms": 1.1, "hands": 1.05,
	"thighs": 1.15, "shins": 1.1, "feet": 1.05
}

ALL_REGIONS = sorted(set(SLIM_SCALES.keys()) | set(FAT_SCALES.keys()))


def compute_scale_dict_from_level(weight_level: int, max_level: int = 10) -> Dict[str, float]:
	"""
	weight_level in [-max_level, max_level]. 0 means no change.
	Negative goes towards SLIM_SCALES, positive towards FAT_SCALES.
	We interpolate from 1.0 to the target scales using |level|/max_level.
	"""
	if weight_level == 0:
		return {region: 1.0 for region in ALL_REGIONS}

	alpha = min(max(abs(weight_level) / float(max_level), 0.0), 1.0)
	target = SLIM_SCALES if weight_level < 0 else FAT_SCALES
	scales: Dict[str, float] = {}
	for region in ALL_REGIONS:
		target_scale = target.get(region, 1.0)
		scales[region] = 1.0 + (target_scale - 1.0) * alpha
	return scales


def process_image_with_weight(image_bgr: np.ndarray, weight_level: int) -> np.ndarray:
	focused = focus_main_person_with_segmentation_bgr(image_bgr)
	points = get_pose_landmarks(focused)
	scales = compute_scale_dict_from_level(weight_level)
	result = deform_region(focused, points, scales)
	return result


class ImageStore:
	"""In-memory image store for the web app. Not for production use."""
	_store: Dict[str, np.ndarray] = {}

	@classmethod
	def put(cls, image_bgr: np.ndarray) -> str:
		image_id = str(uuid.uuid4())
		cls._store[image_id] = image_bgr
		return image_id

	@classmethod
	def get(cls, image_id: str) -> np.ndarray:
		if image_id not in cls._store:
			raise KeyError("Image not found")
		return cls._store[image_id]

	@classmethod
	def replace(cls, image_id: str, image_bgr: np.ndarray) -> None:
		cls._store[image_id] = image_bgr 