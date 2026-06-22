from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
shutil.copy(path, "models/yolov8n-face.pt")
print("Đã copy model vào models/yolov8n-face.pt")