# train.py
import os
import cv2
import pickle
from keras_facenet import FaceNet
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

embedder = FaceNet()
data, labels = [], []

base_dir = "training_data"

for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = embedder.extract(image_rgb, threshold=0.95)
            if results:
                embedding = results[0]['embedding']
                data.append(embedding)
                labels.append(person)

os.makedirs("embeddings", exist_ok=True)
with open("embeddings/encodings.pickle", "wb") as f:
    pickle.dump({"embeddings": data, "names": labels}, f)

print("✅ FaceNet embeddings saved to embeddings/encodings.pickle")

