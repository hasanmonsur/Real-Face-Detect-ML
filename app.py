import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from skimage import exposure

# Add this before importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
#Load Kaggle Dataset

# Load metadata (if available)
df = pd.read_csv('kaggle/input/celeba-dataset/list_attr_celeba.csv')

# Define paths
image_dir = 'kaggle/input/celeba-dataset/img_align_celeba'
output_dir = 'kaggle/working/processed_faces'
#output_img_dir = 'kaggle/working/processed_faces/img_align_celeba'

os.makedirs(output_dir, exist_ok=True)

#Preprocess Data

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

#Histogram Equalization (Optional)
def enhance_contrast(img):
    img = exposure.equalize_hist(img)
    return img


#Face Detection

detector = MTCNN()

def detect_face_mtcnn(img):
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
        face = img[y:y+h, x:x+w]
        return face
    else:
        print("No face detected in this image.")
        return None  # Skip this image


#Using Haar Cascades (Traditional)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    return None



# call full and all function

failed_images = []

for idx, filename in enumerate(os.listdir(image_dir)):
    try:
        image_path = os.path.join(image_dir, filename)
        img = preprocess_image(image_path)
        
        if img is None:
            failed_images.append(filename)
            continue
        
        face = detect_face_mtcnn(img)
        if face is not None:
            face = cv2.resize(face, (160, 160))  # Resize to target size
            # Save or process the face
            output_path = os.path.join(output_dir, f'face_{idx}.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(face*255, cv2.COLOR_RGB2BGR))
            #print(f"Saved: {save_path}")
        else:
            failed_images.append(filename)
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"Failed to detect faces in {len(failed_images)} images.")


#Data Augmentation (Optional)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Example: Augment one image
img = np.expand_dims(face, axis=0)
aug_iter = datagen.flow(img, batch_size=1)
augmented = [next(aug_iter)[0].astype('uint8') for _ in range(5)]  # Generate 5 variants


#Verify Results

import matplotlib.pyplot as plt

def plot_faces(sample_dir=output_dir, n=5):
    plt.figure(figsize=(15, 5))
    for i, filename in enumerate(os.listdir(sample_dir)[:n]):
        img = cv2.imread(os.path.join(sample_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

plot_faces()



#Save Metadata
processed_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
pd.DataFrame({'processed_files': processed_files}).to_csv('processed_metadata.csv', index=False)

