
# # train_full.py
# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # === USER CONFIGURATION ===
# DATA_DIR      = 'C:/Users/HUAWEI/Desktop/28_may_submission/chest_xray'  # change to your local path
# BATCH_SIZE    = 16
# IMG_SIZE      = (224, 224)
# EPOCHS        = 5
# LEARNING_RATE = 1e-4
# # ==========================

# # === IMAGE PREPROCESSING FUNCTION ===
# def preprocess_image(img):
#     """Assumes image is loaded as array by ImageDataGenerator (float32, 0-255)."""
#     img = img.astype(np.uint8)
#     # Histogram equalization (grayscale)
#     if img.shape[-1] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img = cv2.equalizeHist(img)
#     img = cv2.medianBlur(img, 5)
#     gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
#     img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
#     img = img.astype(np.float32) / 255.0
#     img = np.stack([img]*3, axis=-1)
#     return img

# # === MODEL DEFINITION ===
# def build_model(num_classes, input_shape=(224,224,3)):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#     x = GlobalAveragePooling2D()(base_model.output)
#     x = Dropout(0.5)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=outputs)
#     return model

# # === MAIN FUNCTION ===
# def main():
#     train_datagen = ImageDataGenerator(
#         preprocessing_function=preprocess_image,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=True
#     )
#     val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

#     train_gen = train_datagen.flow_from_directory(
#         os.path.join(DATA_DIR, 'train'),
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         shuffle=True
#     )
#     val_gen = val_datagen.flow_from_directory(
#         os.path.join(DATA_DIR, 'val'),
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         shuffle=False
#     )

#     num_classes = len(train_gen.class_indices)
#     model = build_model(num_classes, input_shape=IMG_SIZE + (3,))
#     model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

#     model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=EPOCHS
#     )

#     test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
#     test_gen = test_datagen.flow_from_directory(
#         os.path.join(DATA_DIR, 'test'),
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         shuffle=False
#     )
#     loss, acc = model.evaluate(test_gen)
#     print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

#     model.save('pneumonia_model_full.h5')
#     print("Model saved to pneumonia_model_full.h5")

# if __name__ == '__main__':
#     main()




import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = 'C:/Users/HUAWEI/Desktop/28_may_submission/pneumonia_model_full.h5'
IMAGE_DIR = r'C:\Users\HUAWEI\Desktop\Chest images new'

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ['NORMAL', 'PNEUMONIA']  # Ensure correct order from training

# Preprocessing function (same as used in training)
def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 5)
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.stack([img]*3, axis=-1)  # Convert to 3-channel
    return np.expand_dims(img, axis=0)

# Inference loop
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(IMAGE_DIR, filename)
        image = preprocess_image(path)
        prediction = model.predict(image)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display result
        print(f"{filename}: {predicted_class} ({confidence:.2f}%)")

        # Optional: show image
        img_display = cv2.imread(path)
        img_display = cv2.resize(img_display, (400, 400))
        cv2.putText(img_display, f"{predicted_class} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if predicted_class == 'NORMAL' else (0, 0, 255), 2)
        cv2.imshow('Prediction', img_display)
        cv2.waitKey(0)

cv2.destroyAllWindows()


