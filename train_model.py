import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# We'll create synthetic operator data since EMNIST needs special loading
CLASSES = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']
NUM_CLASSES = len(CLASSES)

def create_operator_samples(num_samples=5000):
    """Create synthetic handwritten-style operator images"""
    import cv2
    images = []
    labels = []
    
    operators = {10: '+', 11: '-', 12: '*', 13: '/'}
    
    for label, op in operators.items():
        for _ in range(num_samples):
            img = np.zeros((28, 28), dtype=np.float32)
            # Random thickness and position for variety
            thickness = np.random.randint(1, 3)
            offset_x = np.random.randint(-3, 3)
            offset_y = np.random.randint(-3, 3)
            
            if op == '+':
                cv2.line(img, (14+offset_x, 7+offset_y), (14+offset_x, 21+offset_y), 1.0, thickness)
                cv2.line(img, (7+offset_x, 14+offset_y), (21+offset_x, 14+offset_y), 1.0, thickness)
            elif op == '-':
                cv2.line(img, (6+offset_x, 14+offset_y), (22+offset_x, 14+offset_y), 1.0, thickness)
            elif op == '*':
                cv2.line(img, (8+offset_x, 8+offset_y), (20+offset_x, 20+offset_y), 1.0, thickness)
                cv2.line(img, (20+offset_x, 8+offset_y), (8+offset_x, 20+offset_y), 1.0, thickness)
            elif op == '/':
                cv2.line(img, (20+offset_x, 6+offset_y), (8+offset_x, 22+offset_y), 1.0, thickness)
            
            # Add noise
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def train():
    print("Loading MNIST digits...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print("Creating operator training data...")
    op_images, op_labels = create_operator_samples(6000)
    op_images_test, op_labels_test = create_operator_samples(1000)

    # Combine digits + operators
    x_train_full = np.concatenate([x_train, op_images], axis=0)
    y_train_full = np.concatenate([y_train, op_labels], axis=0)
    x_test_full = np.concatenate([x_test, op_images_test], axis=0)
    y_test_full = np.concatenate([y_test, op_labels_test], axis=0)

    # Expand dims for CNN
    x_train_full = np.expand_dims(x_train_full, -1)
    x_test_full = np.expand_dims(x_test_full, -1)

    # Shuffle
    idx = np.random.permutation(len(x_train_full))
    x_train_full, y_train_full = x_train_full[idx], y_train_full[idx]

    print(f"Total training samples: {len(x_train_full)}")
    print(f"Classes: {NUM_CLASSES} ({CLASSES})")

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\nTraining model...")
    model.fit(
        x_train_full, y_train_full,
        epochs=15,
        batch_size=128,
        validation_data=(x_test_full, y_test_full)
    )

    loss, accuracy = model.evaluate(x_test_full, y_test_full)
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

    model.save('model/math_model.keras')
    print("Model saved! ✅")

    # Save class labels
    np.save('model/classes.npy', np.array(CLASSES))
    print("Classes saved! ✅")

if __name__ == '__main__':
    train()

