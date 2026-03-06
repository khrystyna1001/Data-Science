import os
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam



# Force RGB format at the top level
tf.keras.backend.set_image_data_format('channels_last')

def main():
    DATASET_PATH = "DL_Project/Data/"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    SEED = 42

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    class_counts = {name: len(os.listdir(os.path.join(DATASET_PATH, name))) for name in class_names}
    total_images = sum(class_counts.values())

    class_weights = {
        i: total_images / (num_classes * class_counts[name])
        for i, name in enumerate(class_names)
    }
    print("Class weights:", class_weights)

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.map(lambda x, y: (preprocess_input(x), y))
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Model (EfficientNet)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    weights_path = tf.keras.utils.get_file(
        "efficientnetb0_notop.h5",
        "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
        cache_subdir="models"
    )
    base_model.load_weights(weights_path)
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout
    x = tf.keras.layers.Dense(256, activation='relu')(x)  # Larger dense layer
    x = tf.keras.layers.Dropout(0.2)(x)  # More dropout
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add learning rate scheduling
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
    )

    history = model.fit(
        train_dataset,
        epochs=30,  # More epochs
        validation_data=validation_dataset,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
            lr_schedule
        ]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()

    if not os.path.exists("DL_Project/saved_models"):
        os.makedirs("DL_Project/saved_models")
    model.save("DL_Project/saved_models/efficientnet_classifier.keras")
    
    print(f"Final model saved. Training completed.")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()