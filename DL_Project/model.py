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
    DATASET_PATH = "Data/"
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
        shuffle=False
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False
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

    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocess_input(x), y))

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
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        epochs=15,
        validation_data=validation_dataset,
        class_weight=class_weights
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

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    model.save("saved_models/efficientnet_classifier.keras")

    # Hyper Parameter Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(hp):
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        base_model.trainable = False

        model = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(hp.Float("rotation_factor", min_value=0.05, max_value=0.3, step=0.05)),
            RandomZoom(hp.Float("zoom_factor", min_value=0.05, max_value=0.3, step=0.05)),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)),
            Dense(
                hp.Int("dense_units", min_value=64, max_value=512, step=64),
                activation='relu'
            ),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Setup Random Search Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory='kt_dir',
        project_name='efficientnet_tune',
        overwrite=True
    )

    # Optional: Early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Run the search
    tuner.search(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[early_stopping],
        class_weight=class_weights
    )

    # Results
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(best_hps.values)

    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate on validation
    val_loss, val_acc = best_model.evaluate(validation_dataset)
    print(f"Best validation accuracy: {val_acc:.4f}")
    best_model.save('saved_models/efficientnet_finetuned.keras')

if __name__ == "__main__":
    main()