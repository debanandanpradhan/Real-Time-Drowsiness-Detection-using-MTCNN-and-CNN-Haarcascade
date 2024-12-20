from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

def train_eye_model():
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_data = data_gen.flow_from_directory(
        r"C:\pythonProject\pythonProject\train",
        target_size=(24, 24),
        color_mode="grayscale",
        class_mode="binary",
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile with accuracy as a metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train and capture the history
    history = model.fit(train_data, epochs=10)

    # Save the model after training
    model.save("eye_model.h5")

    # Plot training accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.show()

train_eye_model()
