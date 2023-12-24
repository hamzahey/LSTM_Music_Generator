
from tensorflow import keras
from preprocess import inputs, targets

MODEL_PATH = "model.h5"

if __name__ == "__main__":
    # Load the model
    model = keras.models.load_model(MODEL_PATH)

    # Evaluate the model on test sequences
    loss, accuracy = model.evaluate(inputs, targets)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")