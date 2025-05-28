# ResNet50 Image Classification Android App

This Android application demonstrates image classification using a pre-trained ResNet50 model, converted to TensorFlow Lite format. Users can pick an image from their device, and the app will classify it, displaying the predicted class and confidence.

## Features

*   Pick images from the device gallery.
*   Perform image classification using an optimized ResNet50 TFLite model.
*   Display the top prediction and its confidence score.

## Screenshots

**(You can add 2-3 screenshots here showcasing the app in action)**

1.  **Main Screen / Image Picker:**
    ![image](https://github.com/user-attachments/assets/5dc87dba-7fce-4b88-8d44-3c338f73f204)

2.  **Image Displayed and Classification Result:**
    ![image](https://github.com/user-attachments/assets/60572d02-3141-43fd-94a3-fe756a8eba4a)


## Model

The app uses a ResNet50 model, originally pre-trained on the ImageNet dataset using Keras. This Keras model has been:
1.  Modified to include image preprocessing steps (scaling pixel values and applying PyTorch-style mean/standard deviation normalization) directly within the model graph.
2.  Converted to TensorFlow Lite format (`.tflite`).
3.  Optimized using default TFLite optimizations (typically dynamic range quantization for weights) to reduce its size.

The specific preprocessing baked into the TFLite model is:
*   Input image pixels (expected as Float32 in the range `[0, 255]`) are first scaled to `[0, 1]`.
*   Then, standard ImageNet (PyTorch-style) mean `[0.485, 0.456, 0.406]` is subtracted, and the result is divided by the standard deviation `[0.229, 0.224, 0.225]`.

The TFLite model file (`lite_model.tflite`) is located in the `app/src/main/ml/` directory.
The class labels (`labels.txt`) are in `app/src/main/assets/`.

## App Architecture

*   **`MainActivity.kt`**: Handles UI interactions, image picking, TFLite model loading, image preprocessing (resizing and scaling to the format expected by the TFLite model's input tensor), running inference, and displaying results.
*   **`activity_main.xml`**: Defines the layout with an `ImageView` to display the selected image, `Button`s for picking and classifying, and a `TextView` for results.
*   **TensorFlow Lite Support Library**: Used for image manipulation (`TensorImage`, `ImageProcessor`) and interacting with the TFLite model.
*   **ML Model Binding**: Used to generate a wrapper class for easy interaction with the TFLite model.

## Preprocessing in Android

The Android application performs the following preprocessing steps before feeding the image to the TFLite model:
1.  The selected image `Bitmap` is loaded.
2.  It is resized to 224x224 pixels.
3.  The pixel values (RGB) are converted to `Float32` and scaled to the range `[0, 255]`. This is the format directly expected by the input tensor of our optimized TFLite model (as the model itself contains the `/255.0` scaling and subsequent normalization operations).

## Setup and Build

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Loki-Silvres/resnet_app.git
    cd https://github.com/Loki-Silvres/resnet_app.git
    ```
2.  **Open in Android Studio:** Open the project in the latest stable version of Android Studio.
3.  **Place Model:**
    *   Ensure the TFLite model file (e.g., `lite_model.tflite`) is in `app/src/main/ml/`. [Kaggle](https://www.kaggle.com/code/lokisilvres/convert-h5-to-tflite)
4.  **Build:** Android Studio should automatically sync Gradle dependencies. Then, build the project (Build > Make Project).
5.  **Run:** Run the app on an Android emulator or a physical device.

## Dependencies

*   AndroidX Libraries (Core, AppCompat, ConstraintLayout, Activity)
*   Material Components
*   TensorFlow Lite (`org.tensorflow:tensorflow-lite`)
*   TensorFlow Lite Support Library (`org.tensorflow:tensorflow-lite-support`)
*   TensorFlow Lite Metadata (`org.tensorflow:tensorflow-lite-metadata`)
*   (Optionally) TensorFlow Lite Select TF Ops, if the model required it.

## Potential Improvements / Future Work

*   Add support for taking pictures with the camera.
*   Implement real-time classification from a camera preview.
*   Allow selection of different TFLite models.
*   Display top N predictions instead of just the top one.
*   Improve UI/UX.
