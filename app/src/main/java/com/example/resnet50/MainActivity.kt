package com.example.resnet50

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.resnet50.ml.Resnet50 // This will be generated from 'lite_model.tflite'
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var btnPickImage: Button
    private lateinit var btnClassify: Button
    private lateinit var tvResult: TextView

    private var selectedBitmap: Bitmap? = null
    private lateinit var model: Resnet50
    private lateinit var labels: List<String>

    private val inputWidth = 224
    private val inputHeight = 224

    // Normalization parameters for ImageNet (PyTorch-style)
    // Assumes input to NormalizeOp is already scaled to [0,1] by TensorImage.load(bitmap)
    // These are applied to RGB channels.
    private val NORM_MEAN_RGB = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val NORM_STD_RGB = floatArrayOf(0.229f, 0.224f, 0.225f)


    private val imagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                try {
                    selectedBitmap = uriToBitmap(it)
                    if (selectedBitmap != null) {
                        imageView.setImageBitmap(selectedBitmap)
                        tvResult.text = "Image selected. Click Classify."
                    } else {
                        tvResult.text = "Could not load image."
                        Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error loading image URI: ${e.message}", e)
                    Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
                    tvResult.text = "Failed to load image."
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        imageView = findViewById(R.id.imageView)
        btnPickImage = findViewById(R.id.btnPickImage)
        btnClassify = findViewById(R.id.btnClassify)
        tvResult = findViewById(R.id.tvResult)

        try {
            model = Resnet50.newInstance(this) // From lite_model.tflite in src/main/ml
            Log.i(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "CRITICAL: Error loading model: ${e.message}", e)
            Toast.makeText(this, "CRITICAL: Error loading model. App will close.", Toast.LENGTH_LONG).show()
            tvResult.text = "Error loading model: ${e.localizedMessage}"
            finish()
            return
        }

        try {
            labels = application.assets.open("labels.txt").bufferedReader().readLines()
            if (labels.isEmpty()){
                Log.e(TAG, "Labels file is empty or could not be read.")
                Toast.makeText(this, "CRITICAL: Labels file is empty. App will close.", Toast.LENGTH_LONG).show()
                tvResult.text = "Error: Labels file empty."
                finish()
                return
            }
            Log.i(TAG, "Labels loaded: ${labels.size} labels")
        } catch (e: IOException) {
            Log.e(TAG, "CRITICAL: Error loading labels: ${e.message}", e)
            Toast.makeText(this, "CRITICAL: Error loading labels. App will close.", Toast.LENGTH_LONG).show()
            tvResult.text = "Error loading labels: ${e.localizedMessage}"
            finish()
            return
        }

        btnPickImage.setOnClickListener {
            imagePickerLauncher.launch("image/*")
        }

        btnClassify.setOnClickListener {
            if (selectedBitmap == null) {
                Toast.makeText(this, "Please pick an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            classifyImage(selectedBitmap!!)
        }
    }

    private fun uriToBitmap(selectedFileUri: Uri): Bitmap? {
        return try {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, selectedFileUri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, selectedFileUri)
            }
            if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.copy(Bitmap.Config.ARGB_8888, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting URI to Bitmap: ${e.message}", e)
            null
        }
    }


    private fun classifyImage(bitmap: Bitmap) {
        try {
            // 1. Create TensorImage from Bitmap.
            // For a Float32 model, this loads the Bitmap, extracts RGB, and scales pixel values to [0,1].
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap) // Bitmap is ARGB_8888

            // 2. Create ImageProcessor for resizing and PyTorch-style normalization.
            // The new model (resnet50_torch_preproc.tflite) expects this.
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
//                .add(NormalizeOp(NORM_MEAN_RGB, NORM_STD_RGB)) // PyTorch-style mean/std
                .build()

            tensorImage = imageProcessor.process(tensorImage)

            // 3. Create input TensorBuffer.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, inputHeight, inputWidth, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // 4. Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            // 5. Process the output
            val probabilities = outputFeature0.floatArray

            if (probabilities.isEmpty()) {
                Log.e(TAG, "Model returned empty probabilities array.")
                tvResult.text = "Classification failed: Empty result."
                return
            }

            var maxProb = 0f
            var maxIdx = -1
            Log.d(TAG, "Output Probabilities (first 10): ${probabilities.take(10).joinToString()}") // For debugging
            for (i in probabilities.indices) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i]
                    maxIdx = i
                }
            }

            Log.d(TAG, "Max Index from model: $maxIdx, Max Probability: $maxProb")

            if (maxIdx != -1 && maxIdx < labels.size) {
                val predictedLabel = labels[maxIdx]
                Log.d(TAG, "Predicted Label from labels.txt: $predictedLabel")
                val resultText = "Prediction: $predictedLabel\nConfidence: %.2f%%".format(maxProb * 100)
                tvResult.text = resultText
                Log.i(TAG, resultText)
            } else {
                tvResult.text = "Could not classify or invalid label index (Idx: $maxIdx, Labels: ${labels.size}, Probs: ${probabilities.size})."
                Log.e(TAG, "Invalid prediction index: $maxIdx, Labels size: ${labels.size}, Probabilities size: ${probabilities.size}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during classification: ${e.message}", e)
            tvResult.text = "Classification Error: ${e.localizedMessage}"
            Toast.makeText(this, "Classification failed: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) {
            model.close()
            Log.i(TAG, "Model closed.")
        }
    }

    companion object {
        private const val TAG = "MainActivityResnet"
    }
}