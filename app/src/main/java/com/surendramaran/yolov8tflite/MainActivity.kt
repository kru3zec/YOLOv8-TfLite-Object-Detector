package com.surendramaran.yolov8tflite

import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var detector: Detector
    private lateinit var overlayView: OverlayView
    private lateinit var signLabel: TextView
    private lateinit var previewView: PreviewView
    private lateinit var returnButton: Button
    private lateinit var signName: TextView
    private lateinit var signDescription: TextView
    private lateinit var signBoxes: TextView
    private lateinit var signImage: ImageView
    private lateinit var cameraExecutor: ExecutorService

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.fragment_camera)

        overlayView = findViewById(R.id.overlayView)
        signLabel = findViewById(R.id.detectedSignText)
        previewView = findViewById(R.id.camera_preview)
        returnButton = findViewById(R.id.returnButton)
        signName = findViewById(R.id.signName)
        signDescription = findViewById(R.id.signDescription)
        signBoxes = findViewById(R.id.signBoxes)
        signImage = findViewById(R.id.signImage)

        cameraExecutor = Executors.newSingleThreadExecutor()

        detector = Detector(this, this)

        startCamera()

        returnButton.setOnClickListener {
            finish()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        val bitmap = imageProxyToBitmap(imageProxy)
                        detector.detect(bitmap)
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e("Camera", "Binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(imageProxy: ImageProxy): android.graphics.Bitmap {
        val image: Image = imageProxy.image ?: return BitmapFactory.decodeResource(resources, android.R.drawable.ic_delete)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val yuv = out.toByteArray()
        return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            overlayView.setResults(emptyList())
            overlayView.invalidate()
            signLabel.text = "Brak wykrycia"
            signName.text = ""
            signDescription.text = ""
            signBoxes.text = ""
            signImage.visibility = ImageView.GONE
        }
    }

    override fun onDetect(boundingBoxes: List<Box>, inferenceTime: Long) {
        runOnUiThread {
            overlayView.setResults(boundingBoxes)
            overlayView.invalidate()

            val topBox = boundingBoxes.maxByOrNull { it.confidence }
            signLabel.text = topBox?.class_id ?: "Brak etykiety"

            topBox?.let { box ->
                signName.text = box.class_id
                signDescription.text = "Wykryto z pewnością ${(box.confidence * 100).toInt()}%"
                signBoxes.text = "(%d, %d, %d, %d)".format(box.x1, box.y1, box.x2, box.y2)
                signImage.visibility = ImageView.VISIBLE
                // Możesz tu dodać dynamiczne przypisywanie grafiki do signImage jeśli masz zasoby
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
