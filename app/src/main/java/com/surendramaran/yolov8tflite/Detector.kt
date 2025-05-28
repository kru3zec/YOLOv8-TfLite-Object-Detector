package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.surendramaran.yolov8tflite.Box
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.exp
import kotlin.math.roundToInt

class Detector(
    private val context: Context,
    private val detectorListener: DetectorListener
) {

    private val modelPath = "best_float32.tflite"
    private val labelPath = "labels.txt"

    private var interpreter: Interpreter
    private var labels = listOf<String>()
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f))
        .add(CastOp(DataType.FLOAT32))
        .build()

    init {
        val options = Interpreter.Options().apply {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
            } else {
                setNumThreads(4)
            }
        }

        interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath), options)
        labels = FileUtil.loadLabels(context, labelPath)
        Log.d(TAG, "Loaded labels: ${labels.size}")
    }

    fun detect(frame: Bitmap) {
        val resized = Bitmap.createScaledBitmap(frame, 640, 640, true)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resized)
        val inputBuffer = imageProcessor.process(tensorImage).buffer

        val outputRaw = Array(1) { Array(104) { FloatArray(8400) } }

        val startTime = SystemClock.uptimeMillis()
        interpreter.run(inputBuffer, outputRaw)
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        val boxes = parseDetections(outputRaw[0])
        if (boxes.isEmpty()) {
            detectorListener.onEmptyDetect()
        } else {
            detectorListener.onDetect(boxes, inferenceTime)
        }
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x)))

    private fun parseDetections(data: Array<FloatArray>): List<Box> {
        val boxes = mutableListOf<Box>()

        for (i in data.indices) {
            val row = data[i]
            if (row.size < 6) continue

            val cx = row[0]
            val cy = row[1]
            val w = row[2]
            val h = row[3]
            val confidence = sigmoid(row[4])
            if (confidence < CONFIDENCE_THRESHOLD) continue

            val classScores = row.copyOfRange(5, row.size).map { sigmoid(it) }
            val maxIdx = classScores.indices.maxByOrNull { classScores[it] } ?: continue
            val score = classScores[maxIdx] * confidence
            if (score < CONFIDENCE_THRESHOLD) continue

            val scaledCx = cx / 640f
            val scaledCy = cy / 640f
            val scaledW = w / 640f
            val scaledH = h / 640f

            val x1 = ((scaledCx - scaledW / 2f) * 640).roundToInt()
            val y1 = ((scaledCy - scaledH / 2f) * 640).roundToInt()
            val x2 = ((scaledCx + scaledW / 2f) * 640).roundToInt()
            val y2 = ((scaledCy + scaledH / 2f) * 640).roundToInt()

            if (x1 < 0 || y1 < 0 || x2 > 640 || y2 > 640) continue
            if (maxIdx >= labels.size) continue

            boxes.add(
                Box(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    class_id = labels[maxIdx],
                    confidence = score,
                    timeDetected = null
                )
            )

            Log.d(TAG, "Detected: ${labels[maxIdx]} (${String.format("%.2f", score * 100)}%)")
        }

        Log.d(TAG, "Total valid detections: ${boxes.size}")
        return applyNMS(boxes).take(10)
    }

    private fun applyNMS(boxes: List<Box>): List<Box> {
        val selected = mutableListOf<Box>()
        val sorted = boxes.sortedByDescending { it.confidence }.toMutableList()

        while (sorted.isNotEmpty()) {
            val chosen = sorted.removeAt(0)
            selected.add(chosen)

            val iter = sorted.iterator()
            while (iter.hasNext()) {
                val next = iter.next()
                val iou = calculateIoU(chosen, next)
                if (iou > IOU_THRESHOLD) iter.remove()
            }
        }
        return selected
    }

    private fun calculateIoU(a: Box, b: Box): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)
        val inter = maxOf(0, x2 - x1) * maxOf(0, y2 - y1)
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter.toFloat() / union.toFloat()
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<Box>, inferenceTime: Long)
    }

    companion object {
        private const val TAG = "Detector"
        private const val CONFIDENCE_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.45f
    }
}
