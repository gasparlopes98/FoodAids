package com.example.easychatgpt

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private var recyclerView: RecyclerView? = null
    private var welcomeTextView: TextView? = null
    private var messageEditText: EditText? = null
    private var sendButton: ImageButton? = null
    private var messageList: MutableList<Message>? = null
    private var messageAdapter: MessageAdapter? = null
    private var joaneView: ImageView? = null
    private var cameraButton:ImageView? = null
    private var client = OkHttpClient()
    private lateinit var currentPhotoPath: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        messageList = ArrayList()
        recyclerView = findViewById(R.id.recycler_view)
        welcomeTextView = findViewById(R.id.welcome_text)
        messageEditText = findViewById(R.id.message_edit_text)
        sendButton = findViewById(R.id.send_btn)
        joaneView = findViewById(R.id.imageView)
        cameraButton = findViewById(R.id.imageButton2)

        //setup recycler view
        messageAdapter = MessageAdapter(messageList as ArrayList<Message>)
        recyclerView?.adapter = messageAdapter
        val llm = LinearLayoutManager(this)
        llm.stackFromEnd = true
        recyclerView?.layoutManager = llm
        sendButton?.setOnClickListener { v: View? ->
            val question = messageEditText?.text.toString().trim { it <= ' ' }
            addToChat(question, Message.SENT_BY_ME)
            messageEditText?.setText("")
            callAPI(question)
            welcomeTextView?.visibility = View.GONE
            joaneView?.visibility = View.GONE
        }

        cameraButton?.setOnClickListener {
            dispatchTakePictureIntent()
        }
    }

    fun addToChat(message: String?, sentBy: String?) {
        if (message != null && sentBy != null){
            runOnUiThread {
                messageList!!.add(Message(message, sentBy))
                messageAdapter!!.notifyDataSetChanged()
                recyclerView!!.smoothScrollToPosition(messageAdapter!!.itemCount)
            }
        }
    }

    fun addResponse(response: String?) {
        messageList!!.removeAt(messageList!!.size - 1)
        addToChat(response, Message.SENT_BY_BOT)
    }

    fun callAPI(question: String?) {
        //okhttp
        messageList!!.add(Message("Typing... ", Message.SENT_BY_BOT))
        val jsonBody = JSONObject()
        try {
            jsonBody.put("model", "text-davinci-003")
            jsonBody.put("prompt", question)
            jsonBody.put("max_tokens", 4000)
            jsonBody.put("temperature", 0)
        } catch (e: JSONException) {
            e.printStackTrace()
        }
        val body: RequestBody = jsonBody.toString().toRequestBody(JSON)
        val request: Request = Request.Builder()
            .url("https://api.openai.com/v1/completions")
            .header("Authorization", "Bearer YOUR_API_KEY")
            .post(body)
            .build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                addResponse("Failed to load response due to " + e.message)
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    var jsonObject: JSONObject? = null
                    try {
                        jsonObject = JSONObject(response.body!!.string())
                        val jsonArray = jsonObject.getJSONArray("choices")
                        val result = jsonArray.getJSONObject(0).getString("text")
                        addResponse(result.trim { it <= ' ' })
                    } catch (e: JSONException) {
                        e.printStackTrace()
                    }
                } else {
                    addResponse("Failed to load response due to " + response.body.toString())
                }
            }
        })
    }

    /**
     * ML Kit Object Detection function. We'll add ML Kit code here in the codelab.
     */
    private fun runObjectDetection(bitmap: Bitmap) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val localModel = LocalModel.Builder()
            .setAssetFilePath("model.tflite").build()
        val options = CustomObjectDetectorOptions.Builder(localModel)
            .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
            .enableClassification()
            .enableMultipleObjects()
            .setClassificationConfidenceThreshold(0.5f)
            .setMaxPerObjectLabelCount(3)
            .build()
        val objectDetector = ObjectDetection.getClient(options)

        objectDetector.process(image).addOnSuccessListener { results ->
            debugPrint(results)

            // Parse ML Kit's DetectedObject and create corresponding visualization data
            results.map {
                var text:String

                // We will show the top confident detection result if it exist
                if (it.labels.isNotEmpty()) {
                    val firstLabel = it.labels.first()
                    text = "${firstLabel.text}, ${firstLabel.confidence.times(100).toInt()}%"
                    println(text)
                    text = "I just found " + firstLabel.text
                    addToChat(text,Message.SENT_BY_ME)
                }
            }

        }.addOnFailureListener {
            Log.e(TAG, it.message.toString())
        }
    }

    /**
     * Print out the object detection result to Logcat.
     */
    private fun debugPrint(detectedObjects: List<DetectedObject>) {
        detectedObjects.forEachIndexed { index, detectedObject ->
            val box = detectedObject.boundingBox

            Log.d(TAG, "Detected object: $index")
            Log.d(TAG, " trackingId: ${detectedObject.trackingId}")
            Log.d(TAG, " boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")
            detectedObject.labels.forEach {
                Log.d(TAG, " categories: ${it.text}")
                Log.d(TAG, " confidence: ${it.confidence}")
            }
        }
    }

    //CAMERA STUFF

    /**
     * Open a camera app to take photo.
     */
    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (e: IOException) {
                    Log.e(TAG, e.message.toString())
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.example.easychatgpt.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
            resultCode == Activity.RESULT_OK
        ) {
            runObjectDetection(getCapturedImage())
        }
    }

    private fun getCapturedImage(): Bitmap {
        // Get the dimensions of the View
        val targetW: Int = 250
        val targetH: Int = 250

        val bmOptions = BitmapFactory.Options().apply {
            // Get the dimensions of the bitmap
            inJustDecodeBounds = true

            BitmapFactory.decodeFile(currentPhotoPath, this)

            val photoW: Int = outWidth
            val photoH: Int = outHeight

            // Determine how much to scale down the image
            val scaleFactor: Int = max(1, min(photoW / targetW, photoH / targetH))

            // Decode the image file into a Bitmap sized to fill the View
            inJustDecodeBounds = false
            inSampleSize = scaleFactor
            inMutable = true
        }
        val exifInterface = ExifInterface(currentPhotoPath)
        val orientation = exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                rotateImage(bitmap, 90f)
            }
            ExifInterface.ORIENTATION_ROTATE_180 -> {
                rotateImage(bitmap, 180f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> {
                rotateImage(bitmap, 270f)
            }
            else -> {
                bitmap
            }
        }
    }

    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    /**
     * Create a file to pass to a camera app for storing captured image.
     */
    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    companion object {
        const val TAG = "MLKit-ODT"
        val JSON: MediaType = "application/json; charset=utf-8".toMediaType()
        const val REQUEST_IMAGE_CAPTURE: Int = 1

    }
}