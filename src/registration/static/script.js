// Core DOM elements
const uploadCanvas = document.getElementById('uploadCanvas');
const ctx = uploadCanvas.getContext('2d');
const video = document.getElementById('webcamVideo');
const captureBtn = document.getElementById('captureBtn');
const resetCaptureBtn = document.getElementById('resetCaptureBtn');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamCtx = webcamCanvas.getContext('2d');

const angleHint = document.getElementById('angle-hint');
const captureCountSpan = document.getElementById('capture-count');
const uploadName = document.getElementById('uploadName');
const webcamName = document.getElementById('webcamName');
const saveUploadBtn = document.getElementById('saveUploadBtn');
const saveWebcamBtn = document.getElementById('saveWebcamBtn');
const imageInput = document.getElementById('imageInput');
const uploadStatus = document.getElementById('upload-status');
const webcamStatus = document.getElementById('webcam-status');
const capturedPreview = document.getElementById('captured-preview');

// --- Webcam workflow state ---
let stream = null;
let capturedImagesArray = []; // up to 5 base64 image strings
const faceAngles = [
    "Look straight ahead",
    "Turn head to the Left",
    "Turn head to the Right",
    "Tilt head Up",
    "Tilt head Down"
];
let currentAngleIndex = 0; // which angle is currently being requested

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const tab = this.dataset.tab;
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');

        if (tab === 'webcam') {
            startWebcam();
            resetWebcamWorkflow(); // restart the multi-angle capture workflow
        } else {
            stopWebcam();
        }
    });
});

// Open/close the webcam stream
function startWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then(s => {
                stream = s;
                video.srcObject = s;
                video.play();
            })
            .catch(err => {
                webcamStatus.textContent = 'Could not access the camera: ' + err.message;
                webcamStatus.style.color = '#dc3545';
            });
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
    }
}

// Refresh the multi-angle workflow UI to match current state
function updateWebcamWorkflowUI() {
    captureCountSpan.textContent = `Captured: ${capturedImagesArray.length}/5`;

    if (currentAngleIndex < faceAngles.length) {
        angleHint.textContent = `Requested: ${faceAngles[currentAngleIndex]}`;
        angleHint.style.color = "#007bff";
        captureBtn.disabled = false;
    } else {
        angleHint.textContent = "🎯 All 5 angles captured!";
        angleHint.style.color = "#28a745";
        captureBtn.disabled = true; // lock further capture once the goal is met
    }
}

// Reset the workflow back to its initial state
function resetWebcamWorkflow() {
    capturedImagesArray = [];
    currentAngleIndex = 0;
    capturedPreview.innerHTML = ''; // clear the preview grid
    webcamStatus.textContent = 'Center your face on screen and click the button to capture the first angle.';
    webcamStatus.style.color = '#555';
    updateWebcamWorkflowUI();
}

// "Start over" button: clears all previously captured images
resetCaptureBtn.addEventListener('click', function() {
    if (capturedImagesArray.length === 0) {
        webcamStatus.textContent = "Nothing captured yet, you can go ahead and capture.";
        return;
    }
    if (confirm("This will clear all the angles captured so far. Start over?")) {
        resetWebcamWorkflow();
    }
});

// Capture the current frame for the current requested angle
captureBtn.addEventListener('click', function() {
    if (currentAngleIndex >= faceAngles.length || !stream) return;

    // Size the hidden canvas to match the actual video stream
    webcamCanvas.width = video.videoWidth || 640;
    webcamCanvas.height = video.videoHeight || 480;
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const dataUrl = webcamCanvas.toDataURL('image/jpeg');

    webcamStatus.textContent = "⏳ Validating face for this angle...";
    webcamStatus.style.color = "#007bff";

    // Send the single frame to the server to confirm it contains a face
    fetch('/get_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            // Face found -> queue the image
            capturedImagesArray.push(dataUrl);

            // Build the preview card with the angle label
            const cardWrapper = document.createElement('div');
            cardWrapper.style.position = 'relative';
            cardWrapper.style.border = '2px solid #28a745';
            cardWrapper.style.borderRadius = '5px';
            cardWrapper.style.overflow = 'hidden';

            const previewImg = document.createElement('img');
            previewImg.src = dataUrl;
            previewImg.style.width = '120px';
            previewImg.style.height = '90px';
            previewImg.style.objectFit = 'cover';
            previewImg.style.display = 'block';

            const angleLabel = document.createElement('div');
            angleLabel.textContent = faceAngles[currentAngleIndex];
            angleLabel.style.position = 'absolute';
            angleLabel.style.bottom = '0';
            angleLabel.style.width = '100%';
            angleLabel.style.background = 'rgba(0, 123, 255, 0.85)';
            angleLabel.style.color = 'white';
            angleLabel.style.fontSize = '10px';
            angleLabel.style.textAlign = 'center';
            angleLabel.style.padding = '2px 0';

            cardWrapper.appendChild(previewImg);
            cardWrapper.appendChild(angleLabel);
            capturedPreview.appendChild(cardWrapper);

            webcamStatus.textContent = `✅ Captured angle: ${faceAngles[currentAngleIndex]}`;
            webcamStatus.style.color = '#28a745';

            // Move to the next angle; each angle is captured exactly once
            currentAngleIndex++;
            updateWebcamWorkflowUI();
        } else {
            // No face found -> keep the same angle and ask to retry
            webcamStatus.textContent = `❌ Error: ${data.message}. Please re-adjust and capture: [ ${faceAngles[currentAngleIndex]} ]`;
            webcamStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        webcamStatus.textContent = 'Lost connection to the API server: ' + err.message;
        webcamStatus.style.color = '#dc3545';
    });
});

// Send the 5 captured images to the backend for embedding + storage
saveWebcamBtn.addEventListener('click', function() {
    const name = webcamName.value.trim();
    if (!name) {
        webcamStatus.textContent = '⚠️ Please enter a name before saving!';
        webcamStatus.style.color = '#dc3545';
        return;
    }
    if (capturedImagesArray.length < 5) {
        webcamStatus.textContent = `❌ You haven't captured all 5 required angles yet (currently: ${capturedImagesArray.length}/5).`;
        webcamStatus.style.color = '#dc3545';
        return;
    }

    webcamStatus.textContent = "⏳ Server is computing embeddings...";
    webcamStatus.style.color = "#007bff";

    fetch('/register_final', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, images: capturedImagesArray })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            webcamStatus.textContent = `🎉 Saved successfully! [ ${name} ]'s dataset is ready.`;
            webcamStatus.style.color = '#28a745';
            webcamName.value = ''; // clear the name field
            resetWebcamWorkflow();  // reset for the next registration
        } else {
            webcamStatus.textContent = '❌ Server error: ' + data.message;
            webcamStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        webcamStatus.textContent = 'Connection lost: ' + err.message;
        webcamStatus.style.color = '#dc3545';
    });
});

// --- Static image upload tab ---
imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
        const img = new Image();
        img.onload = function() {
            uploadCanvas.width = img.width;
            uploadCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const dataUrl = uploadCanvas.toDataURL('image/jpeg');
            fetch('/get_landmarks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    // Draw the extracted landmarks on top of the canvas
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    data.landmarks.forEach(pt => {
                        ctx.beginPath();
                        ctx.arc(pt[0], pt[1], 3, 0, 2 * Math.PI);
                        ctx.fillStyle = 'red';
                        ctx.fill();
                    });
                    uploadStatus.textContent = '✅ Valid face detected in the image';
                    uploadStatus.style.color = '#28a745';
                } else {
                    uploadStatus.textContent = '❌ No face found: ' + data.message;
                    uploadStatus.style.color = '#dc3545';
                }
            })
            .catch(err => {
                uploadStatus.textContent = 'Error analyzing image: ' + err.message;
            });
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
});

saveUploadBtn.addEventListener('click', function() {
    const name = uploadName.value.trim();
    if (!name) {
        uploadStatus.textContent = '⚠️ Please enter the person\'s name!';
        uploadStatus.style.color = '#dc3545';
        return;
    }
    const dataUrl = uploadCanvas.toDataURL('image/jpeg');
    fetch('/register_from_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl, name: name })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            uploadStatus.textContent = '✅ ' + data.message;
            uploadStatus.style.color = '#28a745';
            uploadName.value = '';
        } else {
            uploadStatus.textContent = '❌ Failed: ' + data.message;
            uploadStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        uploadStatus.textContent = 'API error: ' + err.message;
    });
});
