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
let capturedImagesArray = [];
const faceAngles = [
    "Look straight ahead",
    "Turn head to the Left",
    "Turn head to the Right",
    "Tilt head Up",
    "Tilt head Down"
];
let currentAngleIndex = 0;

// --- Body Registration Workflow State ---
let bodyImagesArray = [];
const bodyAngles = ["Phía trước (Mặt hướng vào camera)", "Phía sau (Quay lưng lại)"];
let currentBodyAngleIndex = 0;
let currentRegisteredName = "";

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
            resetWebcamWorkflow();
        } else {
            stopWebcam();
        }
    });
});

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

// ==========================================
// FACE WORKFLOW
// ==========================================
function updateWebcamWorkflowUI() {
    captureCountSpan.textContent = `Captured: ${capturedImagesArray.length}/5`;

    if (currentAngleIndex < faceAngles.length) {
        angleHint.textContent = `Requested: ${faceAngles[currentAngleIndex]}`;
        angleHint.style.color = "#007bff";
        captureBtn.disabled = false;
    } else {
        angleHint.textContent = "🎯 All 5 angles captured!";
        angleHint.style.color = "#28a745";
        captureBtn.disabled = true;
    }
}

function resetWebcamWorkflow() {
    capturedImagesArray = [];
    currentAngleIndex = 0;
    capturedPreview.innerHTML = '';
    webcamStatus.textContent = 'Center your face and click the button to capture the first angle.';
    webcamStatus.style.color = '#555';
    updateWebcamWorkflowUI();
    
    // Hiện phần Face, ẩn phần Body
    document.getElementById('face-registration-step').style.display = 'block';
    document.getElementById('webcam-name-input').style.display = 'flex';
    document.getElementById('body-registration-panel').style.display = 'none';
}

resetCaptureBtn.addEventListener('click', function() {
    if (capturedImagesArray.length === 0) return;
    if (confirm("This will clear all the angles captured so far. Start over?")) {
        resetWebcamWorkflow();
    }
});

captureBtn.addEventListener('click', function() {
    if (currentAngleIndex >= faceAngles.length || !stream) return;

    webcamCanvas.width = video.videoWidth || 640;
    webcamCanvas.height = video.videoHeight || 480;
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const dataUrl = webcamCanvas.toDataURL('image/jpeg');

    webcamStatus.textContent = "⏳ Validating face...";
    webcamStatus.style.color = "#007bff";

    fetch('/get_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            capturedImagesArray.push(dataUrl);
            const cardWrapper = document.createElement('div');
            cardWrapper.style.cssText = 'position: relative; border: 2px solid #28a745; border-radius: 5px; overflow: hidden; width: 120px; height: 90px;';
            const previewImg = document.createElement('img');
            previewImg.src = dataUrl;
            previewImg.style.cssText = 'width: 120px; height: 90px; object-fit: cover; display: block;';
            const angleLabel = document.createElement('div');
            angleLabel.textContent = faceAngles[currentAngleIndex];
            angleLabel.style.cssText = 'position: absolute; bottom: 0; width: 100%; background: rgba(0, 123, 255, 0.85); color: white; font-size: 10px; text-align: center; padding: 2px 0;';
            cardWrapper.appendChild(previewImg);
            cardWrapper.appendChild(angleLabel);
            capturedPreview.appendChild(cardWrapper);

            webcamStatus.textContent = `✅ Captured: ${faceAngles[currentAngleIndex]}`;
            webcamStatus.style.color = '#28a745';
            currentAngleIndex++;
            updateWebcamWorkflowUI();
        } else {
            webcamStatus.textContent = `❌ Error: ${data.message}. Please capture: [ ${faceAngles[currentAngleIndex]} ]`;
            webcamStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        webcamStatus.textContent = 'API error: ' + err.message;
        webcamStatus.style.color = '#dc3545';
    });
});

// SAVE FACE -> TRANSITION TO BODY
saveWebcamBtn.addEventListener('click', function() {
    const name = webcamName.value.trim();
    if (!name) {
        webcamStatus.textContent = '⚠️ Please enter a name before saving!';
        webcamStatus.style.color = '#dc3545';
        return;
    }
    if (capturedImagesArray.length < 5) {
        webcamStatus.textContent = `❌ You haven't captured all 5 angles yet (${capturedImagesArray.length}/5).`;
        webcamStatus.style.color = '#dc3545';
        return;
    }

    webcamStatus.textContent = "⏳ Saving face data...";
    webcamStatus.style.color = "#007bff";

    fetch('/register_final', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, images: capturedImagesArray })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            currentRegisteredName = name; // Lưu tên để dùng cho bước body
            webcamStatus.textContent = `🎉 Face data saved! Preparing body registration...`;
            webcamStatus.style.color = '#28a745';
            
            // Chỉ ẩn khối điều khiển Face, KHÔNG ẩn video nữa
            setTimeout(() => {
                document.getElementById('face-registration-step').style.display = 'none';
                document.getElementById('webcam-name-input').style.display = 'none';
                document.getElementById('body-registration-panel').style.display = 'block';
                resetBodyWorkflow();
            }, 1500);
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


// ==========================================
// BODY WORKFLOW
// ==========================================
function updateBodyWorkflowUI() {
    document.getElementById('body-count').textContent = `Đã chụp: ${bodyImagesArray.length}/2`;
    if (currentBodyAngleIndex < bodyAngles.length) {
        document.getElementById('body-hint').textContent = `Yêu cầu: ${bodyAngles[currentBodyAngleIndex]}`;
        document.getElementById('body-hint').style.color = "#007bff";
        document.getElementById('captureBodyFrontBtn').disabled = currentBodyAngleIndex !== 0;
        document.getElementById('captureBodyBackBtn').disabled = currentBodyAngleIndex !== 1;
    } else {
        document.getElementById('body-hint').textContent = "🎯 Đã chụp đủ 2 hướng!";
        document.getElementById('body-hint').style.color = "#28a745";
        document.getElementById('captureBodyFrontBtn').disabled = true;
        document.getElementById('captureBodyBackBtn').disabled = true;
        document.getElementById('saveBodyBtn').disabled = false;
    }
}

function resetBodyWorkflow() {
    bodyImagesArray = [];
    currentBodyAngleIndex = 0;
    document.getElementById('body-preview').innerHTML = '';
    document.getElementById('body-status').textContent = 'Lùi xa lại để thấy nửa thân trên, sau đó bấm "1. Chụp phía trước".';
    document.getElementById('body-status').style.color = '#555';
    document.getElementById('saveBodyBtn').disabled = true;
    updateBodyWorkflowUI();
}

document.getElementById('captureBodyFrontBtn').addEventListener('click', function() {
    captureBodyImage('front');
});

document.getElementById('captureBodyBackBtn').addEventListener('click', function() {
    captureBodyImage('back');
});

function captureBodyImage(type) {
    if (!stream) return;

    webcamCanvas.width = video.videoWidth || 640;
    webcamCanvas.height = video.videoHeight || 480;
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const dataUrl = webcamCanvas.toDataURL('image/jpeg');

    document.getElementById('body-status').textContent = "⏳ Đang kiểm tra ảnh...";
    document.getElementById('body-status').style.color = "#007bff";

    if (type === 'front') {
        fetch('/get_landmarks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                addBodyImage(dataUrl, bodyAngles[currentBodyAngleIndex]);
            } else {
                document.getElementById('body-status').textContent = `❌ Lỗi: ${data.message}. Ảnh phía trước BẮT BUỘC phải thấy mặt.`;
                document.getElementById('body-status').style.color = '#dc3545';
            }
        });
    } else {
        addBodyImage(dataUrl, bodyAngles[currentBodyAngleIndex]);
    }
}

function addBodyImage(dataUrl, label) {
    bodyImagesArray.push(dataUrl);
    
    const cardWrapper = document.createElement('div');
    cardWrapper.style.cssText = 'position: relative; border: 2px solid #28a745; border-radius: 5px; overflow: hidden; width: 120px; height: 90px;';
    
    const previewImg = document.createElement('img');
    previewImg.src = dataUrl;
    previewImg.style.cssText = 'width: 120px; height: 90px; object-fit: cover; display: block;';
    
    const angleLabel = document.createElement('div');
    angleLabel.textContent = label;
    angleLabel.style.cssText = 'position: absolute; bottom: 0; width: 100%; background: rgba(40, 167, 69, 0.85); color: white; font-size: 10px; text-align: center; padding: 2px 0;';
    
    cardWrapper.appendChild(previewImg);
    cardWrapper.appendChild(angleLabel);
    document.getElementById('body-preview').appendChild(cardWrapper);

    document.getElementById('body-status').textContent = `✅ Đã chụp: ${label}`;
    document.getElementById('body-status').style.color = '#28a745';
    
    currentBodyAngleIndex++;
    updateBodyWorkflowUI();
}

document.getElementById('resetBodyBtn').addEventListener('click', function() {
    if (bodyImagesArray.length > 0 && confirm("Xóa ảnh cơ thể đã chụp và làm lại?")) {
        resetBodyWorkflow();
    }
});

document.getElementById('saveBodyBtn').addEventListener('click', function() {
    if (bodyImagesArray.length < 2) return;
    
    document.getElementById('body-status').textContent = "⏳ Đang lưu đặc trưng cơ thể...";
    document.getElementById('body-status').style.color = "#007bff";
    document.getElementById('saveBodyBtn').disabled = true;

    fetch('/register_body', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: currentRegisteredName, images: bodyImagesArray })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('body-status').innerHTML = `🎉 ${data.message}<br><b>Hoàn tất đăng ký!</b>`;
            document.getElementById('body-status').style.color = '#28a745';
            
            setTimeout(() => {
                resetWebcamWorkflow();
                webcamName.value = '';
            }, 3000);
        } else {
            document.getElementById('body-status').textContent = '❌ ' + data.message;
            document.getElementById('body-status').style.color = '#dc3545';
            document.getElementById('saveBodyBtn').disabled = false;
        }
    })
    .catch(err => {
        document.getElementById('body-status').textContent = 'Mất kết nối: ' + err.message;
        document.getElementById('body-status').style.color = '#dc3545';
        document.getElementById('saveBodyBtn').disabled = false;
    });
});


// ==========================================
// STATIC IMAGE UPLOAD TAB
// ==========================================
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