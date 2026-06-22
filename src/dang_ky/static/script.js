// Khai báo các thành phần DOM Core của hệ thống
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

// --- HỆ THỐNG QUẢN LÝ BIẾN TOÀN CỤC CHO WEBCAM WORKFLOW ---
let stream = null;
let capturedImagesArray = []; // Mảng lưu trữ tối đa 5 chuỗi ảnh base64
const faceAngles = [
    "Nhìn thẳng đối diện",
    "Quay nghiêng sang Trái",
    "Quay nghiêng sang Phải",
    "Ngước mặt lên Trên",
    "Cúi mặt xuống Dưới"
];
let currentAngleIndex = 0; // Vị trí góc đang yêu cầu thực hiện

// Điều khiển logic chuyển đổi các thanh Tabs ứng dụng
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const tab = this.dataset.tab;
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');
        
        if (tab === 'webcam') {
            startWebcam();
            resetWebcamWorkflow(); // Khởi tạo lại trạng thái quy trình chụp góc độ
        } else {
            stopWebcam();
        }
    });
});

// Điều khiển Đóng/Mở phần cứng luồng Stream Webcam WiFi/USB
function startWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then(s => {
                stream = s;
                video.srcObject = s;
                video.play();
            })
            .catch(err => {
                webcamStatus.textContent = 'Lỗi truy cập thiết bị phần cứng Camera: ' + err.message;
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

// Cập nhật lại giao diện luồng làm việc đa góc độ theo trạng thái thực tế
function updateWebcamWorkflowUI() {
    captureCountSpan.textContent = `Đã chụp: ${capturedImagesArray.length}/5`;
    
    if (currentAngleIndex < faceAngles.length) {
        angleHint.textContent = `Yêu cầu: ${faceAngles[currentAngleIndex]}`;
        angleHint.style.color = "#007bff";
        captureBtn.disabled = false;
    } else {
        angleHint.textContent = "🎯 Đã thu đủ 5 góc độ!";
        angleHint.style.color = "#28a745";
        captureBtn.disabled = true; // KHÓA không cho chụp thêm nữa khi đã hoàn thành mục tiêu
    }
}

// Đặt lại dữ liệu workflow về trạng thái ban đầu
function resetWebcamWorkflow() {
    capturedImagesArray = [];
    currentAngleIndex = 0;
    capturedPreview.innerHTML = ''; // Làm sạch toàn bộ vùng hiển thị ảnh cũ
    webcamStatus.textContent = 'Hãy căn chỉnh khuôn mặt vào tâm màn hình và bấm nút để chụp góc đầu tiên.';
    webcamStatus.style.color = '#555';
    updateWebcamWorkflowUI();
}

// Lắng nghe sự kiện click nút CHỤP ẢNH LẠI (Xóa toàn bộ ảnh chụp trước đó)
resetCaptureBtn.addEventListener('click', function() {
    if (capturedImagesArray.length === 0) {
        webcamStatus.textContent = "Hệ thống chưa lưu ảnh tạm nào, bạn có thể chụp luôn.";
        return;
    }
    if (confirm("Hành động này sẽ xóa toàn bộ các góc khuôn mặt đã chụp trước đó! Bạn muốn chụp lại chứ?")) {
        resetWebcamWorkflow();
    }
});

// Logic thực thi tác vụ chụp frame của góc độ hiện tại
captureBtn.addEventListener('click', function() {
    if (currentAngleIndex >= faceAngles.length || !stream) return;

    // Thiết lập kích thước Canvas ẩn bằng luồng Video thực tế
    webcamCanvas.width = video.videoWidth || 640;
    webcamCanvas.height = video.videoHeight || 480;
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const dataUrl = webcamCanvas.toDataURL('image/jpeg');

    webcamStatus.textContent = "⏳ Đang phân tích xác thực cấu trúc góc mặt...";
    webcamStatus.style.color = "#007bff";

    // Gửi ảnh đơn lên máy chủ kiểm tra xem góc chụp có bắt được khuôn mặt không
    fetch('/get_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            // Trường hợp có mặt -> Lưu ảnh vào hàng chờ
            capturedImagesArray.push(dataUrl);

            // Tiến hành dựng khối hiển thị ảnh Preview đính kèm nhãn tên góc độ tương ứng
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

            webcamStatus.textContent = `✅ Đã ghi nhận góc: ${faceAngles[currentAngleIndex]}`;
            webcamStatus.style.color = '#28a745';

            // CHUYỂN góc sang trạng thái tiếp theo, mỗi góc chỉ chụp duy nhất 1 lần
            currentAngleIndex++;
            updateWebcamWorkflowUI();
        } else {
            // Góc quay lỗi không thấy mặt -> giữ nguyên góc bắt chụp lại
            webcamStatus.textContent = `❌ Lỗi: ${data.message}. Hãy căn chỉnh lại để chụp góc: [ ${faceAngles[currentAngleIndex]} ]`;
            webcamStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        webcamStatus.textContent = 'Mất kết nối API Server: ' + err.message;
        webcamStatus.style.color = '#dc3545';
    });
});

// Tác vụ đóng gói mảng 5 hình ảnh gửi lên Backend để nhúng Vector CSDL
saveWebcamBtn.addEventListener('click', function() {
    const name = webcamName.value.trim();
    if (!name) {
        webcamStatus.textContent = '⚠️ Hãy nhập tên định danh trước khi lưu dữ liệu!';
        webcamStatus.style.color = '#dc3545';
        return;
    }
    if (capturedImagesArray.length < 5) {
        webcamStatus.textContent = `❌ Bạn chưa hoàn thành đủ bộ 5 góc độ khuôn mặt yêu cầu (Hiện tại mới có: ${capturedImagesArray.length}/5).`;
        webcamStatus.style.color = '#dc3545';
        return;
    }

    webcamStatus.textContent = "⏳ Máy chủ đang tính toán tạo sinh nhúng mảng CSDL...";
    webcamStatus.style.color = "#007bff";

    fetch('/register_final', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, images: capturedImagesArray })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            webcamStatus.textContent = `🎉 Lưu thành công! Bộ dữ liệu của [ ${name} ] đã sẵn sàng hoạt động.`;
            webcamStatus.style.color = '#28a745';
            webcamName.value = ''; // Làm sạch ô nhập tên
            resetWebcamWorkflow();  // Reset luồng chụp phục vụ lượt đăng ký tiếp theo
        } else {
            webcamStatus.textContent = '❌ Lỗi hệ thống: ' + data.message;
            webcamStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        webcamStatus.textContent = 'Mất kết nối dữ liệu: ' + err.message;
        webcamStatus.style.color = '#dc3545';
    });
});

// --- LOGIC PHÂN HỆ TÁC VỤ 1: CHỌN VÀ TẢI ẢNH TĨNH ---
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
                    // Tiến hành vẽ điểm trích xuất Landmark lưới khuôn mặt lên giao diện canvas hiển thị
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    data.landmarks.forEach(pt => {
                        ctx.beginPath();
                        ctx.arc(pt[0], pt[1], 3, 0, 2 * Math.PI);
                        ctx.fillStyle = 'red';
                        ctx.fill();
                    });
                    uploadStatus.textContent = '✅ Phát hiện khuôn mặt hợp lệ trong ảnh';
                    uploadStatus.style.color = '#28a745';
                } else {
                    uploadStatus.textContent = '❌ Không tìm thấy thực thể mặt: ' + data.message;
                    uploadStatus.style.color = '#dc3545';
                }
            })
            .catch(err => {
                uploadStatus.textContent = 'Lỗi phân tích ảnh: ' + err.message;
            });
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
});

saveUploadBtn.addEventListener('click', function() {
    const name = uploadName.value.trim();
    if (!name) {
        uploadStatus.textContent = '⚠️ Vui lòng nhập tên người đăng ký!';
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
            uploadStatus.textContent = '❌ Thất bại: ' + data.message;
            uploadStatus.style.color = '#dc3545';
        }
    })
    .catch(err => {
        uploadStatus.textContent = 'Lỗi API: ' + err.message;
    });
});