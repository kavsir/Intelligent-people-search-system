// --- Clock (same pattern as dashboard.js) ---
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('vi-VN');
}
setInterval(updateClock, 1000);
updateClock();

const EXERCISE_LABEL = { squat: 'Squat', pushup: 'Hít đất' };

const DOOR_LABEL = {
  OPEN: { text: '🔓 Đang mở', cls: 'pp-door-open' },
  CLOSED: { text: '🔒 Đang đóng', cls: 'pp-door-closed' },
  UNKNOWN: { text: '⏳ Không rõ', cls: 'pp-door-unknown' },
};

const RESULT_LABEL = {
  success: { text: '✅ Thành công', cls: 'pp-result-success' },
  fail: { text: '❌ Thất bại', cls: 'pp-result-fail' },
  pending: { text: '⏳ Đang thực hiện', cls: 'pp-result-pending' },
  none: { text: '—', cls: 'pp-result-none' },
};

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderRow(p) {
  const room = p.room_name
    ? `<span class="pp-room">${escapeHtml(p.room_name)}</span>`
    : `<span class="pp-room pp-dash">— Không xác định —</span>`;

  const door = DOOR_LABEL[p.door_state] || DOOR_LABEL.UNKNOWN;
  const doorCell = p.room_name
    ? `<span class="pp-door ${door.cls}">${door.text}</span>`
    : `<span class="pp-door pp-door-unknown">—</span>`;

  // Đã xóa cột "Đang làm gì" – không cần activity nữa

  const reps = p.assigned
    ? `<span class="pp-reps">${p.count} <span class="pp-target">/ ${p.target_reps}</span></span>`
    : `<span class="pp-reps pp-target">—</span>`;

  const statusCell = p.assigned
    ? (p.online
        ? `<span class="pp-status pp-status-online">🟢 Đang tập</span>`
        : `<span class="pp-status pp-status-offline">⚪ Ngoại tuyến</span>`)
    : `<span class="pp-status pp-status-offline">—</span>`;

  const resultKey = p.assigned ? (p.result || 'pending') : 'none';
  const result = RESULT_LABEL[resultKey] || RESULT_LABEL.none;

  return `
    <tr>
      <td class="pp-name">${escapeHtml(p.name)}</td>
      <td>${room}</td>
      <td>${doorCell}</td>
      <td>${reps}</td>
      <td>${statusCell}</td>
      <td><span class="pp-result ${result.cls}">${result.text}</span></td>
    </tr>
  `;
}

async function pollPeople() {
  try {
    const res = await fetch('/api/people_overview');
    const data = await res.json();
    const tbody = document.getElementById('people-tbody');

    if (!data.people || data.people.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="pp-empty">Chưa có ai được đăng ký khuôn mặt.</td></tr>`;
    } else {
      tbody.innerHTML = data.people.map(renderRow).join('');
    }

    document.getElementById('people-status-update').textContent =
      `Cập nhật lúc ${new Date().toLocaleTimeString('vi-VN')} · ${data.people.length} người`;
  } catch (err) {
    document.getElementById('people-status-update').textContent = 'Mất kết nối tới server...';
  }
}

pollPeople();
setInterval(pollPeople, 1500);