// --- Clock (same pattern as dashboard.js) ---
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('vi-VN');
}
setInterval(updateClock, 1000);
updateClock();

const DOOR_LABEL = {
  OPEN: { text: '🔓 Đang mở', cls: 'pp-door-open' },
  CLOSED: { text: '🔒 Đang đóng', cls: 'pp-door-closed' },
  UNKNOWN: { text: '⏳ Không rõ', cls: 'pp-door-unknown' },
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

  return `
    <tr>
      <td class="pp-name">${escapeHtml(p.name)}</td>
      <td>${room}</td>
      <td>${doorCell}</td>
    </tr>
  `;
}

async function pollPeople() {
  try {
    const res = await fetch('/api/people_overview');
    const data = await res.json();
    const tbody = document.getElementById('people-tbody');

    if (!data.people || data.people.length === 0) {
      tbody.innerHTML = `<tr><td colspan="3" class="pp-empty">Chưa có ai được đăng ký khuôn mặt.</td></tr>`;
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