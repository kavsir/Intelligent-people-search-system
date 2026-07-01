// ── Config (fetched once from server) ──────────────────────────────────────
let CFG = { inferred_presence_timeout_sec: 60, floor_plan: [], target_fps: 30 };
let seenEventTimes = new Set();
let timelineItems = [];
const MAX_TIMELINE = 80;

// ── Socket.IO ──────────────────────────────────────────────────────────────
const socket = io();

// ── Door state -- one entry per room, each door is fully independent ────────
// roomId -> { enabled: bool, state: 'OPEN'|'CLOSED'|'UNKNOWN' }
let doorStates = {};

// ── Update ONE room's door UI ───────────────────────────────────────────
function updateDoorUI(roomId, enabled, state) {
  doorStates[roomId] = { enabled, state: state || 'UNKNOWN' };

  const doorBtn = document.getElementById('doorBtn-' + roomId);
  const doorStatusText = document.getElementById('doorStatusText-' + roomId);
  if (!doorBtn || !doorStatusText) return;  // this room's door card isn't in the DOM (yet)

  // Button label reflects the ACTION it will perform next, based on the
  // ESP32's real door state -- not just whether it's clickable.
  //   CLOSED/UNKNOWN -> "Mở cửa" (will OPEN)
  //   OPEN           -> "Đóng cửa" (will CLOSE)
  const willOpen = state !== 'OPEN';
  const label = willOpen ? 'Mở cửa' : 'Đóng cửa';
  const icon = willOpen ? (enabled ? '🔓' : '🔒') : '🔐';

  if (enabled) {
    doorBtn.classList.add('enabled');
  } else {
    doorBtn.classList.remove('enabled');
  }
  doorBtn.textContent = `${icon} ${label}`;

  // Update status text
  doorStatusText.className = 'door-status';
  if (state === 'OPEN') {
    doorStatusText.textContent = '🔓 Đang mở';
    doorStatusText.classList.add('open');
  } else if (state === 'CLOSED') {
    doorStatusText.textContent = '🔒 Đã đóng';
    doorStatusText.classList.add('closed');
  } else if (enabled) {
    doorStatusText.textContent = '✅ Sẵn sàng';
    doorStatusText.classList.add('open');
  } else {
    doorStatusText.textContent = '⛔ Không có người đăng ký';
    doorStatusText.classList.add('unknown');
  }
}

// ── Socket event handlers ──────────────────────────────────────────────
socket.on('connect', function() {
  console.log('[WebSocket] Connected to server');
});

socket.on('door_status', function(data) {
  console.log('[WebSocket] Door status:', data);
  updateDoorUI(data.room_id, data.enabled, data.state);
});

socket.on('door_response', function(data) {
  if (data.status !== 'success') {
    alert('❌ ' + data.message);
  }
  // success case: no popup needed -- the button/status text update via
  // door_status already gives clear visual feedback.
});

// ── Door button click -- always sends WHICH room's door was pressed ─────
function toggleDoor(roomId) {
  const st = doorStates[roomId];
  if (!st || !st.enabled) return;
  socket.emit('toggle_door', { room_id: roomId });
}

// Build the "Đăng ký khuôn mặt" link. Deliberately hardcoded to "localhost"
// (NOT window.location.hostname / a LAN IP): the registration page's webcam
// tab uses getUserMedia(), which Chrome only allows on a secure context --
// https://, or http://localhost / http://127.0.0.1. Opening it via a plain
// LAN IP (e.g. http://192.168.0.8:5000) silently blocks camera access even
// though the server itself is running fine. This dashboard is always
// accessed from the same machine that runs the server, so localhost is
// always reachable here.
fetch('/api/config')
  .then(res => res.json())
  .then(cfg => {
    const registerLink = document.getElementById('registerLink');
    if (registerLink && cfg.registration_port) {
      registerLink.href = `http://localhost:${cfg.registration_port}/`;
    }
  })
  .catch(err => console.error('Failed to load /api/config:', err));

// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('vi-VN');
}
setInterval(updateClock, 1000);
updateClock();

// ── Floor map update ───────────────────────────────────────────────────────
const ROOM_IDS = ['p1','p2','p3','p4','p5'];

function stateColor(room) {
  if (room.has_target) {
    if (room.state === 'TRACKING_FACE')   return { fill:'#14532d', stroke:'#22c55e', glow:'glow-green', pulse:'green' };
    if (room.state === 'FALLBACK_PERSON') return { fill:'#431407', stroke:'#f97316', glow:'glow-green', pulse:'green' };
    return { fill:'#1c2d1a', stroke:'#22c55e', glow:'glow-green', pulse:'green' };
  }
  if (room.inferred) return { fill:'#2d2500', stroke:'#eab308', glow:'glow-yellow', pulse:'yellow' };
  if (room.state === 'LOST') return { fill:'#2d1515', stroke:'#ef4444', glow:null, pulse:null };
  return { fill:'#1a1d27', stroke:'#2e3250', glow:null, pulse:null };
}

function badgeClass(state, inferred) {
  if (inferred) return 'badge-inferred';
  if (state === 'TRACKING_FACE') return 'badge-tracking';
  if (state === 'FALLBACK_PERSON') return 'badge-fallback';
  if (state === 'LOST') return 'badge-lost';
  if (state === 'SEARCHING') return 'badge-search';
  return 'badge-empty';
}

function badgeLabel(state, inferred, hasCam) {
  if (inferred) return '⚠ Suy luận';
  if (state === 'TRACKING_FACE')   return '✔ Face';
  if (state === 'FALLBACK_PERSON') return '~ Person';
  if (state === 'LOST')            return '✗ Lost';
  if (state === 'SEARCHING')       return '⌕ Search';
  if (state === 'EMPTY')           return hasCam ? '⌕ Trống' : '— Không cam';
  return state || '--';
}

function updateFloorMap(rooms) {
  rooms.forEach(room => {
    const col = stateColor(room);
    const rect = document.getElementById('rect-' + room.id);
    const sub  = document.getElementById('sub-' + room.id);
    const idEl = document.getElementById('id-'  + room.id);
    if (!rect) return;

    rect.setAttribute('fill', col.fill);
    rect.setAttribute('stroke', col.stroke);
    rect.setAttribute('filter', col.glow ? `url(#${col.glow})` : '');

    // pulse ring
    const circle = rect.previousElementSibling;
    if (circle && circle.tagName === 'circle') {
      if (col.pulse === 'green') {
        circle.setAttribute('stroke', '#22c55e');
        circle.setAttribute('r', '45');
        circle.setAttribute('opacity', '0');
        circle.style.animation = 'pulse-ring 1.2s ease-out infinite';
      } else if (col.pulse === 'yellow') {
        circle.setAttribute('stroke', '#eab308');
        circle.setAttribute('r', '45');
        circle.setAttribute('opacity', '0');
        circle.style.animation = 'pulse-ring-slow 2s ease-out infinite';
      } else {
        circle.style.animation = 'none';
        circle.setAttribute('r', '0');
        circle.setAttribute('opacity', '0');
      }
    }

    // sub text: state label
    if (sub) {
      if (room.has_target) {
        sub.textContent = room.state === 'TRACKING_FACE' ? '✔ Face tracking' :
                          room.state === 'FALLBACK_PERSON' ? '~ Person tracking' : room.state;
        sub.setAttribute('fill', room.state === 'TRACKING_FACE' ? '#22c55e' : '#f97316');
      } else if (room.inferred) {
        const t = CFG.inferred_presence_timeout_sec;
        const elapsed = room.inferred_since || 0;
        const remain = t > 0 ? Math.max(0, t - elapsed) : '∞';
        sub.textContent = `⚠ Có thể ở đây (${remain}s)`;
        sub.setAttribute('fill', '#eab308');
      } else if (room.state === 'LOST') {
        sub.textContent = '✗ Mất dấu';
        sub.setAttribute('fill', '#ef4444');
      } else {
        sub.textContent = room.has_cam ? '⌕ Đang quét' : '—';
        sub.setAttribute('fill', '#64748b');
      }
    }

    // identity text
    if (idEl) {
      idEl.textContent = room.identity ? `👤 ${room.identity}` : '';
      idEl.setAttribute('fill', '#a855f7');
    }
  });
}

// ── Room detail cards ──────────────────────────────────────────────────────
function updateDetailCards(rooms) {
  const container = document.getElementById('rooms-detail');
  // Build or update cards
  rooms.forEach(room => {
    let card = document.getElementById('card-' + room.id);
    if (!card) {
      card = document.createElement('div');
      card.id = 'card-' + room.id;
      card.className = 'room-detail-card';
      card.innerHTML = `
        <div class="room-dot" id="dot-${room.id}"></div>
        <div class="room-info">
          <div class="room-info-name">${room.room_name}${room.has_cam ? ' 📷' : ''}</div>
          <div class="room-info-sub" id="card-sub-${room.id}">--</div>
        </div>
        <span class="room-state-badge" id="card-badge-${room.id}">--</span>
      `;
      container.appendChild(card);
    }

    const dot   = document.getElementById('dot-' + room.id);
    const sub   = document.getElementById('card-sub-' + room.id);
    const badge = document.getElementById('card-badge-' + room.id);

    card.className = 'room-detail-card' + (room.has_target ? ' active' : room.inferred ? ' inferred' : '');
    dot.className  = 'room-dot' + (room.has_target ? ' active' : room.inferred ? ' inferred' : '');

    let subText = '';
    if (room.identity) subText += `👤 ${room.identity}  `;
    if (room.latency_ms !== null && room.latency_ms !== undefined)
      subText += `⏱ ${room.latency_ms}ms`;
    if (room.inferred && room.inferred_since !== null) {
      const t = CFG.inferred_presence_timeout_sec;
      const remain = t > 0 ? Math.max(0, t - room.inferred_since) + 's' : '∞';
      subText = `Suy luận từ lân cận  (tự xóa: ${remain})`;
    }
    sub.textContent = subText || (room.has_cam ? 'Không phát hiện' : 'Không có camera');

    badge.textContent = badgeLabel(room.state, room.inferred, room.has_cam);
    badge.className   = 'room-state-badge ' + badgeClass(room.state, room.inferred);
  });
}

// ── Timeline ───────────────────────────────────────────────────────────────
function tlClass(event) {
  if (event.includes('ACQUIRED'))   return 'tl-ACQUIRED';
  if (event.includes('LOST'))       return 'tl-LOST';
  if (event.includes('REACQUIRED')) return 'tl-REACQUIRED';
  if (event.includes('IGNORED'))    return 'tl-IGNORED';
  if (event.includes('INFERRED'))   return 'tl-INFERRED';
  return 'tl-other';
}

async function pollTimeline() {
  try {
    const r = await fetch('/api/events?n=60');
    const data = await r.json();
    const list = document.getElementById('timeline-list');
    let changed = false;

    data.events.forEach(ev => {
      const key = ev.time + '|' + ev.event + '|' + ev.details;
      if (seenEventTimes.has(key)) return;
      seenEventTimes.add(key);
      timelineItems.unshift(ev);  // newest first
      changed = true;
    });

    if (changed) {
      if (timelineItems.length > MAX_TIMELINE) timelineItems = timelineItems.slice(0, MAX_TIMELINE);
      list.innerHTML = timelineItems.map(ev => `
        <div class="tl-item ${tlClass(ev.event)}">
          <span class="tl-time">${ev.time}</span>
          <span class="tl-event">${ev.event}</span>
          <span class="tl-detail">${ev.details}</span>
        </div>
      `).join('');
    }
  } catch(e) {}
}

function clearTimeline() {
  timelineItems = [];
  seenEventTimes.clear();
  document.getElementById('timeline-list').innerHTML =
    '<div class="tl-item tl-other"><span class="tl-time">--</span><span class="tl-event">CLEARED</span><span class="tl-detail"></span></div>';
}

// ── Reset inference ────────────────────────────────────────────────────────
async function resetInference() {
  try {
    await fetch('/api/reset_inference', { method: 'POST' });
    document.getElementById('status-inference').textContent = 'Đã reset suy luận';
    setTimeout(() => document.getElementById('status-inference').textContent = '', 3000);
  } catch(e) {}
}

// ── Exercise assignment panel ────────────────────────────────────────────
async function loadRegisteredPeople() {
  try {
    const r = await fetch('/api/registered_people');
    const data = await r.json();
    const sel = document.getElementById('exPersonSelect');
    const current = sel.value;
    sel.innerHTML = '<option value="">-- Chọn người --</option>' +
      data.names.map(n => `<option value="${n}">${n}</option>`).join('');
    if (data.names.includes(current)) sel.value = current;
  } catch (e) {}
}

async function assignExercise() {
  const name = document.getElementById('exPersonSelect').value;
  const exercise = document.getElementById('exTypeSelect').value;
  const target_reps = document.getElementById('exRepsInput').value;

  if (!name) { alert('Vui lòng chọn người.'); return; }

  try {
    const r = await fetch('/api/exercises/assign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, exercise, target_reps })
    });
    const data = await r.json();
    if (data.status !== 'ok') {
      alert('❌ ' + data.message);
      return;
    }
    renderExerciseTable(data.rows);
  } catch (e) {
    alert('❌ Lỗi kết nối server.');
  }
}

async function deleteExerciseRow(name) {
  if (!confirm(`Xoá hàng của "${name}"? Người này sẽ trở về trạng thái off và số lần reset về 0.`)) return;
  try {
    const r = await fetch('/api/exercises/' + encodeURIComponent(name), { method: 'DELETE' });
    const data = await r.json();
    renderExerciseTable(data.rows);
  } catch (e) {}
}

function exMarkCell(row, type) {
  return row.exercise === type
    ? `<span class="ex-check">✔</span>`
    : `<span class="ex-dash">—</span>`;
}

function exResultBadge(result) {
  if (result === 'success') return '<span class="ex-badge ex-success">✔ Thành công</span>';
  if (result === 'fail')    return '<span class="ex-badge ex-fail">✗ Thất bại</span>';
  return '<span class="ex-badge ex-pending">… Đang chờ</span>';
}

function renderExerciseTable(rows) {
  const tbody = document.getElementById('exercise-tbody');
  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="ex-empty">Chưa có ai được gán bài tập</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(row => `
    <tr>
      <td>${row.name}</td>
      <td>${exMarkCell(row, 'squat')}</td>
      <td>${exMarkCell(row, 'pushup')}</td>
      <td>${row.count} / ${row.target_reps}</td>
      <td><span class="ex-badge ${row.online ? 'ex-online' : 'ex-offline'}">${row.online ? 'Online' : 'Offline'}</span></td>
      <td>${exResultBadge(row.result)}</td>
      <td><button class="ex-del-btn" onclick="deleteExerciseRow('${row.name}')" title="Xoá">🗑</button></td>
    </tr>
  `).join('');
}

async function pollExercises() {
  try {
    const r = await fetch('/api/exercises');
    const data = await r.json();
    renderExerciseTable(data.rows);
  } catch (e) {}
}

// ── Main poll loop ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/room_status');
    const data = await r.json();

    updateFloorMap(data.rooms);
    updateDetailCards(data.rooms);

    const active = data.rooms.filter(r => r.has_target).map(r => r.room_name);
    const inferred = data.rooms.filter(r => r.inferred).map(r => r.room_name);
    document.getElementById('status-update').textContent =
      'Cập nhật: ' + new Date().toLocaleTimeString('vi-VN') +
      (active.length ? '  |  🔴 Phát hiện: ' + active.join(', ') : '') +
      (inferred.length ? '  |  🟡 Suy luận: ' + inferred.join(', ') : '');
  } catch(e) {
    document.getElementById('status-update').textContent = 'Mất kết nối: ' + e.message;
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
(async () => {
  await fetchConfig();
  await pollStatus();
  await pollTimeline();
  await loadRegisteredPeople();
  await pollExercises();
  setInterval(pollStatus, 500);
  setInterval(pollTimeline, 2000);
  setInterval(pollExercises, 1000);
  setInterval(loadRegisteredPeople, 10000);
})();