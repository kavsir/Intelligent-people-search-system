"""
servo_controller.py
--------------------
Bo dieu khien servo pan/tilt: PID + loc nhieu do luong + dead-zone tre
(hysteresis) + khoi dong em (step ramp-up).

Day la ban chuyen tu project pan-tilt tracking doc lap cua ban
(controller.py) vao package operation/ cua he thong AIoT da camera chinh --
logic PID/loc/ramp-up giu NGUYEN VEN, khong doi. Khac biet:

  1) Sai so (error_x, error_y) den tu AIPipeline (ai_pipeline.py) -- tam
     muc tieu da khoa (Kalman-smoothed center) cua mot nguoi DA DANG KY,
     khong phai tu detector.py/tracker.py rieng cua project cu.

  2) TRANSPORT: ban goc gui lenh qua Serial USB toi mot ESP32 PCA9685
     rieng. Ban nay gui qua CUNG ket noi WebSocket dung cho 2 servo cua
     (operation/door_ws_server.py -> esp32_servo.ino), vi gio ca 4 servo
     (pan, tilt, cua 1, cua 2) deu nam tren MOT board ESP32 + MOT PCA9685
     duy nhat, noi chuyen voi server qua WiFi thay vi cam day Serial vao
     may tinh. Khong con phu thuoc pyserial/cong COM nua.

  3) SCAN MODE (moi): ngoai che do bam muc tieu bang PID (update()), servo
     con co the duoc yeu cau chu dong "quet" pan+tilt qua lai trong mot
     dai goc chi dinh (start_scan()/tick_scan()/stop_scan()) khi khong con
     error do duoc de bam theo -- dung cho kich ban ban giao Phong 2 -> 1:
     neu qua HANDOFF_WAIT_BEFORE_SCAN_SEC giay ma Phong 1 van chua tim ra
     nguoi vua mat dau, Phong 2 chu dong xoay di tim thay vi dung yen cho.

ServoController khong quan tam muc tieu la ai / duoc tim thay bang cach
nao (FACE / BODY / BODY_SHAPE) -- no chi nhan (error_x, error_y) moi frame
va tinh goc moi, roi gui qua door_ws.send_pan_tilt(pan, tilt).
"""

import time


class ServoController:
    def __init__(self, config: dict, door_ws=None):
        """
        door_ws: an operation.door_ws_server.DoorWebSocketServer instance
            (the SAME one app_dashboard.py uses for the door servos) --
            pan/tilt commands are sent via door_ws.send_pan_tilt(pan, tilt)
            over that shared WebSocket connection. If None (e.g. running
            under app_operation.py's standalone CLI dashboard, which has
            no DoorWebSocketServer), the controller still computes angles
            normally but simply doesn't send anything -- same graceful
            "simulate" behavior the old Serial-based version had when no
            hardware was plugged in.
        """
        c = config["control"]
        self.kp_pan = c["kp_pan"]
        self.kp_tilt = c["kp_tilt"]
        self.kd_pan = c.get("kd_pan", 0.0)
        self.kd_tilt = c.get("kd_tilt", 0.0)
        self.ki_pan = c.get("ki_pan", 0.0)
        self.ki_tilt = c.get("ki_tilt", 0.0)
        self.integral_limit = c.get("integral_limit_px", 300.0)

        # Loc muot sai so do TRUOC khi tinh P/I/D
        self.error_filter_alpha = c.get("error_filter_alpha", 0.35)
        self._filtered_error_x = 0.0
        self._filtered_error_y = 0.0

        self.dead_zone = c["dead_zone_px"]
        self.hysteresis_factor = c.get("dead_zone_hysteresis_factor", 1.6)
        self._settled_x = True
        self._settled_y = True

        # Khoi dong em: tran buoc di chuyen hien tai, tang dan moi frame
        self.max_step = c.get("max_step_per_frame", 5)
        self.step_ramp_per_frame = c.get("step_ramp_per_frame_deg", 0.5)
        self._current_cap_x = 0.0
        self._current_cap_y = 0.0

        self.pan_min, self.pan_max = c["pan_min"], c["pan_max"]
        self.tilt_min, self.tilt_max = c["tilt_min"], c["tilt_max"]
        self.invert_pan = c.get("invert_pan", False)
        self.invert_tilt = c.get("invert_tilt", False)

        self.pan_angle = c["pan_center"]
        self.tilt_angle = c["tilt_center"]

        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0

        self.send_interval = c.get("send_interval_sec", 0.05)
        self._last_send = 0.0

        # --- Che do quet chu dong (khong dung PID, khong co error do) ---
        # Dung khi ban giao muc tieu sang phong khac va phong nguon can
        # chu dong xoay di "tim" thay vi dung yen cho gap lai.
        self.scanning = False
        self._scan_pan_dir = 1
        self._scan_tilt_dir = 1
        self._scan_cfg = None

        self.door_ws = door_ws
        self.simulate = door_ws is None
        if self.simulate:
            print(
                "[servo_controller] Khong co door_ws (WebSocket server) duoc truyen vao -- "
                "chay o che do MO PHONG (van tinh goc, khong gui lenh xuong ESP32)."
            )
        else:
            print("[servo_controller] Se gui lenh pan/tilt qua WebSocket dung chung voi cua (esp32_servo.ino).")

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def reset_integral(self):
        """Goi khi mat khoa muc tieu / huy khoa / an toan ve tam."""
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._filtered_error_x = 0.0
        self._filtered_error_y = 0.0
        self._settled_x = True
        self._settled_y = True
        self._current_cap_x = 0.0
        self._current_cap_y = 0.0

    def _update_axis(self, raw_error, filtered_prev, settled, prev_error, integral,
                      current_cap, kp, ki, kd):
        """Tinh step cho 1 truc (dung chung cho pan/tilt). Tra ve:
        (step, filtered_error_moi, settled_moi, integral_moi, cap_moi)"""
        filtered = (self.error_filter_alpha * raw_error
                    + (1 - self.error_filter_alpha) * filtered_prev)

        threshold = self.dead_zone * self.hysteresis_factor if settled else self.dead_zone

        if abs(filtered) > threshold:
            was_settled = settled
            settled = False
            if was_settled:
                current_cap = 0.0
            current_cap = min(current_cap + self.step_ramp_per_frame, self.max_step)

            integral = self._clamp(integral + filtered, -self.integral_limit, self.integral_limit)
            derivative = filtered - prev_error
            step = kp * filtered + ki * integral + kd * derivative
            step = self._clamp(step, -current_cap, current_cap)
        else:
            settled = True
            integral = 0.0
            current_cap = 0.0
            step = 0.0

        return step, filtered, settled, integral, current_cap

    def update(self, error_x, error_y):
        """
        Tinh goc pan/tilt moi tu sai lech pixel (PID + loc nhieu + dead-zone
        tre + khoi dong em). Tra ve (pan_angle, tilt_angle).
        """
        (step_pan, self._filtered_error_x, self._settled_x,
         self._integral_x, self._current_cap_x) = self._update_axis(
            error_x, self._filtered_error_x, self._settled_x, self._prev_error_x,
            self._integral_x, self._current_cap_x, self.kp_pan, self.ki_pan, self.kd_pan,
        )

        (step_tilt, self._filtered_error_y, self._settled_y,
         self._integral_y, self._current_cap_y) = self._update_axis(
            error_y, self._filtered_error_y, self._settled_y, self._prev_error_y,
            self._integral_y, self._current_cap_y, self.kp_tilt, self.ki_tilt, self.kd_tilt,
        )

        self._prev_error_x = self._filtered_error_x
        self._prev_error_y = self._filtered_error_y

        if self.invert_pan:
            step_pan = -step_pan
        if self.invert_tilt:
            step_tilt = -step_tilt

        self.pan_angle = self._clamp(self.pan_angle + step_pan, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.tilt_angle - step_tilt, self.tilt_min, self.tilt_max)

        self._send()
        return self.pan_angle, self.tilt_angle

    def go_to_center(self, config):
        """Co che an toan khi mat muc tieu: dua servo ve vi tri mac dinh.
        Cung dung de KET THUC che do quet chu dong (vd khi phong dich da
        tim ra nguoi vua ban giao)."""
        self.stop_scan()
        self.reset_integral()
        self.pan_angle = config["control"]["pan_center"]
        self.tilt_angle = config["control"]["tilt_center"]
        self._send(force=True)

    def preempt_to_angle(self, pan_angle, tilt_angle=None):
        """
        Lenh 'don dau' (handoff preempt): ep servo quay NGAY toi pan_angle
        chi dinh, KHONG qua PID/ramp-up thong thuong -- vi luc nay chua he
        co sai so do (error_x/error_y) nao ca, ta dang CHU DONG doan truoc
        vi tri doi tuong SAP xuat hien dua tren tin hieu tu phong khac, chu
        khong phai dang bam theo detection cua chinh phong nay.

        Reset luon bo dieu khien PID de khong bi nhieu boi sai so/tich luy
        cu con sot lai tu lan bam muc tieu truoc do, va huy che do quet chu
        dong neu dang bat (uu tien lenh don dau tuc thi hon quet).
        """
        self.stop_scan()
        self.reset_integral()
        self.pan_angle = self._clamp(pan_angle, self.pan_min, self.pan_max)
        if tilt_angle is not None:
            self.tilt_angle = self._clamp(tilt_angle, self.tilt_min, self.tilt_max)
        self._send(force=True)

    # -------------------------------------------------------------------
    # Che do quet chu dong (active scan) -- dung khi ban giao muc tieu
    # sang phong khac va can chu dong xoay tim thay vi dung yen cho.
    # -------------------------------------------------------------------

    def start_scan(self, pan_min, pan_max, tilt_min, tilt_max, step_deg=3, tilt_center=None):
        """
        Bat dau quet chu dong CHI PAN qua lai trong dai [pan_min, pan_max]
        (khong dung PID, vi khong co error do duoc -- day la "mo" tim,
        khac han bam muc tieu binh thuong). TILT duoc dua ve co dinh o
        tilt_center (mac dinh 90 do) va GIU NGUYEN suot qua trinh quet --
        khong quet doc theo tilt nua, theo yeu cau thuc te (quet ca tilt
        tu 40->160 lam camera "nguoc len" qua cao, khong huu ich).

        Dung khi: muc tieu vua duoc ban giao sang phong khac (vd Phong 2 ->
        Phong 1) va da qua thoi gian cho (config.HANDOFF_WAIT_BEFORE_SCAN_SEC)
        ma phong dich van chua tu tim ra nguoi do.

        pan_min/pan_max: thuong la bien logic cua phong (vd
            HANDOFF_CONFIG["cam2"]["pan_left_boundary"/"pan_right_boundary"]),
            khong nhat thiet trung voi gioi han co khi pan_min/pan_max cua
            servo.
        tilt_min/tilt_max: GIOI HAN AN TOAN cho tilt trong che do quet (vd
            config.HANDOFF_SCAN_TILT_MIN/MAX = 40/160) -- chi dung de KEP
            tilt_center vao trong khoang nay cho an toan co khi, KHONG con
            dung de quet doc nua.
        tilt_center: goc tilt co dinh khi quet. None -> dung tilt_center
            cua SERVO_CONFIG (thuong la 90 do).
        """
        self.reset_integral()
        self.scanning = True
        if tilt_center is None:
            tilt_center = (self.tilt_min + self.tilt_max) / 2.0
        tilt_center = self._clamp(tilt_center, tilt_min, tilt_max)
        self._scan_cfg = {
            "pan_min": pan_min,
            "pan_max": pan_max,
            "tilt_center": tilt_center,
            "step": step_deg,
        }
        self._scan_pan_dir = 1
        # Dua tilt ve co dinh NGAY khi bat dau quet, gui lenh ngay lap tuc
        # thay vi doi den tick dau tien -- tranh tilt bi "treo" o goc cu
        # (vd 150-160 do neu vua bam muc tieu o bien tren truoc do).
        self.tilt_angle = self._clamp(tilt_center, self.tilt_min, self.tilt_max)
        self._send(force=True)

    def stop_scan(self):
        """Dung che do quet chu dong (goc hien tai giu nguyen, khong tu
        nhay ve center -- goi go_to_center() rieng neu muon ve mac dinh)."""
        self.scanning = False
        self._scan_cfg = None

    def tick_scan(self):
        """
        Goi dinh ky (vd moi HANDOFF_SCAN_TICK_INTERVAL_SEC giay, tu
        HandoffManager.check_pending() trong app_dashboard.py) de tien
        them 1 buoc quet PAN. Khong lam gi neu scanning=False.

        Pan di chuyen kieu "con thoi" (bat khi cham bien thi doi chieu).
        Tilt GIU NGUYEN o tilt_center da chot luc start_scan() -- khong
        con quet doc theo tilt.
        """
        if not self.scanning or self._scan_cfg is None:
            return
        c = self._scan_cfg

        self.pan_angle += self._scan_pan_dir * c["step"]
        if self.pan_angle >= c["pan_max"]:
            self.pan_angle = c["pan_max"]
            self._scan_pan_dir = -1
        elif self.pan_angle <= c["pan_min"]:
            self.pan_angle = c["pan_min"]
            self._scan_pan_dir = 1

        # Tilt co dinh -- khong tang/giam theo buoc quet nua.
        self.tilt_angle = c["tilt_center"]

        # Van kep trong gioi han co khi cua servo (pan_min/pan_max,
        # tilt_min/tilt_max) de khong bao gio vuot qua phan cung.
        self.pan_angle = self._clamp(self.pan_angle, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.tilt_angle, self.tilt_min, self.tilt_max)

        self._send(force=True)

    def _send(self, force=False):
        now = time.time()
        if not force and (now - self._last_send) < self.send_interval:
            return
        self._last_send = now

        if self.simulate or self.door_ws is None:
            return  # no WebSocket server reference -- angles still computed, just not sent

        self.door_ws.send_pan_tilt(self.pan_angle, self.tilt_angle)
        # send_pan_tilt() is fire-and-forget and returns False if the
        # ESP32 isn't currently connected -- not fatal, PID state keeps
        # updating normally and delivery resumes the moment it reconnects
        # (esp32_servo.ino auto-retries every 3s).

    def close(self):
        """No persistent connection of our own to close -- door_ws (owned
        by app_dashboard.py) manages the actual WebSocket lifecycle.
        Kept for API compatibility with the old Serial-based version."""
        pass