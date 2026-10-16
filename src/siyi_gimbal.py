"""
SIYI A8 Mini Gimbal Controller - TCP Port 37256
"""
import socket
import struct
import threading
import time

def _crc16_ccitt_table():
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


class SIYIGimbalController:
    HEADER = 0x6655
    CMD_FIRMWARE_VER = 0x01
    CMD_CENTER = 0x08
    CMD_GIMBAL_ROTATE = 0x07
    CMD_SET_ANGLES = 0x0E
    CMD_GET_ATTITUDE = 0x0D

    # SIYI SDK protocol: CRC16, G(X) = X^16 + X^12 + X^5 + 1 (CCITT, init 0, not reflected).
    crc16_tab = _crc16_ccitt_table()

    def __init__(self, host='192.168.2.25', port=37256):
        self.host = host
        self.port = port
        self.seq = 0
        self.sock = None
        self._lock = threading.Lock()  # tüm socket işlemleri için
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self._running = False
        self._attitude_thread = None
        self._connected = False
        self._silent_reconnect = False
        self._connect()

    def _connect(self):
        try:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.1)
            self._connected = True
            if not self._silent_reconnect:
                print(f"[GIMBAL] TCP baglanti OK {self.host}:{self.port}")
            self._silent_reconnect = True
        except Exception as e:
            self._connected = False
            self.sock = None
            if not self._silent_reconnect:
                print(f"[GIMBAL] Baglanti hatasi: {e}")

    def _calc_crc(self, data):
        crc = 0
        for b in data:
            crc = ((crc << 8) & 0xFFFF) ^ self.crc16_tab[((crc >> 8) ^ b) & 0xFF]
        return crc

    def _build_packet(self, cmd_id, data=b''):
        # _lock alınmış halde çağrılmalı
        self.seq = (self.seq + 1) % 65536
        packet = struct.pack('<HBHHB',
                             self.HEADER, 0x01, len(data),
                             self.seq, cmd_id)
        packet += data
        crc = self._calc_crc(packet)
        packet += struct.pack('<H', crc)
        return packet

    def _send_recv(self, cmd_id, data=b''):
        with self._lock:
            if not self._connected or self.sock is None:
                self._connect()
                if not self._connected:
                    return None
            packet = self._build_packet(cmd_id, data)
            try:
                self.sock.send(packet)
                return self.sock.recv(1024)
            except socket.timeout:
                return None
            except Exception:
                self._connected = False
                return None

    def _send_only(self, cmd_id, data=b''):
        """Yanit beklenmeden gonder - rotate gibi yuksek frekanslı komutlar icin."""
        with self._lock:
            if not self._connected or self.sock is None:
                self._connect()
                if not self._connected:
                    return
            packet = self._build_packet(cmd_id, data)
            try:
                self.sock.send(packet)
            except Exception:
                self._connected = False

    def get_firmware_version(self):
        for _ in range(3):
            with self._lock:
                if self.sock is not None:
                    try:
                        self.sock.settimeout(1.0)
                    except Exception:
                        pass
            resp = self._send_recv(self.CMD_FIRMWARE_VER)
            with self._lock:
                if self.sock is not None:
                    try:
                        self.sock.settimeout(0.1)
                    except Exception:
                        pass
            if resp and len(resp) > 0:
                return resp.hex()
            time.sleep(0.2)
        return None

    def center_gimbal(self):
        self._send_recv(self.CMD_CENTER, struct.pack('<B', 1))

    def set_gimbal_rotation(self, yaw_speed, pitch_speed):
        yaw_speed = max(-100, min(100, int(yaw_speed)))
        pitch_speed = max(-100, min(100, int(pitch_speed)))
        data = struct.pack('<bb', yaw_speed, pitch_speed)
        # Rotate komutu yüksek frekanslı; yanıt beklemeye gerek yok
        self._send_only(self.CMD_GIMBAL_ROTATE, data)

    @classmethod
    def parse_attitude(cls, resp):
        """Return (yaw, pitch, roll) in degrees from a 0x0D reply, or None.

        Frame: STX(2) CTRL(1) DATA_LEN(2) SEQ(2) CMD_ID(1) DATA CRC16(2).
        DATA for 0x0D starts with yaw, pitch, roll as int16 in 0.1 degree.
        """
        if not resp:
            return None
        idx = resp.find(b'\x55\x66')
        if idx < 0 or idx + 14 > len(resp) or resp[idx + 7] != cls.CMD_GET_ATTITUDE:
            return None
        yaw, pitch, roll = struct.unpack_from('<hhh', resp, idx + 8)
        return yaw / 10.0, pitch / 10.0, roll / 10.0

    def get_attitude(self):
        att = self.parse_attitude(self._send_recv(self.CMD_GET_ATTITUDE))
        if att is not None:
            self.current_yaw, self.current_pitch, self.current_roll = att
        return (self.current_yaw, self.current_pitch, self.current_roll)

    def start_attitude_polling(self, interval=0.5):
        self._running = True
        def _poll():
            while self._running:
                try:
                    self.get_attitude()
                except Exception:
                    pass
                time.sleep(interval)
        self._attitude_thread = threading.Thread(target=_poll, daemon=True)
        self._attitude_thread.start()

    def stop(self):
        self._running = False
        if self._attitude_thread:
            self._attitude_thread.join(timeout=2)
        try:
            self.set_gimbal_rotation(0, 0)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
