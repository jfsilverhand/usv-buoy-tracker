#!/usr/bin/env python3
"""
USV Otonom Hedef Takip Sistemi — DeepStream Pipeline
[rtspsrc | uridecodebin] → nvstreammux → nvinfer (YOLOv8 TRT, duba) → nvtracker (NvDCF)
                         → nvdsosd → [ximagesink | fakesink | H.264 mp4]
Probe callback (osd sink pad): hedef seç → PID → SiYi gimbal hız komutu

Kullanım:
  python3 src/ds_main.py                                   # RTSP kamera + ekran + gimbal
  python3 src/ds_main.py --src videos/x.mp4 --sink file    # kayıt video → eval/ds_out.mp4, gimbal yok
  python3 src/ds_main.py --src videos/x.mp4 --sink fake --max-frames 300 --no-gimbal   # headless smoke test

Ortam değişkenleri: RTSP_URI, GIMBAL_IP, GIMBAL_PORT, PGIE_CONFIG, CONF_THRESH
"""
import argparse
import os
import signal
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

sys.path.insert(0, os.path.dirname(__file__))
from siyi_gimbal import SIYIGimbalController
from pid_controller import PIDController

# ── Konfigürasyon ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTSP_URI        = os.environ.get("RTSP_URI",  "rtsp://192.168.2.25:8554/main.264")
GIMBAL_IP       = os.environ.get("GIMBAL_IP", "192.168.2.25")
GIMBAL_PORT     = int(os.environ.get("GIMBAL_PORT", "37256"))
PGIE_CONFIG     = os.environ.get("PGIE_CONFIG", os.path.join(BASE_DIR, "configs", "config_infer_primary_yolov8.txt"))
TRACKER_CONFIG  = os.path.join(BASE_DIR, "configs", "tracker_config.yml")
TRACKER_LIB     = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
LABELS_FILE     = os.path.join(BASE_DIR, "configs", "labels.txt")

STREAM_W, STREAM_H = 1280, 720
TARGET_CLASS_ID = -1      # -1 = herhangi bir duba; 0..4 = sadece o sınıf (configs/labels.txt sırası)
CONF_THRESH     = float(os.environ.get("CONF_THRESH", "0.40"))
STICKY_TARGET   = True    # True: kilitlenen tracker object_id görünür olduğu sürece hedef değişmez
                          # (eval/summary_sticky.json: hedef sıçraması 63 → 7)

PID_YAW_KP,   PID_YAW_KI,   PID_YAW_KD   = 35.0, 1.0, 5.0
PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD = 30.0, 0.8, 4.0
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="USV buoy tracker (DeepStream)")
    p.add_argument("--src", default=RTSP_URI, help="rtsp://... veya video dosyası yolu")
    p.add_argument("--sink", choices=["display", "fake", "file"], default="display")
    p.add_argument("--out", default=os.path.join(BASE_DIR, "eval", "ds_out.mp4"), help="--sink file çıktısı")
    p.add_argument("--max-frames", type=int, default=0, help="N kare sonra dur (0 = sınırsız)")
    p.add_argument("--no-gimbal", action="store_true", help="gimbal'a bağlanmayı deneme")
    p.add_argument("--pgie", default=PGIE_CONFIG, help="nvinfer config dosyası")
    p.add_argument("--tracker", default=TRACKER_CONFIG, help="nvtracker low-level config (A/B için)")
    return p.parse_args()


def load_labels(path):
    try:
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]
    except OSError:
        return ["kirmiziduba", "sariduba", "siyahduba", "turuncuduba", "yesilduba"]


def make_element(factory_name, element_name):
    el = Gst.ElementFactory.make(factory_name, element_name)
    if not el:
        raise RuntimeError(f"[ERROR] '{factory_name}' elementi oluşturulamadı — DeepStream kurulu mu?")
    return el


def request_pad(element, name):
    # GStreamer >= 1.20: request_pad_simple; eskisi: get_request_pad
    fn = getattr(element, "request_pad_simple", None) or element.get_request_pad
    return fn(name)


class BuoyTracker:
    def __init__(self, args):
        Gst.init(None)
        self.args     = args
        self.is_live  = args.src.startswith("rtsp://")
        self.loop     = GLib.MainLoop()
        self.pipeline = None
        self.labels   = load_labels(LABELS_FILE)
        self._fps_count  = 0
        self._fps_time   = time.time()
        self._frame_total = 0
        self._target_frames = 0
        self._class_hits = {}
        self._t_first = None
        self._target_ids = set()   # hedef olarak kilitlenen farklı tracker ID'leri
        self._locks = 0            # hedefsiz → hedefli geçiş sayısı
        self._switches = 0         # hedef A → hedef B doğrudan atlama sayısı
        self._eos_sent   = False
        self.current_fps = 0.0
        self.locked_id   = None   # sticky hedefin tracker object_id'si

        for p in (args.pgie, args.tracker):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Config bulunamadı: {p}")
        if not self.is_live and not os.path.exists(args.src):
            raise FileNotFoundError(f"Video bulunamadı: {args.src}")

        # Gimbal
        self.gimbal = None
        self.gimbal_ok = False
        if not args.no_gimbal:
            self.gimbal = SIYIGimbalController(GIMBAL_IP, GIMBAL_PORT)
            fw = self.gimbal.get_firmware_version()
            if fw:
                print(f"[OK] Gimbal bağlandı: {fw}")
                self.gimbal_ok = True
                self.gimbal.center_gimbal()
                time.sleep(0.5)
                self.gimbal.start_attitude_polling(interval=0.1)
            else:
                print("[WARN] Gimbal yanıt vermiyor — sadece tespit modu.")
        else:
            print("[INFO] Gimbal devre dışı (--no-gimbal).")

        # PID
        self.pid_y = PIDController(PID_YAW_KP,   PID_YAW_KI,   PID_YAW_KD)
        self.pid_p = PIDController(PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD)

        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ── Pipeline kurulumu ─────────────────────────────────────────────────────

    def _build_source(self, pipeline, mux):
        """Kaynağı kurar ve mux sink_0'a bağlar. RTSP: rtspsrc zinciri; dosya: uridecodebin."""
        mux_sink = request_pad(mux, "sink_0")
        if self.is_live:
            src = make_element("rtspsrc", "src")
            src.set_property("location", self.args.src)
            src.set_property("latency", 100)
            src.set_property("protocols", 4)  # TCP
            depay   = make_element("rtph264depay", "depay")
            parse   = make_element("h264parse",    "parse")
            decoder = make_element("nvv4l2decoder", "decoder")
            for el in (src, depay, parse, decoder):
                pipeline.add(el)
            src.connect("pad-added", self._on_rtsp_pad_added, depay)
            depay.link(parse)
            parse.link(decoder)
            decoder.get_static_pad("src").link(mux_sink)
        else:
            src = make_element("uridecodebin", "src")
            src.set_property("uri", "file://" + os.path.abspath(self.args.src))
            pipeline.add(src)
            src.connect("pad-added", self._on_decodebin_pad_added, mux_sink)

    def _build_sink(self, pipeline, osd):
        """osd çıkışını seçilen sink'e bağlar."""
        mode = self.args.sink
        if mode == "fake":
            sink = make_element("fakesink", "sink")
            sink.set_property("sync", 0)
            pipeline.add(sink)
            osd.link(sink)
            return

        vidconv2 = make_element("nvvideoconvert", "vidconv2")
        caps     = make_element("capsfilter", "capsfilter")
        pipeline.add(vidconv2); pipeline.add(caps)

        if mode == "display":
            # NVMM → CPU belleğe; ximagesink SHM kullanmaz, Docker/X11 uyumlu
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=(string)BGRx"))
            sink = make_element("ximagesink", "sink")
            sink.set_property("sync", 0)
            pipeline.add(sink)
            osd.link(vidconv2); vidconv2.link(caps); caps.link(sink)
            return

        # mode == "file": H.264 → mp4. Donanım encoder varsa onu, yoksa x264enc.
        os.makedirs(os.path.dirname(os.path.abspath(self.args.out)), exist_ok=True)
        enc = Gst.ElementFactory.make("nvv4l2h264enc", "enc")
        if enc:
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)I420"))
            enc.set_property("bitrate", 6000000)
            print("[INFO] Encoder: nvv4l2h264enc (GPU)")
        else:
            enc = make_element("x264enc", "enc")
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=(string)I420"))
            enc.set_property("speed-preset", "ultrafast")
            enc.set_property("tune", "zerolatency")
            print("[INFO] Encoder: x264enc (CPU)")
        parse = make_element("h264parse", "encparse")
        mux   = make_element("qtmux", "qtmux")
        sink  = make_element("filesink", "sink")
        sink.set_property("location", self.args.out)
        sink.set_property("sync", 0)
        for el in (enc, parse, mux, sink):
            pipeline.add(el)
        osd.link(vidconv2); vidconv2.link(caps); caps.link(enc)
        enc.link(parse); parse.link(mux); mux.link(sink)
        print(f"[INFO] Çıktı: {self.args.out}")

    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("usv-pipeline")

        mux = make_element("nvstreammux", "mux")
        mux.set_property("batch-size", 1)
        mux.set_property("width",  STREAM_W)
        mux.set_property("height", STREAM_H)
        mux.set_property("live-source", 1 if self.is_live else 0)
        mux.set_property("batched-push-timeout", 33000)
        pipeline.add(mux)

        self._build_source(pipeline, mux)

        pgie = make_element("nvinfer", "pgie")
        pgie.set_property("config-file-path", self.args.pgie)

        tracker = make_element("nvtracker", "tracker")
        tracker.set_property("ll-lib-file", TRACKER_LIB)
        tracker.set_property("ll-config-file", self.args.tracker)
        tracker.set_property("tracker-width",  640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)

        vidconv = make_element("nvvideoconvert", "vidconv")
        osd     = make_element("nvdsosd",        "osd")
        osd.set_property("process-mode", 0)
        osd.set_property("display-text", 1)

        for el in (pgie, tracker, vidconv, osd):
            pipeline.add(el)
        mux.link(pgie)
        pgie.link(tracker)
        tracker.link(vidconv)
        vidconv.link(osd)

        self._build_sink(pipeline, osd)

        osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._detection_probe, None)
        self.pipeline = pipeline

    @staticmethod
    def _on_rtsp_pad_added(element, pad, depay):
        sink_pad = depay.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and not caps.is_empty():
            name = caps.get_structure(0).get_name()
            if "video" not in name and "application/x-rtp" not in name:
                return
        pad.link(sink_pad)

    @staticmethod
    def _on_decodebin_pad_added(element, pad, mux_sink):
        caps = pad.get_current_caps() or pad.query_caps(None)
        name = caps.get_structure(0).get_name() if caps and not caps.is_empty() else ""
        if not name.startswith("video/"):
            return
        if mux_sink.is_linked():
            return
        if pad.link(mux_sink) != Gst.PadLinkReturn.OK:
            print("[ERROR] decodebin → nvstreammux bağlanamadı (NVMM buffer yok mu?)")

    # ── Hedef seçimi ──────────────────────────────────────────────────────────

    def _select_target(self, frame_meta):
        """Eşik + sınıf filtresi; STICKY_TARGET ise önce kilitli object_id aranır."""
        best, best_conf, locked = None, 0.0, None
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            ok = obj.confidence >= CONF_THRESH and (TARGET_CLASS_ID < 0 or obj.class_id == TARGET_CLASS_ID)
            if ok:
                if STICKY_TARGET and obj.object_id == self.locked_id:
                    locked = obj
                if obj.confidence > best_conf:
                    best, best_conf = obj, obj.confidence
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        target = locked if locked is not None else best
        new_id = target.object_id if target is not None else None
        if new_id != self.locked_id:
            if new_id is not None:
                self._target_ids.add(new_id)
                if self.locked_id is None:
                    self._locks += 1
                else:
                    self._switches += 1
            self.locked_id = new_id
            if new_id is None:
                self.pid_y.reset(); self.pid_p.reset()
        return target

    # ── Probe: tespit + gimbal ────────────────────────────────────────────────

    def _detection_probe(self, pad, info, _u):
        gst_buf = info.get_buffer()
        if not gst_buf:
            return Gst.PadProbeReturn.OK
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            best = self._select_target(frame_meta)

            if best is not None:
                r  = best.rect_params
                cx = r.left + r.width  / 2.0
                cy = r.top  + r.height / 2.0
                ex = (cx - STREAM_W / 2.0) / (STREAM_W / 2.0)
                ey = (cy - STREAM_H / 2.0) / (STREAM_H / 2.0)
                y_cmd = int(self.pid_y.compute(ex))
                p_cmd = int(self.pid_p.compute(-ey))
                if self.gimbal_ok:
                    try:
                        self.gimbal.set_gimbal_rotation(y_cmd, p_cmd)
                    except Exception:
                        pass
            elif self.gimbal_ok:
                try:
                    self.gimbal.set_gimbal_rotation(0, 0)
                except Exception:
                    pass

            # FPS / durum çıktısı (2 sn'de bir)
            now = time.time()
            if self._frame_total == 0:          # engine build süresi FPS'e sayılmasın
                self._t_first = now
                self._fps_time = now
            self._fps_count += 1
            self._frame_total += 1
            if best is not None:
                self._target_frames += 1
                self._class_hits[best.class_id] = self._class_hits.get(best.class_id, 0) + 1
            elapsed = now - self._fps_time
            if elapsed >= 2.0:
                self.current_fps = self._fps_count / elapsed
                self._fps_count  = 0
                self._fps_time   = now
                gimbal_str = ""
                if self.gimbal_ok:
                    gimbal_str = f" | Gimbal Y:{self.gimbal.current_yaw:+.1f} P:{self.gimbal.current_pitch:+.1f}"
                if best is not None:
                    label = self.labels[best.class_id] if best.class_id < len(self.labels) else "?"
                    target_str = f"TAKIP: {label} id={best.object_id} %{best.confidence*100:.0f}"
                else:
                    target_str = "ARAMA..."
                print(f"[INFO] frame {self._frame_total} | FPS: {self.current_fps:.1f} | {target_str}{gimbal_str}")

            if self.args.max_frames and self._frame_total >= self.args.max_frames and not self._eos_sent:
                self._eos_sent = True
                print(f"[INFO] {self.args.max_frames} kare işlendi, EOS gönderiliyor.")
                GLib.idle_add(lambda: self.pipeline.send_event(Gst.Event.new_eos()) and False)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ── Bus mesajları ─────────────────────────────────────────────────────────

    def _on_bus_message(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print(f"[INFO] Stream bitti ({self._frame_total} kare).")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[ERROR] {err.message}")
            if dbg:
                print(f"[DEBUG] {dbg}")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, _ = message.parse_warning()
            print(f"[WARN] {err.message}")
        return True

    def _on_signal(self, signum, frame):
        print("\n[INFO] Kapatılıyor...")
        if self.args.sink == "file" and not self._eos_sent:
            self._eos_sent = True
            self.pipeline.send_event(Gst.Event.new_eos())   # mp4 düzgün kapansın
        else:
            self.loop.quit()

    # ── Ana döngü ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  USV OTONOM HEDEF TAKIP SİSTEMİ — DeepStream")
        print("=" * 60)
        print(f"[INFO] Kaynak: {self.args.src} ({'canlı' if self.is_live else 'dosya'}) | Sink: {self.args.sink}")
        print(f"[INFO] Sınıflar: {' / '.join(self.labels)} | eşik {CONF_THRESH} | sticky={STICKY_TARGET}")
        print(f"[INFO] Tracker config: {self.args.tracker}")
        print("[INFO] Pipeline kuruluyor... (ilk çalıştırmada TRT engine derlenir, ~5 dk)")

        self._build_pipeline()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self.loop)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Pipeline başlatılamadı!")
            return 1

        print("[INFO] Pipeline çalışıyor. Çıkış için Ctrl+C.")
        try:
            self.loop.run()
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            print("[INFO] Temizlik yapılıyor...")
            self.pipeline.set_state(Gst.State.NULL)
            if self.gimbal_ok:
                try:
                    self.gimbal.set_gimbal_rotation(0, 0)
                    self.gimbal.stop()
                except Exception:
                    pass
            print("[INFO] Sistem kapatıldı.")
            self._print_summary()
        return 0 if self._frame_total > 0 else 2

    def _print_summary(self):
        n = self._frame_total
        if n == 0:
            print("[SUMMARY] frames=0")
            return
        dur = (time.time() - self._t_first) if self._t_first else 0.0
        fps = n / dur if dur > 0 else 0.0
        rate = self._target_frames / n
        per_cls = ", ".join(f"{self.labels[c] if c < len(self.labels) else c}={k}"
                            for c, k in sorted(self._class_hits.items()))
        print(f"[SUMMARY] frames={n} target_frames={self._target_frames} "
              f"target_rate={rate:.3f} avg_fps={fps:.1f} | {per_cls or 'no targets'}")
        print(f"[SUMMARY] tracker={os.path.basename(self.args.tracker)} distinct_target_ids={len(self._target_ids)} "
              f"locks={self._locks} switches={self._switches}")


if __name__ == "__main__":
    sys.exit(BuoyTracker(parse_args()).run())
