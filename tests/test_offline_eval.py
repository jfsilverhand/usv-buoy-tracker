import numpy as np
import pytest

from offline_eval import iou, letterbox, postprocess, preprocess


def test_iou():
    a = [0, 0, 10, 10]
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, [20, 20, 30, 30]) == 0.0
    assert iou(a, [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_letterbox_keeps_aspect_and_centers():
    img = np.zeros((720, 1280, 3), np.uint8)
    canvas, r, left, top = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert r == pytest.approx(0.5)
    assert (left, top) == (0, 140)
    assert canvas[0, 0, 0] == 114  # padding colour


def test_preprocess_tensor_layout():
    x, r, px, py = preprocess(np.full((720, 1280, 3), 255, np.uint8))
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def _raw(nc=5, n=8400):
    return np.zeros((1, 4 + nc, n), np.float32)


def _put(out, k, cx, cy, w, h, cls, score):
    out[0, :4, k] = (cx, cy, w, h)
    out[0, 4 + cls, k] = score


def test_postprocess_maps_box_back_to_frame():
    out = _raw()
    _put(out, 0, 320, 320, 100, 50, cls=2, score=0.9)   # in 640x640 letterbox space
    _put(out, 1, 100, 300, 20, 20, cls=0, score=0.2)    # below threshold
    dets = postprocess(out, r=0.5, px=0, py=140, w=1280, h=720, conf_th=0.4, iou_th=0.45)
    assert len(dets) == 1
    box, conf, cls = dets[0]
    assert cls == 2 and conf == pytest.approx(0.9)
    assert box == pytest.approx([540, 310, 740, 410])


def test_postprocess_nms_is_per_class():
    out = _raw()
    _put(out, 0, 320, 320, 100, 100, cls=1, score=0.9)
    _put(out, 1, 322, 320, 100, 100, cls=1, score=0.8)  # duplicate, same class -> suppressed
    _put(out, 2, 322, 320, 100, 100, cls=4, score=0.7)  # same place, other class -> kept
    dets = postprocess(out, r=1.0, px=0, py=0, w=640, h=640, conf_th=0.4, iou_th=0.45)
    assert sorted(d[2] for d in dets) == [1, 4]


def test_postprocess_empty():
    assert postprocess(_raw(), 1.0, 0, 0, 640, 640, 0.4, 0.45) == []
