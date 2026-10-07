# Benchmarks

Measured on this PC. Source `samples/dashcam.mp4` (1920x1080, 29.97 fps). 320 timed frames after 20 warm-up frames, frames held in RAM so the decoder is not counted twice.

- GPU: **NVIDIA RTX A4000**
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- torch 2.6.0+cu124, OpenCV 5.0.0, Python 3.11.0

`cpu %` is this process against one core (so >100% means multiple threads). `gpu %` and VRAM are whole-device readings from `nvidia-smi`, sampled at 10 Hz, and include anything else using the GPU.

| stage | frames | FPS | mean ms | p50 | p95 | max | cpu % mean | gpu % mean | VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video decode (1920x1080 h264, CPU) | 320 | 455.9 | 2.19 | 2.19 | 2.35 | 2.76 | - | - | - |
| redact() mask + bottom crop | 320 | 971.8 | 1.03 | 1.02 | 1.13 | 1.35 | 105.7 | 0.3 | 2543.0 |
| classical stripe baseline (CPU, OpenCV) | 320 | 51.5 | 19.42 | 19.39 | 19.84 | 20.29 | 101.1 | 0.6 | 2543.0 |
| video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 640 | 320 | 122.5 | 8.16 | 7.91 | 9.67 | 12.31 | 99.4 | 30.2 | 2823.0 |
| video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 960 | 320 | 118.8 | 8.42 | 8.19 | 9.84 | 12.2 | 98.3 | 37.2 | 2859.0 |
| video loop + overlay: yolo11n-seg.pt @ cpu, imgsz 640 | 150 | 49.4 | 20.23 | 19.84 | 22.81 | 24.75 | 217.8 | 0.6 | 2859.0 |

## End-to-end live loop

- `video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 640` + decode = **10.35 ms/frame (96.6 FPS)** end to end, against targets of >= 10 FPS and < 300 ms.
- `video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 960` + decode = **10.61 ms/frame (94.3 FPS)** end to end, against targets of >= 10 FPS and < 300 ms.
- `video loop + overlay: yolo11n-seg.pt @ cpu, imgsz 640` + decode = **22.42 ms/frame (44.6 FPS)** end to end, against targets of >= 10 FPS and < 300 ms.

The classical baseline at ~19.6 ms is **slower than the neural model on the GPU** (~8 ms) while detecting nothing correctly, so it is not a cheap fallback either.

### Superseded: the milestone-1 numbers

An earlier report gave 49.8 FPS / 15.0 ms for `cuda:0, imgsz 640` and 11.1 FPS / 97.4 ms for CPU. Those are not comparable to the table above: they were measured on 1920x810 frames (`crop_top_fraction` was 0.15, now 0.40, so the model sees 1920x540), and they included decode, the HUD and video writing in the timed section. Use this file, not those figures.

## Notes

- **video decode (1920x1080 h264, CPU)** — cv2.VideoCapture.read(), unavoidable in the live loop
- **redact() mask + bottom crop** — privacy step applied to every saved/processed frame
- **classical stripe baseline (CPU, OpenCV)** — redact + road band + detect. 0 geometric true positives: see BASELINE_RESULT.md
- **video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 640** — crop -> infer -> overlay. GENERIC COCO WEIGHTS, not a crosswalk model
- **video loop + overlay: yolo11n-seg.pt @ cuda:0, imgsz 960** — crop -> infer -> overlay. GENERIC COCO WEIGHTS, not a crosswalk model
- **video loop + overlay: yolo11n-seg.pt @ cpu, imgsz 640** — crop -> infer -> overlay. GENERIC COCO WEIGHTS, not a crosswalk model

## Not benchmarked yet

| model | status | why |
|---|---|---|
| **Trained crosswalk student** (`yolo11n-seg` fine-tuned) | **does not exist** | no training has been run; there are only 9 labelled-positive seconds, which is not a trainable dataset. See `docs/FOOTAGE_PLAN.md`. |
| **Mask2Former Swin-L Mapillary teacher** | **not downloaded** | weights are CC BY-NC 4.0 and not cleared; 826 MB. Gated behind `teacher.enabled: false`. See `LICENSING.md`. |
| **ONNX / TensorRT / OpenVINO exports** | **not built** | nothing worth exporting until a trained model exists. |

The `video loop + overlay` rows use the **generic COCO `yolo11n-seg`** fallback. They measure the pipeline's speed, not crosswalk accuracy — that model cannot detect a crosswalk at all. Treat them as an upper bound on what a same-size crosswalk student would cost.

Regenerate with `python scripts/benchmark.py`.
