# qt-deepx-example

A Qt5/QML application that demonstrates real-time object detection and person tracking on the
[i.MX 95 FRDM](https://www.nxp.com/design/design-center/development-boards/i-mx-evaluation-and-development-boards/i-mx-95-freedom-board:FRDM-IMX95)
board using the [DeepX](https://deepx.ai) NPU runtime and a YOLO26x model.

## Features

- **Object detection** — YOLO26x inference via the DeepX NPU runtime (`dxrt`), with NMS
  post-processing and configurable score/IoU thresholds.
- **Person tracking** — Persistent per-person IDs across frames using
  [cv::TrackerVit](https://docs.opencv.org/4.10.0/d9/d26/classcv_1_1TrackerVit.html)
  (OpenCV contrib). Each detected person (COCO label 0) gets a stable ID that survives
  temporary occlusions for up to 5 missed frames.
- **Live video** — GStreamer pipeline input (USB/CSI camera or any `v4l2src`), letterbox-scaled
  to the model's 640×640 input.
- **Status panel** — Right-side QML panel showing CPU load, memory usage and a cumulative
  *People Seen* counter, updated every second.
- **Multi-stream** — Up to two independent video streams displayed side-by-side.

## Architecture

```
main.cpp
├── QQmlApplicationEngine  — loads resources/qml/main.qml
├── SystemStats            — reads /proc/stat + /proc/meminfo every 1 s
├── PeopleCounter          — accumulates VideoStreamer::peopleSeen() signals
└── VideoStreamer (per stream, runs on its own QThread)
    ├── dxrt::InferenceEngine  — async NPU inference
    ├── Yolo::PostProc()       — decode + NMS → std::vector<BoundingBox>
    ├── PersonTracker          — TrackerVit pool, IoU matching, stable IDs
    └── DisplayBoundingBox / DisplayPersonTracks → QImage → QML
```

## Dependencies

| Dependency | Notes |
|---|---|
| Qt 5.12+ | Core, Gui, Quick, Qml, QuickControls2 |
| OpenCV 4.x | core, imgproc, videoio, **tracking** (contrib) |
| GStreamer | gstreamer-1.0, plugins-base/good/bad |
| DeepX RT (`dxrt`) | NPU inference SDK — only needed for `USE_DXRT` builds |
| `vitTracker.onnx` | Required at runtime for person tracking |

## Build

### Desktop (no DeepX hardware)

```bash
cmake -B build -DENABLE_DXRT=OFF -DENABLE_OPENCV=ON
cmake --build build --parallel
```

When built without `ENABLE_DXRT` the application renders a synthetic moving rectangle so the
QML/streaming pipeline can be validated without an NPU.

### Cross-compile for i.MX 95 (inside Avocado SDK container)

```bash
cmake -B build \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<toolchain>.cmake \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_BUILD_TYPE=Release \
  -DDXRT_DIR=${SDKTARGETSYSROOT}/usr \
  -DOpenCV_DIR=${SDKTARGETSYSROOT}/usr/lib/cmake/opencv4
cmake --build build --parallel
```

The [Avocado OS](https://avocadoos.io) build scripts in `files/dx-qt-example-compile.sh` handle
this automatically when building the `deepx-qt-example` extension.

## Usage

```
qt_deepx_example <model_path> \
    --labels <labels.json> \
    <gstreamer-pipeline> \
    [--tracker-model <vitTracker.onnx>] \
    [--score-threshold N]   (default: 0.25) \
    [--nms-threshold N]     (default: 0.45) \
    [-v]
```

### Arguments

| Argument | Description |
|---|---|
| `<model_path>` | Path to the compiled YOLO26x `.dxnn` model |
| `--labels <path>` | JSON file with a `"labels"` array of class names |
| `<pipeline>` | GStreamer pipeline string **or** a device path / file path |
| `--tracker-model <path>` | Path to `vitTracker.onnx`; omit to disable tracking |
| `--score-threshold N` | Minimum detection confidence (0–1, default 0.25) |
| `--nms-threshold N` | IoU threshold for non-maximum suppression (default 0.45) |
| `-v` | Enable verbose debug logging to stderr |

A second pipeline argument adds a second stream displayed alongside the first.

### Example — USB camera on i.MX 95

```bash
qt_deepx_example \
    /usr/local/lib/dx-models/yolo26x.dxnn \
    --labels /usr/local/lib/dx-models/labels.json \
    --tracker-model /usr/local/lib/qt-deepx-example/vitTracker.onnx \
    --score-threshold 0.7 \
    --nms-threshold 0.25 \
    "v4l2src device=/dev/video6 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! videoconvert ! appsink"
```

### Example — video file (desktop)

```bash
./build/qt_deepx_example \
    dummy_model \
    --labels labels.json \
    /path/to/video.mp4
```

## Person Tracking

When `--tracker-model` is supplied the application:

1. Filters YOLO detections to label 0 (`person`).
2. Updates all live `cv::TrackerVit` instances against the current frame.
3. Greedily matches updated tracks to new detections by IoU (threshold 0.3).
4. Re-initialises matched trackers to the confirmed detection position.
5. Creates a new tracker (new ID) for any unmatched detection.
6. Drops tracks that have been missed for more than 5 consecutive frames.

Each person is drawn with a **green** bounding box and an `ID: N` label. The YOLO detection box
is drawn separately in the class colour underneath. The *People Seen* counter in the status panel
counts unique IDs ever assigned (i.e. total persons seen, not just those currently visible).

The `vitTracker.onnx` model can be obtained from the
[opencv_extra](https://github.com/opencv/opencv_extra/blob/4.x/testdata/dnn/onnx/models/vitTracker.onnx)
repository.

## License

MIT — see [LICENSE](LICENSE).
