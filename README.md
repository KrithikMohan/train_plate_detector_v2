# License Plate Detector Training

This directory contains the training pipeline used to produce `plate_detector.tflite`, the on-device license plate detection model used by the Dashcam Utility Android app.

## Overview

- **Architecture:** YOLOv8n (Ultralytics), single class (`license_plate`)
- **Input size:** 640x640
- **Output format:** TensorFlow Lite (`.tflite`), used on-device via a raw TFLite `Interpreter` (no Task Library metadata)
- **Training environment:** Google Colab (free T4 GPU tier)

The trained model's job is limited to **localizing** license plates in a camera frame (drawing a bounding box). Reading the plate text is handled separately on-device via ML Kit OCR plus custom text validation — this repo only covers the detection model.

## Training data attribution

This model was trained on a combination of the following datasets. Copy this section wherever attribution needs to be surfaced (app "About" screen, etc.) alongside this README.

1. **Car License Plate** — MakeML / Dataset Ninja.
   License: CC0 1.0 (Public Domain), no attribution required.
   https://datasetninja.com/car-license-plate

2. **Licence plate** — Roboflow Universe.
   License: Public Domain, no attribution required.
   https://universe.roboflow.com/licence-plate-detection-wampn/licence-plate-shtqm

3. **License Plate Detection Dataset** — Fares Elmenshawii, Kaggle.
   License: CC0: Public Domain, no attribution required.
   https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset

4. **License Plate Recognition Dataset** — Adil Shamim, Kaggle (originally curated via Roboflow Universe).
   License: **CC BY 4.0 — attribution required.**
   https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition
   Used as part of a merged training set for a custom-trained model; no unmodified redistribution of the dataset itself occurs in this project.

Datasets 1–3 do not require attribution. Dataset 4 is CC BY 4.0 and is credited above per its license terms.

## Training pipeline

The notebook (`train_plate_detector.ipynb`) does the following, in order:

1. Extracts all four dataset archives.
2. Converts dataset 1 (Supervisely-format annotations) into YOLO-format labels.
3. Merges datasets 2–4 (already YOLO-format) into a single combined `images/` + `labels/` directory structure with `train`/`val` splits.
4. Trains a YOLOv8n model on the merged dataset (100 epochs, imgsz=640, batch=16).
5. Exports the trained weights to TensorFlow Lite.
6. Downloads the resulting `plate_detector.tflite` for use in the Android app (`app/src/main/assets/plate_detector.tflite`).

To retrain from scratch, upload the four dataset archives into a fresh Colab session and run the notebook top to bottom. To continue training from a previously trained checkpoint instead of starting from the base `yolov8n.pt` weights, upload that checkpoint and change the model-loading line in the training cell accordingly.

## Model

The trained model, `plate_detector.tflite`, is included in this repository (see the root/`model` directory).

Path expected by the Android app: `app/src/main/assets/plate_detector.tflite`.

To regenerate the model instead of using the committed copy, run `train_plate_detector.ipynb` end to end per the steps above.

## Known limitations

- **Single-class detector.** The model only localizes "a license plate is here" — it makes no distinction between jurisdictions, plate types, or vanity vs. standard plates. That distinction is left to the downstream text-validation layer.
- **Dataset size.** The merged training set is on the order of several thousand images, which is workable but below the ~10,000+ images-per-class Ultralytics generally recommends for maximum robustness. Detection quality may degrade on unusual angles, extreme lighting, or heavy occlusion.
- **Resolution-limited misses.** The detector (and any detector) cannot recover detail that was never captured — a plate that is genuinely too small, too distant, or too pixelated in the source frame may go undetected even though a human could read it. This is a capture-resolution limitation, not something more training data alone resolves.
- **Not evaluated against a held-out real-world dashcam test set.** Training/validation splits come from the merged public datasets. Real-world accuracy should be verified through in-app testing, not assumed from training metrics alone.

## License

Model weights and training code in this repository are licensed under the MIT License — see `LICENSE`. See "Training data attribution" above for the licensing terms governing the datasets used to produce the model.
