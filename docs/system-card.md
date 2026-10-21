# System card

## What it is

DefectIQ is a supervised image classifier that decides whether a product image
shows a defect, predicts the defect type, and highlights the influential image
region with Grad-CAM. It is packaged as a FastAPI service with an inspection log
and a Streamlit dashboard.

## Intended use

- Assist human inspectors by pre-screening product images on a production line.
- Triage defect types (scratch, dent, crack, stain) to route rework.
- Demonstrate an end-to-end CV deployment: training, evaluation, explainability,
  API, monitoring.

## Out of scope

- Fully autonomous rejection without human review.
- Safety-critical certification (CE, ISO 9001 audit evidence) — the model is a
  decision aid, not a certified measuring instrument.
- Products, cameras or lighting conditions not represented in training data.
- Pixel-level defect localisation; Grad-CAM is coarse attribution, not a mask.

## Data

- Demo corpus: 320 synthetic images generated deterministically by the
  repository (200 train / 50 val / 70 test).
- Real deployments must supply labelled images from the target line. Splits are
  verified with SHA-256 hashes; duplicates across splits are reported as leakage.
- Uploaded inspection images are stored only in the request flow; the database
  stores the SHA-256, dimensions, filename and prediction — not the image bytes.

## Model

- ResNet18 with ImageNet weights, a dropout + linear head, class-weighted
  cross-entropy, AdamW and cosine learning-rate decay.
- Eight epochs on CPU in ~100 s; best checkpoint selected on validation macro F1.
- Binary decision derives from `1 - P(good)` with a configurable threshold.
- Version string (`resnet18-YYYY.MM.DD-HHMMSS`) is stored with every prediction.

## Performance (demo test split, 70 images)

Binary: accuracy 0.986, precision 1.000, recall 0.982, F1 0.991, ROC AUC 0.996.
Five-way: accuracy 0.971, macro F1 0.971. Latency 21.8 ms/image on CPU (batch 1).
Details and limitations: [results.md](results.md).

## Human oversight

- The API returns a recommendation plus confidence; the dashboard adds a
  "manual inspection recommended" message for defective predictions.
- The operating threshold should be set by quality engineering from the cost of
  a missed defect versus the cost of a false alarm.
- Grad-CAM is provided so an inspector can sanity-check that the model looked at
  the defect rather than at a background artefact.

## Risks

1. **Distribution shift.** New lighting, lenses, product finishes or camera
   angles change the input distribution. Monitor defect rates and confidence;
   retrain on the new conditions.
2. **Uncalibrated confidence.** Softmax scores can be overconfident. Calibrate
   before quoting probabilities externally.
3. **Rare defect classes.** A class with few examples can be missed. Track
   per-class recall and add targeted data.
4. **Adversarial or accidental artefacts.** Labels, reflections or fixtures in
   the frame can influence predictions. Keep capture conditions fixed and
   review Grad-CAM on failures.
5. **No outcome loop yet.** Accuracy drift requires human-verified labels joined
   to prediction rows; the schema supports this, but the join is not implemented.

## Change control

- Any model, preprocessing or threshold change requires a fresh evaluation run
  and a comparison with the previous report.
- Keep the previous checkpoint for rollback; predictions record the exact model
  version that scored them.
- Re-run the split manifest check whenever new images are added.
