# Interview notes

## Dataset

**How did you collect the dataset?**
Real production data was not available, so the repository ships a deterministic
generator (`scripts/generate_demo_dataset.py`) that renders brushed-metal
surfaces and four defect types: scratch, dent, crack and stain. It is honest
about being synthetic, and the loader accepts any ImageFolder dataset with
`train/`, `val/`, `test/` and a `good` class, so the same codebase trains on a
customer's labelled images.

**How did you split train/test?**
Stratified by class: 40/10/14 images per class for train/val/test (200/50/70).
The split is written to `splits.json` with a SHA-256 per image, and a verifier
reports duplicates across splits. Augmentation runs only on the training split;
validation and test use a deterministic resize + centre crop.

**What is class imbalance?**
When classes have very different frequencies. On a real line, `good` dominates
and some defect types are rare. Accuracy then hides failures: predicting `good`
for everything can be 95% accurate. This project handles it with inverse
frequency class weights in the loss and per-class precision/recall/F1 in the
report, not just a global score.

**What is data leakage?**
Information from the test set influencing training. Classic vision examples: the
same image (or an augmented copy) appearing in train and test, frames from the
same video split across both, or normalising with statistics computed on the
full dataset. Here the manifest hash check catches duplicates, and augmentation
happens after the split.

## Model

**What is a CNN?**
A neural network that learns local patterns with shared filters. Convolution
layers detect edges and textures, pooling reduces spatial resolution, and deeper
layers compose those features into parts and objects. Weight sharing makes CNNs
far cheaper than a fully connected network on images.

**What is ResNet?**
A convolutional architecture with residual connections: each block learns a
residual `F(x)` and adds the input `x`, so gradients flow through skip
connections and very deep networks train without degrading. ResNet18 is the
18-layer variant used here.

**What is transfer learning?**
Starting from a model trained on a large dataset (ImageNet) and fine-tuning it on
a small target dataset. The early filters already detect generic edges and
textures; only the head (and later the whole network) needs to adapt. With 200
images this is the difference between a usable model and overfitting.

**What is an epoch?**
One full pass over the training set. Here each epoch is ~7 batches of 32 images;
eight epochs take about 100 seconds on CPU.

**What is backpropagation?**
The chain rule applied backwards through the network: compute the loss gradient
with respect to the output, propagate it layer by layer to obtain gradients for
every weight, then let the optimiser (AdamW) update the weights.

**Why PyTorch?**
Dynamic computation graphs make debugging and custom hooks (Grad-CAM) natural,
`torchvision` ships pretrained weights and transforms, and the same code runs on
CPU/GPU/MPS. The export path (TorchScript/ONNX) is mature as well.

## Evaluation

**Why is accuracy insufficient?**
On an imbalanced line, a constant "normal" prediction scores high while missing
every defect. Precision answers "of the units we rejected, how many were really
defective?" and recall answers "of the defective units, how many did we catch?".
The business cost decides the operating point.

**What is a false negative here?**
A defective unit classified as normal. It ships to the customer. In the current
test run there is one false negative out of 56 defective units (recall 0.982).

**How do you prevent overfitting?**
Transfer learning, augmentation (crop, flip, rotation, colour jitter), dropout in
the head, weight decay, early stopping on validation macro F1, and a held-out
test split that is never used for tuning. With tiny datasets, freeze the backbone
first so the pretrained filters are not destroyed by noisy early gradients.

**Confidence 0.99 - is that a calibrated probability?**
No. Softmax scores are usually overconfident, especially for tree/CNN models
trained on small data. Calibrate with temperature scaling or isotonic regression
on validation data before quoting probabilities to operators. Treat the score as
a ranking signal and the threshold as a business decision.

## Explainability

**What is explainable AI?**
Methods that make a model's decision inspectable by humans: attribution maps,
saliency, surrogate models, counterfactuals. In manufacturing the goal is trust
and faster root-cause analysis, not academic completeness.

**What is Grad-CAM?**
Gradient-weighted Class Activation Mapping. It takes the feature maps of the last
convolutional layer, weights each channel by the average gradient of the target
class score, sums them, applies ReLU, and upsamples the result. The output is a
heatmap of regions that increased the class score.

**Why is interpretability important in manufacturing?**
An inspector must decide whether to reject a part. A heatmap shows whether the
model looked at the actual defect or at a background artefact (lighting, fixture,
label). It also shortens debugging when a model fails on a new product variant.

**Is the heatmap a defect mask?**
No. It is coarse (7×7 feature grid upsampled to 224×224) and shows influence, not
segmentation. For pixel-accurate masks you need segmentation labels and a
segmentation model.

## Deployment and monitoring

**How is the model served?**
FastAPI loads the checkpoint once at startup and keeps it in `eval()` mode.
`/predict` validates the upload, applies the eval transform, runs one forward
pass, derives the binary decision and stores the inspection. Batches reuse a
single forward pass. The Streamlit dashboard talks to the API and falls back to
local inference.

**How do you optimise latency?**
Batch requests, avoid re-loading the model, use CPU-friendly eval transforms,
disable gradient computation, and pick a smaller backbone for edge hardware.
`scripts/benchmark.py` measures batch size vs throughput so the decision is data
driven.

**What do you monitor?**
Volume, normal/defective counts, defect-type mix, average/p95 latency, failed
predictions, model version and device. Accuracy drift needs verified outcomes
that arrive later; the schema is ready for a `verified_label` join.
