# Partial Face Recognition

A face recognition system that can identify people **even when part of the face is hidden** — for example by a mask, hand, scarf or the edge of the camera frame. It is built on a lightweight **MobileFaceNet** model trained with **ArcFace loss**, and includes a Streamlit **staff attendance app** that registers faces, recognises them from a webcam and logs attendance.

## Results

Evaluated on a held-out test set (200 identities):

| Test | Metric | Result |
|---|---|---|
| Verification (1:1 matching, 400 positive + 400 negative pairs) | Accuracy | **94.6%** |
| | AUC-ROC | **0.983** |
| | True positive rate / False positive rate | 94.0% / 4.75% |
| Identification (1:N matching, 48,394 samples) | Top-1 accuracy | **96.7%** |
| | Top-5 accuracy | **98.5%** |

Best validation accuracy during training: **84.4%** (50 epochs). Full training logs, ROC curve, t-SNE embedding plot and real-world partial-face tests are in `Model_PFR.ipynb`.

## How it works

1. **Occlusion-aware training** – a custom `RandomFaceOcclusion` transform randomly hides the top, middle, bottom, left or right third of each training face, so the model learns to recognise people from partial information.
2. **Data preparation** – the dataset is augmented and split into train / validation / test sets.
3. **Model** – MobileFaceNet (512-dimensional embeddings, small enough for real-time use) trained with ArcFace angular-margin loss (s = 64, m = 0.5) in PyTorch.
4. **Matching** – faces are compared using cosine similarity of their embeddings against a tuned threshold.
5. **Application** – `app.py` is a Streamlit app with three modes: *Register* (capture a face from the camera), *Recognize* (live recognition) and *View Attendance*. `dashboard.py` shows daily attendance and confidence charts. Embeddings and attendance records are stored in Supabase.

## Tech stack

Python · PyTorch · torchvision · OpenCV · scikit-learn · Streamlit · Supabase · pandas · Matplotlib / Seaborn · Jupyter

## Repository contents

| File | Description |
|---|---|
| `Model_PFR.ipynb` | Data preparation, training, evaluation and visualisation |
| `occ.py` | Custom random face-occlusion augmentation |
| `best_mobilefacenet.pth` | Best checkpoint (highest validation accuracy) |
| `mobilefacenet_final.pth` | Final-epoch checkpoint |
| `app.py` | Streamlit staff attendance app |
| `dashboard.py` | Attendance analytics dashboard |
| `requirements.txt` | Python dependencies |
| `Partial Facial Recognition.pptx` | Project presentation |
| `individual reflective report.docx` | Reflective report |

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects a `utils/` package (database connection, model loader, face utilities and recognition helpers) and a Supabase project with `staff_embeddings` and attendance tables. Supabase credentials should be supplied through environment variables, never committed to the repository.

## Author

**Omidu Dulen Jayasinghe** — BSc Applied Artificial Intelligence, University of Bradford
[GitHub](https://github.com/omidu2003) · [LinkedIn](https://www.linkedin.com/in/omidu-jayasinghe-88589a32b/)
