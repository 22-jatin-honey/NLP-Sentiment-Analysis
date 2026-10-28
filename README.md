# ReviewSense — Deep Learning NLP Sentiment Analyzer

A resume-ready NLP project that classifies a movie or product review as positive or negative using a **Bidirectional LSTM** neural network. It includes model training, evaluation, saved artifacts, and an interactive Streamlit interface.

## Features
- End-to-end training pipeline using the IMDB movie-review dataset.
- Text sequence padding and vocabulary-index mapping.
- Bidirectional LSTM model with embedding, dropout, and sigmoid output.
- Evaluation on a held-out test set (loss, accuracy, precision, recall).
- Interactive web app that displays the predicted sentiment and confidence.

## Project structure
```text
nlp_sentiment_dl/
├── app.py                 # Streamlit interface and inference
├── train.py               # Download data, train, evaluate, save artifacts
├── requirements.txt
├── README.md
└── artifacts/             # Created by train.py; not included in Git
    ├── bilstm_sentiment.keras
    ├── word_index.json
    └── test_metrics.json
```

## Run locally
Python 3.10–3.11 is recommended for the TensorFlow version range in `requirements.txt`.

1. Create and activate a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Train the model (internet is needed the first time to download IMDB data):
   ```bash
   python train.py
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

The first training run can take several minutes or longer depending on your hardware. CPU training works; a GPU is optional. Training creates the `artifacts/` directory and model files.

## Model overview
- **Input:** review text converted into integer token IDs and padded/truncated to 250 tokens.
- **Embedding:** learns a dense representation for each token.
- **Bidirectional LSTM:** captures context from both directions in a word sequence.
- **Dense + sigmoid:** outputs a score between 0 and 1; `>= 0.5` is classified as positive.
- **Loss:** binary cross-entropy; **optimizer:** Adam.

## Important notes
- The dataset is the pre-tokenized IMDB dataset distributed by Keras. The app uses its vocabulary to encode new text.
- The model is trained on movie reviews, so performance may be lower on sarcasm, mixed sentiment, short fragments, or specialized product domains.
- Report the actual test metrics printed by your own run on your resume; do not claim a result before training and evaluating the model.

## Resume bullets (customize after running)
- Built an end-to-end NLP sentiment-analysis application using TensorFlow/Keras and a Bidirectional LSTM to classify review text as positive or negative.
- Implemented text preprocessing, sequence padding, model training, and held-out evaluation using accuracy, precision, and recall.
- Deployed an interactive Streamlit interface for real-time predictions with confidence scores.

## Possible extensions
1. Compare LSTM, BiLSTM, and a TF-IDF baseline.
2. Add a confusion matrix and ROC-AUC to evaluation.
3. Deploy the app on Streamlit Community Cloud after adding the trained artifacts or a reproducible model-download step.
4. Add a small labeled custom dataset to test domain generalization.
