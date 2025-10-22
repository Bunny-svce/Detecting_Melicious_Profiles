# app.py
from datetime import datetime
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
STATIC_IMG_FOLDER = os.path.join("static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper Functions ---


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_target_column(df):
    candidates = ['label', 'target', 'class',
                  'is_bot', 'bot', 'malicious', 'y']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: assume last column if it looks categorical
    last = df.columns[-1]
    if df[last].nunique() <= 10:
        return last
    return None


def preprocess_df(df, target_col):
    df = df.copy()
    thresh = int(0.6 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col].str.replace(
                r'[^0-9\.-]', '', regex=True), errors='coerce')

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    skew_thresh = 2.0
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if (X[col] > 0).all() and X[col].skew() > skew_thresh:
            X[col] = np.log1p(X[col])

    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=(y if y.nunique() > 1 else None))

    if y_train.dtype == 'O' or y_train.dtype.name == 'category':
        mapping = {v: i for i, v in enumerate(sorted(y_train.unique()))}
        y_train_mapped = y_train.map(mapping)
        y_test_mapped = y_test.map(mapping)
    else:
        mapping = None
        y_train_mapped = y_train
        y_test_mapped = y_test

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train_mapped)

    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1] if y_test_mapped.nunique(
        ) == 2 else clf.predict_proba(X_test)
    else:
        y_prob = clf.predict(X_test)

    acc = accuracy_score(y_test_mapped, y_pred)
    prec = precision_score(y_test_mapped, y_pred, average='binary' if y_test_mapped.nunique(
    ) == 2 else 'macro', zero_division=0)
    rec = recall_score(y_test_mapped, y_pred, average='binary' if y_test_mapped.nunique(
    ) == 2 else 'macro', zero_division=0)
    f1 = f1_score(y_test_mapped, y_pred, average='binary' if y_test_mapped.nunique(
    ) == 2 else 'macro', zero_division=0)

    cm = confusion_matrix(y_test_mapped, y_pred)

    roc_auc = None
    fpr = tpr = None
    if y_test_mapped.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test_mapped, y_prob)
        roc_auc = auc(fpr, tpr)

    return {
        'model': clf,
        'mapping': mapping,
        'X_test': X_test,
        'y_test': y_test_mapped,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }


def save_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_roc_curve(fpr, tpr, roc_auc, outpath):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_prediction_pie(y_pred, outpath):
    unique, counts = np.unique(y_pred, return_counts=True)
    labels = [str(u) for u in unique]
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Predicted Class Distribution')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        flash('Logged in (demo).', 'success')
        return redirect(url_for('upload_csv'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash('Registered (demo). You can log in now.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'], f"{timestamp}_{file.filename}")
            file.save(filepath)

            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin1')

            target = detect_target_column(df)

            # --- Prediction-only Mode ---
            if target is None:
                flash(
                    'No label column found — running in prediction-only mode.', 'warning')
                X = df.select_dtypes(include=['number']).fillna(0)
                preds = np.zeros(len(X))  # Dummy predictions
                pie_path = os.path.join(
                    STATIC_IMG_FOLDER, f'pie_{timestamp}.png')
                save_prediction_pie(preds, pie_path)

                metrics = {'accuracy': 0, 'precision': 0,
                           'recall': 0, 'f1': 0, 'auc': None}

                return render_template('results.html',
                                       pie_img=os.path.basename(pie_path),
                                       metrics=metrics)

            # --- Training Mode ---
            X, y, scaler = preprocess_df(df, target)
            results = train_and_evaluate(X, y)

            cm_path = os.path.join(STATIC_IMG_FOLDER, f'cm_{timestamp}.png')
            if results['mapping'] is not None:
                labels = [k for k, v in sorted(
                    results['mapping'].items(), key=lambda x: x[1])]
            else:
                labels = sorted(np.unique(results['y_test']))
            save_confusion_matrix(results['cm'], labels, cm_path)

            roc_path = None
            if results['fpr'] is not None:
                roc_path = os.path.join(
                    STATIC_IMG_FOLDER, f'roc_{timestamp}.png')
                save_roc_curve(results['fpr'], results['tpr'],
                               results['roc_auc'], roc_path)

            pie_path = os.path.join(STATIC_IMG_FOLDER, f'pie_{timestamp}.png')
            save_prediction_pie(results['y_pred'], pie_path)

            os.makedirs('models', exist_ok=True)
            model_path = os.path.join('models', f'model_{timestamp}.joblib')
            joblib.dump(
                {'model': results['model'], 'scaler': scaler, 'mapping': results['mapping']}, model_path)

            metrics = {
                'accuracy': results['acc'],
                'precision': results['prec'],
                'recall': results['rec'],
                'f1': results['f1'],
                'auc': results['roc_auc']
            }

            return render_template('results.html',
                                   cm_img=os.path.basename(cm_path),
                                   roc_img=(os.path.basename(roc_path)
                                            if roc_path else None),
                                   pie_img=os.path.basename(pie_path),
                                   metrics=metrics)

    return render_template('upload.html')


@app.route('/static/images/<path:filename>')
def static_images(filename):
    return send_from_directory(STATIC_IMG_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
