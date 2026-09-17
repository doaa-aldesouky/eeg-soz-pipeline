"""
03_cnn_classifier.py
=====================

MODULE 3 of the ground-up EEG deep learning roadmap: a basic 1D
Convolutional Neural Network that learns its own features directly
from raw EEG, instead of relying on hand-crafted ones (band power,
line length) like Module 2's classical baseline.

WHAT'S DIFFERENT FROM MODULE 2
---------------------------------
- Module 2 fed 108 hand-computed numbers per epoch into a Random
  Forest. This module feeds the RAW (n_channels, n_samples) signal
  directly into a small CNN, which learns its own filters.
- Every epoch must now be EXACTLY the same length (WINDOW_SEC seconds),
  since raw signals of different lengths can't be stacked into one
  batch tensor the way duration-invariant features could.
- Evaluated with the same Leave-One-Subject-Out (LOSO) cross-validation
  as Module 2, on purpose -- this gives a fair, apples-to-apples
  comparison against the classical baseline's result.

REUSED FROM MODULE 2 (see that file for detailed explanations)
------------------------------------------------------------------
parse_summary, load_edf, preprocess_raw, dedupe_and_rename_channels,
select_common_channels, COMMON_CHANNELS, extract_negative_epochs.
Duplicated here for the same reason as before: keeping each numbered
module runnable on its own.
"""
import os
import re
import numpy as np
import glob
import mne
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mne.set_log_level('WARNING')  # suppress MNE's verbose output

WINDOW_SEC = 10  # each epoch must be exactly this many seconds long
SFREQ_EXPECTED = 256.0
WINDOW_SAMPLES = int(
    WINDOW_SEC * SFREQ_EXPECTED
)  # every epoch must be exactly this many samples

COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]
N_CHANNELS = len(COMMON_CHANNELS)


# ---------------------------------------------------------------------------
# Reused from Module 1/2
# ---------------------------------------------------------------------------
def parse_summary(summary_path):
    with open(summary_path, "r") as f:
        text = f.read()
    chunks = text.split("File Name: ")[1:]
    seizures_by_file = {}
    for chunk in chunks:
        file_name = chunk.strip().splitlines()[0].strip()
        starts = re.findall(r"Seizure(?:\s\d+)?\sStart Time:\s*(\d+)\s*seconds", chunk)
        ends = re.findall(r"Seizure(?:\s\d+)?\sEnd Time:\s*(\d+)\s*seconds", chunk)
        if seizure_window:= [(int(start), int(end)) for start, end in zip(starts, ends)]:
            seizures_by_file[file_name] = seizure_window
    return seizures_by_file


def load_edf(edf_path):
    return mne.io.read_raw_edf(edf_path, preload=True)


def preprocess_raw(raw, l_freq=1.0, h_freq=4.0, notch_freq=60.0):
    raw.notch_filter(freqs=notch_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def dedupe_and_rename_channels(raw):
    seen_base_names = set()
    channels_to_keep = []
    rename_map = {}
    for ch in raw.ch_names:
        base = re.sub(r"-\d+$", "", ch)
        if base in seen_base_names:
            continue
        seen_base_names.add(base)
        channels_to_keep.append(ch)
        if base != ch:
            rename_map[ch] = base
    raw.pick_channels(channels_to_keep)
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def select_common_channels(raw):
    if missing_channels:= [ch for ch in COMMON_CHANNELS if ch not in raw.ch_names]:
        raise ValueError(f"Missing channels: {missing_channels}")
    raw.pick_channels(COMMON_CHANNELS)
    return raw


def load_and_clean(edf_path):
    """Convenience wrapper: load, filter, dedupe, and restrict to COMMON_CHANNELS in one call."""
    raw = preprocess_raw(load_edf(edf_path))
    raw = dedupe_and_rename_channels(raw)
    raw = select_common_channels(raw)
    return raw


def extract_negative_epochs(
    non_seizure_files, n_needed, window_sec, start_offset_sec=60, stride_sec=120
):
    epochs = []
    for edf_path in non_seizure_files:
        if len(epochs) >= n_needed:
            break
        try:
            raw = load_and_clean(edf_path)
        except ValueError as e:
            print(f"  Skipping {os.path.basename(edf_path)} — {e}")
            continue
        sfreq = raw.info["sfreq"]
        data = raw.get_data()
        total_samples = data.shape[1]

        offset_sec = start_offset_sec
        while len(epochs) < n_needed:
            start_sample = int(offset_sec * sfreq)
            end_sample = start_sample + int(window_sec * sfreq)
            if end_sample > total_samples:
                break
            epochs.append(data[:, start_sample:end_sample])
            offset_sec += stride_sec
    return epochs


# ---------------------------------------------------------------------------
# NEW: building a dataset of RAW (not hand-crafted-feature) epochs
# ---------------------------------------------------------------------------
def build_raw_dataset(root_data_dir):
    """
    Same overall structure as Module 2's build_dataset, but keeps the
    raw (n_channels, WINDOW_SAMPLES) signal instead of computing
    features from it. Every epoch is forced to EXACTLY WINDOW_SAMPLES
    long -- anything shorter (e.g. a seizure too close to a file's
    end) is dropped, since a CNN batch needs uniform shape.
 
    Returns
    -------
    X      : array, shape (n_epochs, n_channels, WINDOW_SAMPLES)
    y      : array, shape (n_epochs,)
    groups : array, shape (n_epochs,)  -- patient ID per row
    """
    X, y, groups = [], [], []
    
    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(root_data_dir, "chb*")) if os.path.isdir(d)
        )
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        summary_files = glob.glob(os.path.join(patient_dir, "*summary.txt"))
        if not summary_files:
            continue
        seizures_by_file = parse_summary(summary_files[0])
        
        all_edf = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        seizure_files = [f for f in all_edf if os.path.basename(f) in seizures_by_file]
        non_seizure_files = [f for f in all_edf if os.path.basename(f) not in seizures_by_file]
        
        X_patient, y_patient = [], []
        
        # --- Positive examples: one fixed-length window per seizure ---
        n_positive_this_patient = 0
        for edf_path in seizure_files:
            file_name = os.path.basename(edf_path)
            try:
                raw = load_and_clean(edf_path)
            except ValueError as e:
                print(f" Skipping {file_name} - {e}")
                continue
            sfreq = raw.info['sfreq']
            data = raw.get_data()
            
            for start_sec, end_sec in seizures_by_file[file_name]:
                start_sample = int(start_sec * sfreq)
                end_sample = start_sample + WINDOW_SAMPLES
                if end_sample > data.shape[1]:
                    continue     # too close to end of file for a full-length window
                epoch = data[:, start_sample:end_sample]
                if epoch.shape[1] != WINDOW_SAMPLES:
                    continue     # extra safety check -- must match exactly
                X_patient.append(epoch)
                y_patient.append(1)
                n_positive_this_patient += 1
                
        # --- Negative examples: match the positive count for this patient ---
        n_negative_needed = max(n_positive_this_patient, 1)
        for epoch in extract_negative_epochs(non_seizure_files, n_negative_needed, WINDOW_SEC):
            if epoch.shape[1] != WINDOW_SAMPLES:
                continue
            X_patient.append(epoch)
            y_patient.append(0)
            
        if not X_patient:
            continue
        
        # --- Subject-wise normalization (same reasoning as Module 2) ---
        # Z-score each patient's raw signal using only their own mean
        # and standard deviation, per channel, computed across their
        # own epochs. Corrects for between-patient amplitude
        # differences without using any other patient's data.
        X_patient = np.array(X_patient)     # shape: (n_patient_epochs, n_channels, WINDOW_SAMPLES)
        patient_mean = X_patient.mean(axis=(0, 2), keepdims=True)
        patient_std = X_patient.std(axis=(0, 2), keepdims=True) + 1e-10
        X_patient = (X_patient - patient_mean) / patient_std
        
        X.extend(X_patient)
        y.extend(y_patient)
        groups.extend([patient_id] * len(y_patient))
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups)


# ---------------------------------------------------------------------------
# NEW: the CNN itself
# ---------------------------------------------------------------------------
class SimpleEEGCNN(nn.Module):
    """
    A small 1D CNN for binary seizure classification.
 
    Architecture, in plain terms:
      raw signal -> [conv -> ReLU -> pool] x3, each stage using more
      filters but a shorter sequence -> global average pooling
      (collapses whatever length is left into one number per filter)
      -> one linear layer -> a single number (a "logit") indicating
      how confident the model is that this window is a seizure.
 
    Using global average pooling at the end (instead of flattening)
    means the network doesn't care about the exact sequence length,
    which is a nice property even though we do keep windows fixed-length
    here for simplicity.
    """
    def __init__(self, n_channels = N_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)     # collapse the time dimension to length 1
        )
        self.classifier = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape coming in: (batch_size, n_channels, WINDOW_SAMPLES)
        x = self.features(x)            # -> (batch_size, 64, 1)
        x = x.squeeze(-1)            # -> (batch_size, 64)
        x = self.classifier(x)       # -> (batch_size, 1)  -- one logit per example
        return x.squeeze(-1)         # -> (batch_size,)
    
    
# ---------------------------------------------------------------------------
# NEW: training and evaluating one LOSO fold
# ---------------------------------------------------------------------------
def train_one_fold(X_train, y_train, n_epochs=15, lr=1e-3, batch_size=16, device="cpu"):
    """
    Train a fresh SimpleEEGCNN from scratch on this fold's training
    patients. Returns the trained model.
    """
    model = SimpleEEGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun = nn.BCEWithLogitsLoss()       # combines a sigmoid + binary cross-entropy in one stable step
    
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch_i in range(n_epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()                    # clear gradients from the previous step
            logits = model(X_batch)                  # forward pass
            loss = loss_fun(logits, y_batch)         # how wrong were we?
            loss.backward()                          # backward pass -- compute gradients
            optimizer.step()                         # nudge weights to reduce the error
            
    return model


def run_loso_cv_cnn(X, y, groups, device="cpu"):
    """
    Same Leave-One-Subject-Out structure as Module 2's run_loso_cv,
    so results are directly comparable to the classical baseline.
    """
    logo = LeaveOneGroupOut()
    fold_results = []
    
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out_patient = groups[test_idx][0]
 
        model = train_one_fold(X_train, y_train, device=device)
        
        model.eval()
        with torch.no_grad():       # no need to track gradients for evaluation
            logits = model(torch.from_numpy(X_test).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()     # convert logits to 0-1 probabilities
            y_pred = (probs >= 0.5).astype(np.float32)      # threshold at 0.5 to get a hard label
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
        print(f"    [debug] predicted labels: {dict(zip(unique_preds, pred_counts))}")
        fold_results.append((held_out_patient, acc, prec, rec, f1))
        print(f"Fold {fold_i} (held out: {held_out_patient}): "
              f"acc={acc:.2f} prec={prec:.2f} rec={rec:.2f} f1={f1:.2f}")
 
    accs = [r[1] for r in fold_results]
    print(f"\nMean accuracy across {len(fold_results)} folds: {np.mean(accs):.2f}")
    return fold_results
 
 
# ---------------------------------------------------------------------------
# DEMO / SELF-TEST using synthetic multi-patient data
# ---------------------------------------------------------------------------
def _build_fake_multi_patient_dataset(root_dir, n_patients=3):
    sfreq = 256.0
    rng = np.random.default_rng(0)
 
    for p in range(1, n_patients + 1):
        patient_id = f"chb{p:02d}"
        patient_dir = os.path.join(root_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
 
        n_samples = int(600 * sfreq)
        data = rng.normal(scale=20e-6, size=(len(COMMON_CHANNELS), n_samples))
        raw = mne.io.RawArray(data, mne.create_info(COMMON_CHANNELS, sfreq, ch_types="eeg"))
        mne.export.export_raw(os.path.join(patient_dir, f"{patient_id}_01.edf"), raw, fmt="edf", overwrite=True)
 
        data2 = rng.normal(scale=20e-6, size=(len(COMMON_CHANNELS), n_samples))
        raw2 = mne.io.RawArray(data2, mne.create_info(COMMON_CHANNELS, sfreq, ch_types="eeg"))
        mne.export.export_raw(os.path.join(patient_dir, f"{patient_id}_02.edf"), raw2, fmt="edf", overwrite=True)
 
        with open(os.path.join(patient_dir, f"{patient_id}-summary.txt"), "w") as f:
            f.write(
                "Data Sampling Rate: 256 Hz\n"
                f"File Name: {patient_id}_01.edf\n"
                "Number of Seizures in File: 1\n"
                "Seizure Start Time: 200 seconds\n"
                "Seizure End Time: 230 seconds\n"
                f"File Name: {patient_id}_02.edf\n"
                "Number of Seizures in File: 0\n"
            )
 
if __name__ == "__main__":
    ROOT_DATA_DIR = "E:\Ph.D\DB\CHB\chb-mit-scalp-eeg-database-1.0.0"
    
    print("\n--- Building raw dataset across all patients ---")
    X, y, groups = build_raw_dataset(ROOT_DATA_DIR)
    print(f"X shape: {X.shape}  (n_epochs, n_channels, n_samples)")
    print(f"y: {y}")
    print(f"groups: {groups}")
 
    print("\n--- Running Leave-One-Subject-Out cross-validation (CNN) ---")
    run_loso_cv_cnn(X, y, groups)
 
    print("\nAll steps complete.")
    print("Next: point ROOT_DATA_DIR at your real chb-mit/ folder to train on genuine EEG.")
    
        
 
 