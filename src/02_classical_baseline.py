"""
02_classical_baseline.py
=========================

MODULE 2 of the ground-up EEG deep learning roadmap: a classical
machine learning baseline for seizure detection, using hand-crafted
features (band power + line length) and a Random Forest classifier,
evaluated with Leave-One-Subject-Out (LOSO) cross-validation.

WHY THIS MODULE EXISTS
------------------------
Before touching any neural network, it's worth having a simple,
interpretable baseline. If a deep learning model can't beat this,
something is wrong with the deep learning approach -- this baseline
is also your sanity check for every module that comes after it.

NOTE ON CODE REUSE
--------------------
This script duplicates a few small functions from
01_data_loading.py (load_edf, preprocess_raw, parse_summary) instead
of importing them. Two reasons: filenames starting with a digit
("01_data_loading") can't be imported normally in Python without
extra workarounds, and keeping each numbered module fully
self-contained makes it easier to read on its own, GitHub-portfolio
style. Once the roadmap is further along, these shared pieces are
good candidates to move into a shared utils.py -- that refactor is a
real, common step in growing any codebase, just not needed yet.

WHERE TO GET REAL DATA
------------------------
Point ROOT_DATA_DIR at a folder containing multiple patient
subfolders downloaded from PhysioNet, e.g.:
    chb-mit/
        chb01/
            chb01-summary.txt
            chb01_03.edf
            chb01_04.edf
            ...
        chb02/
            ...

The more patients you have locally, the more meaningful the LOSO
result becomes -- with only one patient, LOSO cross-validation has
nothing to hold out, so this script needs at least 2.
"""
import os
import re
import glob
import numpy as np
import mne
from scipy.signal import welch

# NumPy renamed trapz -> trapezoid in version 2.0. This line picks
# whichever name is actually available, so the script works on either
# an older or newer NumPy install without you needing to know which.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mne.set_log_level("WARNING")  # suppress MNE info messages, keep warnings

# EEG frequency bands in Hz, used for band-power features
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

WINDOW_SEC = 10  # length of each epoch we classify, in seconds

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


def dedupe_and_rename_channels(raw):
    """
    MNE appends '-0', '-1', etc. to duplicate channel names on load
    (e.g. two channels both named T8-P8 become T8-P8-0 and T8-P8-1).
    This is a genuine, documented CHB-MIT quirk in some files -- not
    an error. We keep only the FIRST of each duplicated pair, rename
    it back to the plain name, and drop the rest. Without this,
    exact-name matching against COMMON_CHANNELS silently fails for
    every file that has a duplicate.
    """
    seen_base_names = set()
    channels_to_keep = []
    rename_map = {}

    for ch in raw.ch_names:
        base = re.sub(
            r"-\d+$", "", ch
        )  # strip a trailing "-<digit>" suffix, if MNE added one
        if base in seen_base_names:
            continue  # this is the second copy of a duplicate -- drop it
        seen_base_names.add(base)
        channels_to_keep.append(ch)
        if base != ch:
            rename_map[ch] = base

    raw.pick(channels_to_keep)
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def select_common_channels(raw):
    """
    Restrict `raw` to exactly COMMON_CHANNELS, in that fixed order.

    Raises a ValueError if any of them are missing (e.g. a file
    recorded with a different montage), so the caller can catch it
    and skip that one file instead of silently producing a
    differently-shaped feature vector.
    """
    missing = [ch for ch in COMMON_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"missing expected channels: {missing}")
    raw.pick(COMMON_CHANNELS)
    return raw


# ---------------------------------------------------------------------------
# Reused from Module 1 (see note above on why these are duplicated here)
# ---------------------------------------------------------------------------
def parse_summary(summary_path):
    with open(summary_path, "r") as f:
        text = f.read()
    chunks = text.split("File Name:")[1:]
    seizures_by_file = {}
    for chunk in chunks:
        filename = chunk.strip().splitlines()[0].strip()
        starts = re.findall(r"Seizure(?:\s\d+)?\sStart Time:\s*(\d+)\s*seconds", chunk)
        ends = re.findall(r"Seizure(?:\s\d+)?\sEnd Time:\s*(\d+)\s*seconds", chunk)
        seizure_windows = [(int(s), int(e)) for s, e in zip(starts, ends)]
        if seizure_windows:
            seizures_by_file[filename] = seizure_windows
    return seizures_by_file


def load_edf(edf_path):
    return mne.io.read_raw_edf(edf_path, preload=True)


def preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=60.0):
    raw.notch_filter(freqs=notch_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


# ---------------------------------------------------------------------------
# NEW: feature extraction
# ---------------------------------------------------------------------------
def extract_band_power(epoch_data, sfreq):
    """
    Compute average power in each canonical EEG band, per channel.

    Args:
        epoch_data: array: shape (n_channels, n_samples)
        sfreq: float: sampling frequency in Hz

    Returns:
        band_power: array: shape (n_channels * n_bands,)

    We use Welch's method (scipy.signal.welch) to estimate the power
    spectral density -- it's the standard, noise-robust way to turn a
    time-domain signal into "how much energy is at each frequency."
    """
    n_channels = epoch_data.shape[0]
    features = []
    for ch in range(n_channels):
        freqs, psd = welch(epoch_data[ch], fs=sfreq, nperseg=min(256, epoch_data.shape[1]))
        for band_name, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = _trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
            features.append(band_power)
    return np.array(features) 

def extract_line_length(epoch_data):
    """
    Line length per channel: sum of absolute differences between
    consecutive samples. A simple, classic seizure-detection feature --
    spikier signal (typical of seizures) produces a larger value.
    
    Args:
        epoch_data: array: shape (n_channels, n_samples)
        
    Returns:
        line_length: array: shape (n_channels,)
    """
    return np.sum(np.abs(np.diff(epoch_data, axis=1)), axis=1)  


def extract_features(epoch_data, sfreq):
    """Combine band power and line length into one feature vector."""
    return np.concatenate([extract_band_power(epoch_data, sfreq),
                           extract_line_length(epoch_data)])


# ---------------------------------------------------------------------------
# NEW: building the dataset across multiple patients
# ---------------------------------------------------------------------------
def build_dataset(root_data_dir):
    """
    Walk every patient subfolder under root_data_dir, extract one
    seizure epoch (label 1) per seizure and one non-seizure epoch
    (label 0) per patient, compute features for each, and return
    arrays ready for scikit-learn.

    Returns
    -------
    X      : array, shape (n_epochs, n_features)
    y      : array, shape (n_epochs,)          -- 1 = seizure, 0 = not
    groups : array, shape (n_epochs,)          -- patient ID per row,
                                                   used for LOSO splitting
    """
    X, y, groups = [], [], []

    patients_dirs = sorted(
        d for d in glob.glob(os.path.join(root_data_dir, "chb*")) if os.path.isdir(d)
        )

    for patient_dir in patients_dirs:
        patient_id = os.path.basename(patient_dir)
        summary_files = glob.glob(os.path.join(patient_dir, "*-summary.txt"))
        if not summary_files:
            continue  
        seizures_by_file = parse_summary(summary_files[0])

        all_eds = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        seizure_files = [f for f in all_eds if os.path.basename(f) in seizures_by_file]
        non_seizure_files = [f for f in all_eds if os.path.basename(f) not in seizures_by_file]

        # --- Positive examples: one epoch per seizure ---
        for edf_path in seizure_files:
            file_name = os.path.basename(edf_path)
            try:
                raw = preprocess_raw(load_edf(edf_path))
                raw = dedupe_and_rename_channels(raw)
                raw = select_common_channels(raw)
            except ValueError as e:
                print(f"Skipping {file_name} for patient {patient_id}: {e}")
                continue
            sfreq = raw.info["sfreq"]
            data = raw.get_data()  # shape (n_channels, n_samples)

            for start_sec, end_sec in seizures_by_file[file_name]:
                start_sample =  int(start_sec * sfreq)
                end_sample = min(int(start_sec * sfreq) + int(WINDOW_SEC * sfreq), data.shape[1])
                epoch = data[:, start_sample:end_sample]
                if epoch.shape[1] < sfreq:
                    continue  # skip if not enough samples for a full epoch
                X.append(extract_features(epoch, sfreq))
                y.append(1)
                groups.append(patient_id)

        # --- Negative examples: one epoch from a seizure-free file ---
        if non_seizure_files:
            edf_path = non_seizure_files[0]
            try:
                raw = preprocess_raw(load_edf(edf_path))
                raw = dedupe_and_rename_channels(raw)
                raw = select_common_channels(raw)
            except ValueError as e:
                print(f"Skipping {edf_path} for patient {patient_id}: {e}")
                raw = None
            if raw is None:
                continue
            sfreq = raw.info["sfreq"]
            data = raw.get_data()
            # Grab a window from partway into the file, avoiding the
            # very start where recording artifacts are more common.
            start_sample = int(60 * sfreq)
            end_sample = start_sample + int(WINDOW_SEC * sfreq)
            if end_sample <= data.shape[1]:
                epoch = data[:, start_sample:end_sample]
                X.append(extract_features(epoch, sfreq))
                y.append(0)
                groups.append(patient_id)

    return np.array(X), np.array(y), np.array(groups)


# ---------------------------------------------------------------------------
# NEW: Leave-One-Subject-Out cross-validation
# ---------------------------------------------------------------------------
def run_loso_cv(X, y, groups):
    """
    Train and evaluate a Random Forest using LOSO cross-validation:
    on each fold, one patient is held out entirely for testing, and
    the model trains on everyone else. This is repeated once per
    patient, so every patient is tested on exactly once.
    """
    logo = LeaveOneGroupOut()
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out_patient = groups[test_idx][0]  # all test indices have the same patient ID

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        # zero_division=0 avoids warnings when a fold has only one class present
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_results.append((held_out_patient, acc, prec, rec, f1))
        print(f"Fold {fold_i} (held out: {held_out_patient}):"
              f"acc={acc:.2f}, prec={prec:.2f}, rec={rec:.2f}, f1={f1:.2f}")

    accs = [r[1] for r in fold_results]
    print(f"\nMean accuracy across {len(fold_results)} folds: {np.mean(accs):.2f}")
    return fold_results


# ---------------------------------------------------------------------------
# DEMO / SELF-TEST using synthetic multi-patient data
# ---------------------------------------------------------------------------
def _build_fake_multi_patient_dataset(root_dir, n_patients=3):
    """
    Build a small fake dataset with several 'patients', each with one
    seizure file and one seizure-free file, purely to prove the full
    pipeline (loading -> features -> LOSO CV) runs without internet
    access. Not real EEG -- swap in a real downloaded CHB-MIT folder
    to get meaningful results.
    """
    channel_names = COMMON_CHANNELS  # use the full real channel set so "normal" fake patients pass the filter
    sfreq = 256.0
    rng = np.random.default_rng(seed=0)

    for p in range(1, n_patients + 1):
        patient_id = f"chb{p:02d}"
        patient_dir = os.path.join(root_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # --- seizure file ---
        n_samples = int(600 * sfreq)  # 10-minute fake recording
        data = rng.normal(scale=20e-6, size=(len(channel_names), n_samples))
        info = mne.create_info(channel_names, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        seizure_edf = os.path.join(patient_dir, f"{patient_id}_01.edf")
        mne.export.export_raw(seizure_edf, raw, fmt="edf", overwrite=True)

        # --- seizure-free file ---
        data2 = rng.normal(scale=20e-6, size=(len(channel_names), n_samples))
        raw2 = mne.io.RawArray(
            data2, mne.create_info(channel_names, sfreq, ch_types="eeg")
        )
        no_seizure_edf = os.path.join(patient_dir, f"{patient_id}_02.edf")
        mne.export.export_raw(no_seizure_edf, raw2, fmt="edf", overwrite=True)

        summary_text = (
            "Data Sampling Rate: 256 Hz\n"
            f"File Name: {patient_id}_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 200 seconds\n"
            "Seizure End Time: 230 seconds\n"
            f"File Name: {patient_id}_02.edf\n"
            "Number of Seizures in File: 0\n"
        )
        with open(os.path.join(patient_dir, f"{patient_id}-summary.txt"), "w") as f:
            f.write(summary_text)

    # One extra patient with a DIFFERENT montage (fewer channels),
    # to prove the common-channel filtering correctly skips it
    # instead of crashing the whole run -- mimicking real chb12.
    bad_patient_id = f"chb{n_patients + 1:02d}"
    bad_dir = os.path.join(root_dir, bad_patient_id)
    os.makedirs(bad_dir, exist_ok=True)
    weird_channels = channel_names[:10]  # missing most of COMMON_CHANNELS
    n_samples = int(600 * sfreq)
    data = rng.normal(scale=20e-6, size=(len(weird_channels), n_samples))
    raw = mne.io.RawArray(data, mne.create_info(weird_channels, sfreq, ch_types="eeg"))
    seizure_edf = os.path.join(bad_dir, f"{bad_patient_id}_01.edf")
    mne.export.export_raw(seizure_edf, raw, fmt="edf", overwrite=True)
    with open(os.path.join(bad_dir, f"{bad_patient_id}-summary.txt"), "w") as f:
        f.write(
            "Data Sampling Rate: 256 Hz\n"
            f"File Name: {bad_patient_id}_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 200 seconds\n"
            "Seizure End Time: 230 seconds\n"
        )


if __name__ == "__main__":
    ROOT_DATA_DIR = "E:\Ph.D\DB\CHB\chb-mit-scalp-eeg-database-1.0.0"  # <-- point this at your local CHB-MIT folder
    
    print("\n--- Building feature dataset across all patients ---")
    X, y, groups = build_dataset(ROOT_DATA_DIR)
    print(f"X shape: {X.shape}  (n_epochs, n_features)")
    print(f"y: {y}  (1 = seizure, 0 = non-seizure)")
    print(f"groups: {groups}")

    print("\n--- Running Leave-One-Subject-Out cross-validation ---")
    run_loso_cv(X, y, groups)

    print("\nAll steps ran successfully on synthetic data.")
    print(
        "Next: point ROOT_DATA_DIR at your real chb-mit/ folder "
        "(with multiple chbXX subfolders) to get a meaningful result."
    )
