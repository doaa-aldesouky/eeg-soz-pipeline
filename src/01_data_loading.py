"""
01_data_loading.py
===================

MODULE 1 of the ground-up EEG deep learning roadmap: loading and
preprocessing raw scalp EEG from the CHB-MIT Scalp EEG Database.

WHAT THIS SCRIPT DOES
----------------------
1. Parses a CHB-MIT `chbXX-summary.txt` file to find which .edf files
   contain seizures, and exactly when (in seconds from file start).
2. Loads a raw .edf recording with MNE-Python.
3. Cleans the signal: removes 60 Hz power-line noise (US grid) and
   band-pass filters to the 1-40 Hz range where seizure activity lives.
4. Cuts short "epochs" (fixed-length windows) around each seizure,
   instead of working with the whole multi-hour recording.

WHERE TO GET REAL DATA
------------------------
This dataset lives on PhysioNet, which this sandboxed environment
cannot reach. Download it yourself from:
    https://physionet.org/content/chbmit/1.0.0/

For a first run, grab just one patient's folder, e.g. chb01/:
    - chb01-summary.txt
    - chb01_03.edf   (this file is known to contain one seizure)

Put them in a local folder and update DATA_DIR below.

Because we can't fetch real files here, the __main__ block at the
bottom builds a small FAKE recording with the same shape and channel
names as a real CHB-MIT file, just so you can confirm the *logic*
runs correctly. Swap in real files once you've downloaded them.
"""

import os
import re
import numpy as np
import mne

# MNE is quite chatty by default (prints info on every call).
# We quiet it down here so our own print statements are easy to read.
mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# STEP 1: Parse the summary.txt file to find seizure timings
# ---------------------------------------------------------------------------
def parse_summary(summary_path):
    """
    Parse a CHB-MIT chbXX-summary.txt file.

    Returns
    -------
    dict mapping filename -> list of (seizure_start_sec, seizure_end_sec)
    e.g. {"chb01_03.edf": [(2996, 3036)], "chb01_04.edf": [(1467, 1494)]}

    Files with zero seizures simply won't have an entry (empty list).

    Why we do this ourselves instead of using a library: CHB-MIT's
    summary files are plain text, not a standard format, so a small
    hand-written parser is the normal, expected approach here.
    """
    with open(summary_path, "r") as f:
        text = f.read()

    # Split the file into per-recording chunks, each starting at "File Name:"
    chunks = text.split("File Name:")[1:]  # [0] is header info, skip it

    seizures_by_file = {}

    for chunk in chunks:
        # First line of each chunk is the filename itself
        filename = chunk.strip().splitlines()[0].strip()

        # Seizures can appear two ways in these files:
        #   "Seizure Start Time: 2996 seconds"          (single-seizure files)
        #   "Seizure 1 Start Time: 1724 seconds"         (multi-seizure files)
        starts = re.findall(r"Seizure(?:\s\d+)?\sStart Time:\s*(\d+)\s*seconds", chunk)
        ends = re.findall(r"Seizure(?:\s\d+)?\sEnd Time:\s*(\d+)\s*seconds", chunk)

        seizure_windows = [(int(s), int(e)) for s, e in zip(starts, ends)]
        if seizure_windows:
            seizures_by_file[filename] = seizure_windows

    return seizures_by_file


# ---------------------------------------------------------------------------
# STEP 2: Load a raw .edf recording
# ---------------------------------------------------------------------------
def load_edf(edf_path):
    """
    Load a single CHB-MIT .edf file as an MNE Raw object.

    preload=True reads the whole file into memory immediately, which is
    fine for these ~1 hour recordings and makes filtering/slicing easier.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    return raw


# ---------------------------------------------------------------------------
# STEP 3: Clean the signal
# ---------------------------------------------------------------------------
def preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=60.0):
    """
    Apply standard EEG cleaning steps, in place, and return the raw object.

    l_freq / h_freq : band-pass range in Hz.
        Below ~1 Hz is usually slow drift (sweat, electrode movement),
        not brain activity. Above ~40 Hz is dominated by muscle artifact
        for scalp EEG. Seizure-related activity lives comfortably inside
        this band, so we keep it and discard the rest.

    notch_freq : the electrical power-line frequency to remove.
        CHB-MIT was recorded in Boston, USA -> 60 Hz grid.
        (Note for later: if you ever preprocess your own clinical data
        from Egypt, this must change to 50 Hz -- Egypt's grid frequency.
        Using the wrong notch frequency silently leaves line noise in
        the signal, so this is a real, not cosmetic, parameter.)
    """
    raw.notch_filter(freqs=notch_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


# ---------------------------------------------------------------------------
# STEP 4: Cut epochs around seizures
# ---------------------------------------------------------------------------
def get_seizure_epochs(raw, seizure_windows, pre_sec=5, post_sec=5):
    """
    Extract fixed windows around each seizure as plain numpy arrays.

    Parameters
    ----------
    raw : mne.io.Raw
        The (already preprocessed) continuous recording.
    seizure_windows : list of (start_sec, end_sec)
        Seizure timing from parse_summary().
    pre_sec, post_sec : float
        Extra seconds to include before the seizure onset and after
        its end, so the model can eventually learn what the transition
        INTO and OUT OF a seizure looks like, not just the middle of it.

    Returns
    -------
    list of numpy arrays, each shaped (n_channels, n_samples)
    """
    sfreq = raw.info["sfreq"]  # samples per second, e.g. 256.0
    data = raw.get_data()      # shape: (n_channels, total_samples)

    epochs = []
    for start_sec, end_sec in seizure_windows:
        start_sample = int((start_sec - pre_sec) * sfreq)
        end_sample = int((end_sec + post_sec) * sfreq)

        # Guard against windows that would run off either end of the file
        start_sample = max(start_sample, 0)
        end_sample = min(end_sample, data.shape[1])

        epochs.append(data[:, start_sample:end_sample])

    return epochs


# ---------------------------------------------------------------------------
# DEMO / SELF-TEST using synthetic data
# ---------------------------------------------------------------------------
def _build_fake_chbmit_file(tmp_dir):
    """
    Build a small FAKE .edf file and matching summary.txt with the
    same channel names and sampling rate as real CHB-MIT data, purely
    so the functions above can be exercised without internet access.

    This is NOT real EEG -- it's random numbers standing in for signal,
    used only to prove the loading/parsing/filtering code runs without
    errors. Swap in a real downloaded file to see it work on real data.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    channel_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3",
        "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
        "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
    ]
    sfreq = 256.0
    duration_sec = 3600  # pretend this is a 1-hour recording, like real files
    n_samples = int(duration_sec * sfreq)

    rng = np.random.default_rng(seed=42)
    # Small-amplitude noise, roughly EEG-scale (microvolts -> volts for MNE)
    fake_data = rng.normal(scale=20e-6, size=(len(channel_names), n_samples))

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(fake_data, info)

    edf_path = os.path.join(tmp_dir, "chb01_03.edf")
    mne.export.export_raw(edf_path, raw, fmt="edf", overwrite=True)

    # Matching fake summary.txt with one seizure, same format as real CHB-MIT
    summary_text = (
        "Data Sampling Rate: 256 Hz\n"
        "File Name: chb01_03.edf\n"
        "File Start Time: 13:43:04\n"
        "File End Time: 14:43:04\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 2996 seconds\n"
        "Seizure End Time: 3036 seconds\n"
    )
    summary_path = os.path.join(tmp_dir, "chb01-summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    return edf_path, summary_path


if __name__ == "__main__":
    DATA_DIR = "E:\Ph.D\DB\CHB\chb-mit-scalp-eeg-database-1.0.0\chb01"  # <-- point this at real chb01/ once you have it

   summary_path = os.path.join(DATA_DIR, "chb01-summary.txt")
   edf_path = os.path.join(DATA_DIR, "chb01_03.edf")

    print("\n--- Step 1: parsing summary.txt ---")
    seizures_by_file = parse_summary(summary_path)
    print(seizures_by_file)

    print("\n--- Step 2: loading the .edf file ---")
    raw = load_edf(edf_path)
    print(f"Loaded {len(raw.ch_names)} channels at {raw.info['sfreq']} Hz, "
          f"{raw.n_times} total samples")

    print("\n--- Step 3: preprocessing (notch + band-pass filter) ---")
    raw = preprocess_raw(raw)
    print("Filtering complete.")

    print("\n--- Step 4: cutting seizure epochs ---")
    filename = os.path.basename(edf_path)
    epochs = get_seizure_epochs(raw, seizures_by_file[filename])
    for i, epoch in enumerate(epochs):
        print(f"Seizure epoch {i}: shape = {epoch.shape} "
              f"(n_channels, n_samples)")

    print("\nAll steps ran successfully on synthetic data.")
    print("Next: download a real chb01 folder from PhysioNet and point "
          "DATA_DIR at it to see this run on genuine EEG.")
