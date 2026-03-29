import mne
from mne.datasets import eegbci
from mne.decoding import CSP

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


# --------------------------------
# 1. Load EEG Data
# --------------------------------

subjects = [1, 2, 3, 4]
runs = [6, 10, 14]

raw_list = []

for subject in subjects:

    files = eegbci.load_data(subject, runs)

    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]

    raw = mne.concatenate_raws(raws)

    raw_list.append(raw)

raw = mne.concatenate_raws(raw_list)

# clean channel names (PhysioNet often has dots like C3..)
raw.rename_channels(lambda x: x.strip('.').upper())

print(raw)


# --------------------------------
# 2. Attach EEG Montage
# --------------------------------

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")


# --------------------------------
# 3. Bandpass Filter
# --------------------------------

raw.filter(7., 30., fir_design='firwin')


# --------------------------------
# 4. Extract Events
# --------------------------------

events, event_id = mne.events_from_annotations(raw)

print("Event IDs:", event_id)


# --------------------------------
# 5. Pick EEG Channels
# --------------------------------

picks = mne.pick_types(raw.info, eeg=True)


# --------------------------------
# 6. Create Epochs
# --------------------------------

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=0,
    tmax=2,
    picks=picks,
    baseline=None,
    preload=True
)

print(epochs)


# --------------------------------
# 7. Prepare Data
# --------------------------------

X = epochs.get_data()

y = epochs.events[:, -1]

# remove rest class
mask = y != event_id['T0']

X = X[mask]
y = y[mask]

print("EEG data shape:", X.shape)


# --------------------------------
# 8. CSP Feature Extraction
# --------------------------------

csp = CSP(n_components=6, log=True, norm_trace=False)


# --------------------------------
# 9. Machine Learning Pipeline
# --------------------------------

svm = SVC(kernel='linear')

pipeline = Pipeline([
    ('CSP', csp),
    ('Scaler', StandardScaler()),
    ('SVM', svm)
])


# --------------------------------
# 10. Stratified Cross Validation
# --------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=cv)

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))