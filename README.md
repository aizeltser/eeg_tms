# eeg_tms

The project is designed to process and analyze electroencephalography (EEG) data, focusing on evoked potentials in two groups of subjects: those who underwent transcranial magnetic stimulation (TMS) and a control group.

## Key features

- Automated analysis of ERP components: **N1**, **P2**, **MMN**, **N2**, **P300**, **N400**
- Group comparison:
  - TMS vs Control
  - Pre-TMS vs Post-TMS
- Multi-component visualization:
  - Topographic maps of potential distribution
  - ERP time series plots
  - Amplitude and latency matrices

## Technical implementation

The project is implemented in **Python**.

## Basic technologies

```python
import mne
import pandas as pd
import seaborn as sns
from scipy import stats

