# Underwater acoustic classification: vessel type and vessel vs ambient noise

A small, evaluation-first study of how well vessel noise in public hydrophone recordings can be
classified, and how far the results hold up on recordings, vessels and sites the model has not seen.

> **Status (2026-10-27): pipeline complete and tested; main results pending.** The full ShipsEar and
> DeepShip audio are only available by email request. Every number this README will quote comes from
> `results/main/summary.md`, produced by one command (below). Until then this README contains no results.

## Question and motivation

Shipping is the dominant source of low-frequency human-made noise in the ocean, and it overlaps the
frequencies many marine animals use to communicate. Soundscape monitoring needs to know *which* sounds
in a long recording come from vessels and roughly what kind, so that noise exposure can be estimated
for conservation and research. The question here is narrow:

1. **Vessel type** (DeepShip: cargo, passenger, tanker, tug).
2. **Vessel vs ambient noise** (ShipsEar).
3. **Does it generalise?** To vessels never seen in training, to another dataset recorded with different
   equipment at another site, and to added real ambient noise.

The scope is research and conservation monitoring only, not tracking or targeting individual vessels.

## Data

| | ShipsEar | DeepShip |
|---|---|---|
| Source | [underwaternoise.atlanttic.uvigo.es](https://underwaternoise.atlanttic.uvigo.es/) | [github.com/irfankamboh/DeepShip](https://github.com/irfankamboh/DeepShip) |
| Recordings / duration | 90 / 188.3 min | 611 / 47.07 h (from the official metafiles) |
| Site, period | Ría de Vigo, Spain, 2012–2014 | Strait of Georgia, Canada, 2016–2018 |
| Sample rate | 52,734 Hz, 24-bit | 32,000 Hz, float32 |
| Vessel identity | ship name, AIS MMSI for some | ship name in metafile |
| Licence | not stated | not stated |

How we verified each fact (with links) is in [docs/step0_data_audit.md](docs/step0_data_audit.md). No audio is
redistributed; [data/README.md](data/README.md) explains how to obtain it and verify checksums.

Citations: D. Santos-Domínguez et al., *Applied Acoustics* 113 (2016) 64–69, doi:10.1016/j.apacoust.2016.06.008.
M. Irfan et al., *Expert Systems with Applications* (2021), doi:10.1016/j.eswa.2021.115270.

## Method

- **Features** (computed once, cached): audio resampled to 16 kHz mono; 4 s segments with 50% overlap;
  64-bin log-mel (25 ms window, 10 ms hop); each segment's mean level is subtracted, because the
  hydrophones are uncalibrated and loudness mostly reflects distance and equipment.
- **Baselines**: 20 MFCCs summarised by mean and std (plus deltas) → logistic regression and random forest.
  Also a majority-class model and the chance level.
- **Main model**: a small 4-block CNN on log-mel segments that keeps the frequency axis (a tonal's
  frequency matters), with early stopping on a validation split carved from the training vessels only.
- **Pre-registered success criterion**: the CNN "clearly beats the baseline" only if its mean
  cross-validated macro-F1 gain over the better baseline exceeds the larger run-to-run (seed) std.
  This criterion is computed by the code and reported whichever way it falls.

Every non-trivial choice and its reason is logged in [DECISIONS.md](DECISIONS.md).

## Evaluation

**Leakage control.** Splits are by **vessel**, not by clip or recording. Many recordings show the
same ship (ShipsEar's 30 passenger recordings come from 7 ferries), so a recording-level split would
let a model recognise individual ships instead of vessel types. Vessel identity comes from AIS MMSI or
ship name. Where DeepShip metadata is ambiguous, all candidate ships are merged into one group, which
can only make the split stricter. Tests in [tests/](tests/) check that no vessel group or recording
appears on both sides of any split, including on the real ShipsEar listing.

**Protocol.** A grouped, stratified 20% hold-out test set is evaluated once. Grouped 5-fold
cross-validation on the remaining 80% is used for model comparison. Models are run with 3 seeds.

**Metrics.** Macro-F1 and balanced accuracy, per-class precision and recall, confusion matrices. All
are reported at segment level and at recording level (mean of segment probabilities). 95% confidence
intervals come from a bootstrap that resamples whole recordings, stratified by class.

**Generalisation tests.**
- *Cross-dataset*: train on DeepShip and test on ShipsEar (and the reverse) using a shared label map
  (passenger = Passengership / Passengers + Ocean liner; cargo = Cargo / RORO). The drop from in-domain
  CV is reported. Hydrophone, site, depth and era all differ between the datasets, so the drop cannot be
  attributed to any one of them.
- *Noise robustness*: real ambient recordings (ShipsEar, different site, never used for training) are
  mixed into DeepShip test clips at SNR = 20, 10, 5, 0, −5 dB (RMS-based), with a paired clean reference.

## Reproduce

```
pip install -e ".[torch,dev]"     # Colab: pip install -e ".[dev]" (keeps Colab's torch)
python -m pytest -q tests          # leakage, feature, metric and model tests
export UAC_DATA_ROOT=/path/to/data # see data/README.md
python -m src.run --config configs/main.yaml
```

`configs/smoke.yaml` runs the full pipeline on synthetic data in about 2 minutes on a CPU, and
`configs/deepship_github.yaml` runs it on the 63 public DeepShip files. Neither is reported as a result.
`notebooks/colab_runner.ipynb` runs the same commands on Colab.

## Results

*Pending the full datasets.* This section will contain, from `results/main/summary.md` only:
- DeepShip 4-class: CV and hold-out macro-F1 and balanced accuracy with 95% CIs, against the majority
  baseline and chance; the clearly-beats verdict; confusion matrices; macro-F1 vs SNR.
- ShipsEar vessel vs ambient, with the site caveat below.
- Cross-dataset drop in both directions.
- Failure analysis: the most-confused class pair, with example spectrograms
  (`results/main/*/failure_examples_*.png`). Acoustic explanations are given only as hypotheses.

## Limitations

- **Small data, few vessels.** DeepShip has 252 distinct ship names in its metafiles (README: 265 ships).
  ShipsEar has a handful of vessels per class: 7 passenger ferries, 4 ocean liners, 3 RORO ships, 2 tugs.
  Confidence intervals will be wide, and the cross-dataset cargo class rests on 3 ShipsEar ships.
- **One site per dataset.** Each dataset comes from a single region with one recording setup, so
  "generalises to a new vessel" is tested; "generalises to a new ocean" is only weakly tested by one
  cross-dataset pair.
- **Site confound in vessel vs ambient.** All 12 ShipsEar ambient recordings come from one platform
  (Cortegada), while the vessels were recorded in and around Vigo port. A model can partly separate
  *sites* rather than vessel presence, so this task's score is an upper bound.
- **Equipment and conditions.** Different hydrophones, gains, depths and sample rates. Level
  normalisation removes overall gain but not frequency-response differences.
- **Class imbalance.** Handled with balanced class weights and macro-averaged metrics, not resampling.
- **Label noise.** Labels are dataset-provided vessel categories. DeepShip's ship-to-file mapping is
  partly ambiguous (merged conservatively), other ships may be audible in the background (ShipsEar notes
  "interference" on some files), and some RORO ships also carry passengers.
- **Information discarded.** Resampling to 16 kHz removes content above 8 kHz, and the 25 ms window
  limits frequency resolution to about 40 Hz, which blurs low-frequency tonals.

## Claim check

The application states: *"evaluated across separate recordings and datasets so the results hold up
beyond the training data."*

- **Supported by design**: separation is by *vessel* (stricter than by recording), with tests, and
  there is a cross-dataset evaluation.
- **Gap to flag**: whether results "hold up" depends on the cross-dataset and noise numbers, which are
  not in yet. The cross-dataset test covers only 2 classes, with very few ShipsEar ships, and each dataset is
  a single site. A safer wording until the numbers are in: *"evaluated on vessels held out from training
  and on a second dataset recorded at a different site, to measure how much performance drops beyond
  the training data."*

## What I would do differently with guidance from ocean scientists

- Choose frequency resolution and band from vessel acoustics (shaft- and blade-rate tonals, cavitation
  band) rather than speech-processing defaults.
- Get calibrated recordings (known hydrophone sensitivity) so absolute received levels could be used.
- Use AIS tracks to label distance and speed per segment, and evaluate as a function of range.
- Define the classes that matter for noise-impact studies (e.g. by source level or size), not
  registry ship types.
- Validate on recordings from additional sites and seasons, and add expert-labelled ambient noise
  from the same sites as the vessels to remove the site confound.
