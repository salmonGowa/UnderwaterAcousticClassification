# Decisions

One line per non-trivial decision, with the reason. Newest at the bottom. Evidence for the
data facts is in `docs/step0_data_audit.md`.

## Data and scope (2026-10-27)

- **Datasets: ShipsEar + DeepShip.** They are the two standard public vessel-noise sets and share some classes, which makes a cross-dataset test possible.
- **Task 2 (marine mammals) cut, approved by the project owner.** The Watkins site was under maintenance and its archive was behind a CAPTCHA, so its format, licence and IDs could not be verified.
- **Full ShipsEar and DeepShip audio come by email request.** That is the only access route either official page offers. Meanwhile the pipeline is developed on the 63 DeepShip files that are public on GitHub.
- **The GitHub DeepShip files are cut at about 200 s.** The file sizes show it: a Tug file is exactly 200.0 s. Results from them are for checking the pipeline only and are never reported.
- **Full DeepShip (about 21.7 GB) goes to Drive, gets resampled and cached, then the raw files are deleted (approved).** That is the only way to fit it within Colab and Drive limits.
- **Shared label map: passenger = DeepShip Passengership + ShipsEar Passengers and Ocean liner; cargo = DeepShip Cargo + ShipsEar RORO (approved).** Tanker is dropped because ShipsEar has none. Tug is not scored because ShipsEar has only 2 tug recordings. Ocean liner counts as passenger because DeepShip files a cruise ship (Celebrity Solstice, type code 60) as passenger.
- **No within-ShipsEar vessel-type experiment.** The paper's 5-class grouping could not be verified (the paper's server refused connections), and most ShipsEar types have fewer than 5 vessels. DeepShip carries the vessel-type task.
- **Vessel vs ambient is within ShipsEar only.** DeepShip has no ambient class, and mixing ambient from one dataset with vessels from the other would let a model learn which dataset a clip came from.

## Leakage control

- **ShipsEar vessel key = AIS MMSI where the listing has one, otherwise the ship name with suffixes like "(arrives)" removed.** The 30 Passenger recordings come from only 7 ships, so splitting by recording would leak.
- **ShipsEar manual merges taken from the listing's own notes:** "Moanha tests sample 1–4" are the dredger Adricristuy, and the two "Duda" entries are one motorboat.
- **Generic ShipsEar names ("Motorboat", "Sailboat", "Pilot ship") count as one vessel each.** That may merge different boats, which can only make the split stricter.
- **ShipsEar ambient recordings form one group each.** There is no vessel; all 12 come from one platform (see Limitations in the README).
- **DeepShip vessel = metafile ship name.** File ids restart inside some metafiles, so a WAV is matched to rows by id and by duration (the metafile duration must be at least the WAV duration). If several ships still match, they are merged into one group with union-find, so no candidate ship can sit on both sides of a split.
- **Splits: grouped, class-stratified 20% hold-out test, evaluated once, plus grouped 5-fold CV on the remaining 80%.** CV is for model comparison and run-to-run variance. The hold-out is the single untouched test.
- **`split_seed` is separate from the model seeds.** Every model and seed sees the same folds, so seed variance is purely model variance.

## Features

- **Resample everything to 16 kHz (project default).** This discards content above 8 kHz: 8–16 kHz for DeepShip (recorded at 32 kHz) and 8–26.4 kHz for ShipsEar (52,734 Hz). Vessel radiated noise is mostly below a few kHz, so this loss is probably small for Task 1. It would matter for many odontocete clicks and whistles, which is one more reason Task 2 needs a different rate.
- **25 ms window zero-padded to n_fft = 512, center=False.** The 400-sample window limits true frequency resolution to about 40 Hz. Padding to 512 only avoids empty low mel filters (a test checks for them). center=False makes each 4 s segment an exact slice of the whole-recording spectrogram, so we cache once per recording (tested).
- **A 40 Hz resolution blurs low-frequency tonals such as blade-rate harmonics.** A longer window is the first ablation to run if time allows.
- **Subtract each segment's mean log-mel level.** The hydrophones and gains are uncalibrated and differ between datasets, so absolute level mostly encodes distance and equipment, not vessel type.
- **MFCC baseline features: 20 DCT coefficients of the normalised log-mel; mean and std over time, plus mean and std of their deltas (80 values).** Simple, standard, and computed once into the cache.
- **Audio cache stored as float16 divided by the recording's peak (the peak is saved).** This keeps about 3 significant digits of relative precision and halves Drive use. It is used only for the noise-mixing test, and the log-mel cache is computed from the float32 audio.
- **Recordings shorter than 4 s produce no segments and drop out.** Padding would create artificial silence.

## Models

- **Baseline hyperparameters are fixed, not tuned:** LR with C = 1; RF with 300 trees and min_samples_leaf = 2. Tuning on this little data mostly fits noise, and fixed settings remove one way to overfit the evaluation.
- **All models use balanced class weights.** The classes are imbalanced, and the metrics are macro-averaged.
- **The CNN averages over time but keeps the frequency axis.** A tonal's frequency is informative; global pooling would make the model frequency-shift invariant.
- **CNN early stopping (patience 4, max 15 epochs) on a grouped 20% validation split carved from the training data only.** The test folds are never used for tuning.
- **Recompute BatchNorm statistics from up to 50 training batches before each evaluation.** The smoke test found a real bug: with few batches the running averages had not converged, giving 100% accuracy in train mode and 25% in eval mode. `tests/test_cnn.py` now guards this, and it fails without the fix.
- **Pretrained audio model deferred.** Decide after the CNN result, and check its licence first.

## Evaluation

- **Pre-registered rule: the CNN "clearly beats the baseline" only if its mean CV macro-F1 gain over the better of LR and RF exceeds the larger of the two models' run-to-run std (std across seeds of the per-seed CV mean).** Segment level is primary; recording level is also reported. Fold-to-fold std is reported separately.
- **Recording-level prediction = argmax of the mean segment probabilities.**
- **Bootstrap: 1000 resamples of recordings, stratified by class, percentile 95% CI.** Segments within a recording are correlated, so resampling segments would give intervals that are far too narrow. Stratifying keeps every class present.
- **Full reports and CIs use the first model seed; the other seeds' test scores are listed too.** One pre-registered seed avoids picking the best seed after seeing the test set.
- **Chance = expected score of a uniform random guess:** balanced accuracy 1/K, macro-F1 from the class prevalences. The majority-class model is reported as a separate baseline.
- **Noise robustness: ShipsEar's 12 real ambient recordings are mixed into DeepShip hold-out test clips at 20, 10, 5, 0 and −5 dB.** The pool comes from a different site and is never used for DeepShip training (asserted in code).
- **SNR here is clip RMS over added-noise RMS.** The clip already contains its own background noise, so this is not a true vessel-to-noise ratio.
- **The same noise chunks are used at every SNR, so the comparison is paired.** The "clean" point is recomputed from the cached 16 kHz audio through the same code path.
- **The cross-dataset test runs both ways and reports the drop from in-domain grouped CV.** ShipsEar→DeepShip trains on only 3 cargo vessels, so that direction is reported with that caveat.

## Environment

- **Pinned versions in pyproject.toml, except torch on Colab, where the preinstalled version is used.** Reinstalling torch on Colab is slow and can break CUDA. Every metrics.json records the versions actually used.
