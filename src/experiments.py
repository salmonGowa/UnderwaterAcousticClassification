"""Experiment runners. Each returns a JSON-serialisable dict of results.

within_dataset: grouped hold-out test set (touched once) + grouped k-fold CV on the rest.
cross_dataset:  train on all of one dataset, test on all of another, shared labels.
Noise robustness runs inside within_dataset on the hold-out test set.
"""

import time

import numpy as np
import torch

from src.config import data_path
from src.data import manifest as manifest_io
from src.evaluation import splits
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import aggregate_to_recordings, chance_levels, full_report, headline
from src.evaluation.noise import mix_at_snr, random_noise_chunk
from src.features.cache import load_audio16k, load_logmels, load_segments
from src.features.logmel import FeatureParams, log_mel, mfcc_summary
from src.models.baselines import DETERMINISTIC, make_baseline
from src.models.cnn import SegmentDataset, predict_proba, train_cnn


class TaskData:
    """Recordings, segments and features of one dataset after applying a task's label map."""

    def __init__(self, config, dataset, label_map, class_names, need_logmels):
        self.dataset = dataset
        self.root = config["data_root"]
        self.class_names = class_names
        self.params = FeatureParams.from_config(config["features"])
        manifest = manifest_io.load(data_path(config, "manifests", f"{dataset}.csv"))
        recordings = manifest_io.apply_label_map(manifest, label_map)
        recordings["y"] = recordings["target"].map({c: i for i, c in enumerate(class_names)})
        segments, mfcc = load_segments(config["data_root"], dataset)
        keep = segments["recording_id"].isin(recordings["recording_id"]).to_numpy()
        self.segments = segments[keep].reset_index(drop=True)
        self.mfcc = mfcc[keep]
        y_of = recordings.set_index("recording_id")["y"]
        self.segments["y"] = self.segments["recording_id"].map(y_of).to_numpy()
        # Recordings shorter than one segment have no segments; drop them everywhere.
        self.recordings = recordings[recordings["recording_id"].isin(self.segments["recording_id"])].reset_index(drop=True)
        self.logmels = load_logmels(self.root, dataset, self.recordings["recording_id"]) if need_logmels else None

    def mask(self, recording_ids):
        return splits.segment_mask(self.segments, recording_ids)


# ---------------------------------------------------------------- fitting and predicting

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_model(name, seed, data, train_ids, config):
    train_mask = data.mask(train_ids)
    if name != "cnn":
        model = make_baseline(name, seed, config.get("baseline_params", {}).get(name))
        model.fit(data.mfcc[train_mask], data.segments["y"].to_numpy()[train_mask])
        return model
    # CNN: early stopping on a grouped validation split carved out of the training recordings.
    train_recordings = data.recordings[data.recordings["recording_id"].isin(train_ids)]
    fit_ids, val_ids = splits.holdout_split(train_recordings, config["cnn"]["validation_fraction"], seed)
    splits.assert_no_group_overlap(data.recordings, fit_ids, val_ids)
    fit_set = _cnn_dataset(data, fit_ids)
    val_set = _cnn_dataset(data, val_ids)
    model, info = train_cnn(fit_set, val_set, len(data.class_names), config["cnn"], seed, device())
    print(f"    cnn seed={seed} best_epoch={info['best_epoch']} val_macro_f1={info['best_val_macro_f1']:.3f}", flush=True)
    return model


def _cnn_dataset(data, ids):
    mask = data.mask(ids)
    return SegmentDataset(data.logmels, data.segments[mask], data.segments["y"].to_numpy()[mask], data.params)


def _expand_proba(model, proba, n_classes):
    """sklearn only outputs columns for classes seen in training; put them in place."""
    full = np.zeros((len(proba), n_classes))
    full[:, model.classes_] = proba
    return full


def predict_clean(name, model, data, ids, config):
    """Class probabilities for every segment of the given recordings (in data.segments order)."""
    if name == "cnn":
        return predict_proba(model, _cnn_dataset(data, ids), config["cnn"]["batch_size"], device())
    return _expand_proba(model, model.predict_proba(data.mfcc[data.mask(ids)]), len(data.class_names))


def predict_arrays(name, model, mfcc, segment_logmels, n_classes, config):
    """Probabilities for features computed outside the cache (used for noisy test clips)."""
    if name != "cnn":
        return _expand_proba(model, model.predict_proba(mfcc), n_classes)
    model.eval()
    x = segment_logmels - segment_logmels.mean(axis=(1, 2), keepdims=True)   # same as normalise_segment
    x = torch.from_numpy((x / 20.0)[:, None].astype(np.float32))
    out = []
    with torch.no_grad():
        for batch in torch.split(x, config["cnn"]["batch_size"]):
            out.append(torch.softmax(model(batch.to(device())), dim=1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- scoring

def score(data, ids, proba):
    """Segment- and recording-level headline metrics for one prediction."""
    seg = data.segments[data.mask(ids)]
    rec_ids, rec_proba = aggregate_to_recordings(seg["recording_id"], proba)
    rec_true = data.recordings.set_index("recording_id").loc[rec_ids, "y"].to_numpy()
    n = len(data.class_names)
    return {"segment": headline(seg["y"].to_numpy(), proba.argmax(axis=1), n),
            "recording": headline(rec_true, rec_proba.argmax(axis=1), n)}


def full_evaluation(data, ids, proba, config):
    """Full reports plus bootstrap CIs over recordings, at both levels."""
    seg = data.segments[data.mask(ids)]
    seg_true, seg_pred = seg["y"].to_numpy(), proba.argmax(axis=1)
    rec_ids, rec_proba = aggregate_to_recordings(seg["recording_id"], proba)
    rec_true = data.recordings.set_index("recording_id").loc[rec_ids, "y"].to_numpy()
    rec_pred = rec_proba.argmax(axis=1)
    ci = bootstrap_ci(seg["recording_id"].to_numpy(), seg_true, seg_pred, rec_true, rec_pred, rec_ids,
                      len(data.class_names), config["evaluation"]["n_bootstrap"], seed=0)
    result = {}
    for level, (t, p) in {"segment": (seg_true, seg_pred), "recording": (rec_true, rec_pred)}.items():
        result[level] = full_report(t, p, data.class_names)
        result[level]["ci95"] = ci[level]
        result[level]["n"] = int(len(t))
    return result, {"segment_recording_ids": seg["recording_id"].tolist(),
                    "segment_index": seg["segment_index"].tolist(),
                    "segment_true": seg_true.tolist(), "segment_pred": seg_pred.tolist()}


def summarise_cv(rows):
    """rows: one dict per (seed, fold). Mean, fold std, and run-to-run (seed) std."""
    out = {}
    for level in ["segment", "recording"]:
        for metric in ["macro_f1", "balanced_accuracy"]:
            values = np.array([r[level][metric] for r in rows])
            seeds = sorted({r["seed"] for r in rows})
            per_seed = [np.mean([r[level][metric] for r in rows if r["seed"] == s]) for s in seeds]
            out[f"{level}_{metric}"] = {
                "mean": float(values.mean()),
                "std_over_folds": float(np.mean([np.std([r[level][metric] for r in rows if r["seed"] == s])
                                                 for s in seeds])),
                "std_over_seeds": float(np.std(per_seed)) if len(seeds) > 1 else 0.0,
                "per_seed_mean": [float(v) for v in per_seed],
            }
    return out


def clearly_beats(cv_summaries, candidate, baselines, key="segment_macro_f1"):
    """Pre-registered rule: gain over the best baseline must exceed the larger run-to-run std."""
    present = [b for b in baselines if b in cv_summaries]
    if candidate not in cv_summaries or not present:
        return None
    best = max(present, key=lambda b: cv_summaries[b][key]["mean"])
    gain = cv_summaries[candidate][key]["mean"] - cv_summaries[best][key]["mean"]
    threshold = max(cv_summaries[candidate][key]["std_over_seeds"], cv_summaries[best][key]["std_over_seeds"])
    return {"candidate": candidate, "best_baseline": best, "metric": key, "gain": float(gain),
            "threshold_run_to_run_std": float(threshold), "clearly_beats": bool(gain > threshold)}


def seeds_for(name, config):
    return [config["seeds"][0]] if name in DETERMINISTIC else list(config["seeds"])


# ---------------------------------------------------------------- experiments

def within_dataset(config, experiment):
    models = experiment["models"]
    data = TaskData(config, experiment["dataset"], experiment["label_map"], experiment["classes"],
                    need_logmels="cnn" in models)
    recs, split_seed = data.recordings, config["split_seed"]
    dev_ids, test_ids = splits.holdout_split(recs, config["evaluation"]["test_fraction"], split_seed)
    splits.assert_no_group_overlap(recs, dev_ids, test_ids)
    dev_recs = recs[recs["recording_id"].isin(dev_ids)]
    folds = splits.grouped_folds(dev_recs, config["evaluation"]["n_folds"], split_seed)
    for train_ids, val_ids in folds:
        splits.assert_no_group_overlap(recs, train_ids, val_ids)

    test_y = recs[recs["recording_id"].isin(test_ids)]["y"].to_numpy()
    test_seg_y = data.segments[data.mask(test_ids)]["y"].to_numpy()
    n = len(data.class_names)
    result = {
        "experiment": experiment["name"], "dataset": data.dataset, "classes": data.class_names,
        "split_unit": "vessel group (see manifest group_note)",
        "counts": {"dev": splits.class_counts(recs, dev_ids), "test": splits.class_counts(recs, test_ids),
                   "test_segments": int(len(test_seg_y))},
        "chance_test": {"segment": chance_levels(test_seg_y, n), "recording": chance_levels(test_y, n)},
        "cv": {}, "test": {}, "test_per_seed": {}, "noise": {},
    }
    final_models, predictions = {}, {}
    for name in models:
        print(f"[{experiment['name']}] {name}: {len(folds)}-fold CV", flush=True)
        start = time.time()
        rows = []
        for seed in seeds_for(name, config):
            for k, (train_ids, val_ids) in enumerate(folds):
                model = fit_model(name, seed, data, train_ids, config)
                rows.append({"seed": seed, "fold": k, **score(data, val_ids, predict_clean(name, model, data, val_ids, config))})
        result["cv"][name] = summarise_cv(rows)
        result["cv"][name]["folds"] = rows

        print(f"[{experiment['name']}] {name}: final fit on dev, evaluate on test", flush=True)
        final_models[name] = []
        result["test_per_seed"][name] = []
        for i, seed in enumerate(seeds_for(name, config)):
            model = fit_model(name, seed, data, dev_ids, config)
            final_models[name].append(model)
            proba = predict_clean(name, model, data, test_ids, config)
            result["test_per_seed"][name].append({"seed": seed, **score(data, test_ids, proba)})
            if i == 0:  # full report and CIs for the first (pre-registered) seed
                result["test"][name], predictions[name] = full_evaluation(data, test_ids, proba, config)
        result["cv"][name]["seconds"] = round(time.time() - start, 1)

    result["clearly_beats"] = {
        level: clearly_beats(result["cv"], "cnn", ["logreg", "random_forest"], key=f"{level}_macro_f1")
        for level in ["segment", "recording"]
    }
    if experiment.get("noise"):
        result["noise"] = noise_robustness(config, experiment["noise"], data, test_ids, final_models,
                                           exclude_ids=dev_ids | test_ids)
    return result, predictions


def noisy_features(data, test_ids, noise_pool, snr_db, seed):
    """Recompute features of every test segment after mixing in ambient noise (snr_db=None: clean)."""
    params = data.params
    rng = np.random.default_rng(seed)   # same seed at every SNR -> the same noise chunks (paired comparison)
    segments = data.segments[data.mask(test_ids)]
    logmels, mfccs, audio_cache = [], [], {}
    for row in segments.itertuples():
        if row.recording_id not in audio_cache:
            audio_cache = {row.recording_id: load_audio16k(data.root, data.dataset, row.recording_id)}
        start = row.segment_index * params.segment_hop_samples
        clip = audio_cache[row.recording_id][start:start + params.segment_samples]
        noise = random_noise_chunk(noise_pool, len(clip), rng)
        mixed = clip if snr_db is None else mix_at_snr(clip, noise, snr_db)
        lm = log_mel(mixed, params)
        logmels.append(lm)
        mfccs.append(mfcc_summary(lm, params.n_mfcc))
    return np.stack(mfccs), np.stack(logmels)


def noise_robustness(config, noise_config, data, test_ids, final_models, exclude_ids):
    pool_manifest = manifest_io.load(data_path(config, "manifests", f"{noise_config['pool_dataset']}.csv"))
    pool = pool_manifest[pool_manifest["label"].isin(noise_config["pool_labels"])]
    leaked = set(pool["recording_id"]) & exclude_ids
    if leaked:
        raise AssertionError(f"noise pool overlaps the experiment's recordings: {sorted(leaked)[:5]}")
    noise_pool = [load_audio16k(config["data_root"], noise_config["pool_dataset"], r) for r in pool["recording_id"]]
    out = {"pool_dataset": noise_config["pool_dataset"], "pool_recordings": sorted(pool["recording_id"]),
           "snr_db": [], "models": {name: [] for name in final_models}}
    for snr in [None] + list(noise_config["snr_db"]):
        print(f"[noise] SNR={snr}", flush=True)
        mfcc, logmels = noisy_features(data, test_ids, noise_pool, snr, noise_config["seed"])
        out["snr_db"].append("clean" if snr is None else snr)
        for name, models in final_models.items():
            per_seed = [score(data, test_ids, predict_arrays(name, m, mfcc, logmels, len(data.class_names), config))
                        for m in models]
            out["models"][name].append({
                level: {"macro_f1_mean": float(np.mean([s[level]["macro_f1"] for s in per_seed])),
                        "macro_f1_std": float(np.std([s[level]["macro_f1"] for s in per_seed]))}
                for level in ["segment", "recording"]
            })
    return out


def cross_dataset(config, experiment):
    """Train on the source dataset, test on the target, using the shared label map."""
    models = experiment["models"]
    src = TaskData(config, experiment["source"], experiment["source_label_map"], experiment["classes"], "cnn" in models)
    tgt = TaskData(config, experiment["target"], experiment["target_label_map"], experiment["classes"], "cnn" in models)
    src_ids, tgt_ids = set(src.recordings["recording_id"]), set(tgt.recordings["recording_id"])
    if src_ids & tgt_ids:
        raise AssertionError("source and target share recordings")
    n = len(src.class_names)
    tgt_seg_y = tgt.segments["y"].to_numpy()
    folds = splits.grouped_folds(src.recordings, config["evaluation"]["n_folds"], config["split_seed"])
    result = {
        "experiment": experiment["name"], "source": src.dataset, "target": tgt.dataset, "classes": src.class_names,
        "counts": {"source": splits.class_counts(src.recordings, src_ids),
                   "target": splits.class_counts(tgt.recordings, tgt_ids),
                   "target_segments": int(len(tgt_seg_y))},
        "chance_target": {"segment": chance_levels(tgt_seg_y, n),
                          "recording": chance_levels(tgt.recordings["y"].to_numpy(), n)},
        "in_domain_cv": {}, "cross": {}, "cross_per_seed": {}, "drop": {},
    }
    predictions = {}
    for name in models:
        print(f"[{experiment['name']}] {name}: in-domain grouped CV on {src.dataset}", flush=True)
        rows = []
        for seed in seeds_for(name, config):
            for k, (train_ids, val_ids) in enumerate(folds):
                model = fit_model(name, seed, src, train_ids, config)
                rows.append({"seed": seed, "fold": k, **score(src, val_ids, predict_clean(name, model, src, val_ids, config))})
        result["in_domain_cv"][name] = summarise_cv(rows)

        print(f"[{experiment['name']}] {name}: train on all {src.dataset}, test on {tgt.dataset}", flush=True)
        result["cross_per_seed"][name] = []
        for i, seed in enumerate(seeds_for(name, config)):
            model = fit_model(name, seed, src, src_ids, config)
            proba = predict_clean(name, model, tgt, tgt_ids, config)
            result["cross_per_seed"][name].append({"seed": seed, **score(tgt, tgt_ids, proba)})
            if i == 0:
                result["cross"][name], predictions[name] = full_evaluation(tgt, tgt_ids, proba, config)
        result["drop"][name] = {
            level: result["in_domain_cv"][name][f"{level}_macro_f1"]["mean"]
            - float(np.mean([s[level]["macro_f1"] for s in result["cross_per_seed"][name]]))
            for level in ["segment", "recording"]
        }
    return result, predictions
