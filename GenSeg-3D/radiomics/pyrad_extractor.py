from typing import Dict, List, Tuple, Optional

import torch


def _lazy_imports():
    import SimpleITK as sitk  # type: ignore
    from radiomics import featureextractor  # type: ignore
    return sitk, featureextractor


def resample_seg_to_image(segmentation, reference_image):
    sitk, _ = _lazy_imports()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    return resampler.Execute(segmentation)


def extract_all_features(image_path: str, mask_path: str, settings: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract the full set of PyRadiomics features, mirroring the notebook approach.
    Returns a dict: feature_name -> value (float).
    """
    sitk, featureextractor = _lazy_imports()
    img = sitk.ReadImage(image_path)
    seg = sitk.ReadImage(mask_path)
    seg = resample_seg_to_image(seg, img)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    if settings:
        extractor.enableAllFeatures()  # ensure full set; settings can enable filters/binWidth etc.
        for k, v in settings.items():
            extractor.settings[k] = v
    else:
        extractor.enableAllFeatures()

    feats = extractor.execute(img, seg)
    # Keep only numerical features
    out = {}
    for k, v in feats.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def align_feature_vectors(dicts: List[Dict[str, float]], names: Optional[List[str]] = None) -> Tuple[torch.Tensor, List[str]]:
    """
    Convert list of dicts to a tensor [N, F] aligned by feature names.
    If names not provided, use sorted union of keys.
    """
    if names is None:
        keys = set()
        for d in dicts:
            keys.update(d.keys())
        names = sorted(keys)
    rows = []
    for d in dicts:
        rows.append([float(d.get(k, 0.0)) for k in names])
    return torch.tensor(rows, dtype=torch.float32), names


def load_feature_importance(csv_path: str) -> Dict[str, float]:
    """
    Load feature importances CSV with columns: Feature,Importance
    Returns dict name->weight (normalized to mean=1 over provided entries).
    """
    import csv
    vals: Dict[str, float] = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Feature') or row.get('feature') or row.get('name')
            imp = row.get('Importance') or row.get('importance') or '1.0'
            if name is None:
                continue
            try:
                vals[name] = float(imp)
            except Exception:
                continue
    if not vals:
        return {}
    import numpy as np
    w = np.array(list(vals.values()), dtype=float)
    mean = w.mean()
    if mean > 0:
        for k in list(vals.keys()):
            vals[k] = vals[k] / mean
    return vals


