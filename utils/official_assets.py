import os


OFFICIAL_TRIS_CHECKPOINTS = {
    "stage2_refcocog_umd.pth": "https://github.com/fawnliu/storage/releases/download/v1.0.1/stage2_refcocog_umd.pth",
}

OFFICIAL_CLIP_CHECKPOINTS = {
    "RN50.pt": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
}


def _require_existing_file(destination, url, asset_label):
    if os.path.exists(destination):
        return destination
    raise FileNotFoundError(
        f"Missing {asset_label}: place the official file at '{destination}'. Download URL: {url}"
    )


def require_official_tris_checkpoint(filename, root):
    if filename not in OFFICIAL_TRIS_CHECKPOINTS:
        raise KeyError(f"Unsupported official checkpoint: {filename}")

    destination = os.path.join(root, filename)
    return _require_existing_file(destination, OFFICIAL_TRIS_CHECKPOINTS[filename], "TRIS checkpoint")


def require_official_clip_checkpoint(filename, root):
    if filename not in OFFICIAL_CLIP_CHECKPOINTS:
        raise KeyError(f"Unsupported official CLIP checkpoint: {filename}")

    destination = os.path.join(root, filename)
    return _require_existing_file(destination, OFFICIAL_CLIP_CHECKPOINTS[filename], "CLIP checkpoint")
