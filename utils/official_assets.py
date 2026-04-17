import os
import urllib.request


OFFICIAL_TRIS_CHECKPOINTS = {
    "stage2_refcocog_umd.pth": "https://github.com/fawnliu/storage/releases/download/v1.0.1/stage2_refcocog_umd.pth",
}


def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        return destination

    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return destination


def ensure_official_tris_checkpoint(filename, root):
    if filename not in OFFICIAL_TRIS_CHECKPOINTS:
        raise KeyError(f"Unsupported official checkpoint: {filename}")

    destination = os.path.join(root, filename)
    return download_file(OFFICIAL_TRIS_CHECKPOINTS[filename], destination)
