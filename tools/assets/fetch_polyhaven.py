#!/usr/bin/env python3
"""Fetch Poly Haven assets (CC0) listed in data/assets/polyhaven.txt.

Each manifest line: <type> <asset_id> <resolution> [maps...]
  type = model  -> glTF + its textures
  type = texture -> listed maps as JPG (e.g. diff nor_gl arm)
Files land in data/raw/polyhaven/<asset_id>/ and are verified against Poly Haven's MD5s.
Poly Haven assets are CC0 (public domain): https://polyhaven.com/license
"""
import hashlib, json, os, sys, urllib.request

API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "phanerozoic-asset-fetch/1.0"}

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()

def fetch(url, md5, dest):
    if os.path.exists(dest) and hashlib.md5(open(dest, "rb").read()).hexdigest() == md5:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = get(url)
    if hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError(f"md5 mismatch for {url}")
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)
    return True

def main(manifest, root):
    for line in open(manifest):
        line = line.split("#")[0].split()
        if not line:
            continue
        kind, asset, res, *maps = line
        files = json.loads(get(API + asset))
        out = os.path.join(root, asset)
        todo = []
        if kind == "model":
            g = files["gltf"][res]["gltf"]
            todo.append((g["url"], g["md5"], os.path.join(out, os.path.basename(g["url"]))))
            for rel, v in g["include"].items():
                todo.append((v["url"], v["md5"], os.path.join(out, rel)))
        else:
            for m in maps:
                key = {"diff": "Diffuse", "nor_gl": "nor_gl", "arm": "arm", "rough": "Rough", "disp": "Displacement"}.get(m, m)
                f = files[key][res]["jpg"]
                todo.append((f["url"], f["md5"], os.path.join(out, f"{asset}_{m}_{res}.jpg")))
        n = sum(fetch(*t) for t in todo)
        print(f"{asset}: {len(todo)} files ({n} downloaded)", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
