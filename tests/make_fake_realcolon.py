"""Generate a tiny synthetic REAL-Colon tree for end-to-end pipeline smoke tests.

    python tests/make_fake_realcolon.py /path/to/fake_real_colon
"""
import os
import sys

import numpy as np
from PIL import Image

VOC = """<annotation>
  <filename>{fn}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object><name>{lid}</name><bndbox>
    <xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>
  </bndbox></object>
</annotation>
"""


def main(root):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    vids = [f"{s:03d}-{v:03d}" for s in (1, 2) for v in (1, 2, 3)]
    W = H = 96
    vinfo, linfo = [], []
    for vi, vid in enumerate(vids):
        fdir = os.path.join(root, f"{vid}_frames")
        adir = os.path.join(root, f"{vid}_annotations")
        os.makedirs(fdir, exist_ok=True)
        os.makedirs(adir, exist_ok=True)
        n_frames = 10
        lesions = [f"{vid}_{j}" for j in (1, 2)]
        for j, lid in enumerate(lesions):
            hc = "AD" if (vi + j) % 2 == 0 else "HP"
            linfo.append(f"{lid},{vid},{3+j},colon,tubular,{hc}")
        for fidx in range(n_frames):
            fn = f"{vid}_{fidx:06d}.jpg"
            arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(fdir, fn), quality=90)
            if fidx % 2 == 0:  # half the frames carry a polyp box
                lid = lesions[fidx // 2 % len(lesions)]
                x0, y0 = int(rng.integers(5, 30)), int(rng.integers(5, 30))
                x1, y1 = x0 + int(rng.integers(20, 40)), y0 + int(rng.integers(20, 40))
                open(os.path.join(adir, f"{vid}_{fidx:06d}.xml"), "w").write(
                    VOC.format(fn=fn, w=W, h=H, lid=lid, x0=x0, y0=y0, x1=x1, y1=y1)
                )
        vinfo.append(f"{vid},55,M,brandX,25,{n_frames},{len(lesions)},2")

    open(os.path.join(root, "video_info.csv"), "w").write(
        "unique_video_name,age,sex,endoscope_brand,fps,num_frames,num_lesions,bbps\n"
        + "\n".join(vinfo) + "\n"
    )
    open(os.path.join(root, "lesion_info.csv"), "w").write(
        "unique_object_id,unique_video_name,size [mm],site,histology_extended,histology_class\n"
        + "\n".join(linfo) + "\n"
    )
    print(f"fake REAL-Colon at {root}: {len(vids)} videos")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_real_colon")
