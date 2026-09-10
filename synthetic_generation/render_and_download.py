"""
render_and_download.py

Renders one synthetic LeFusion case as a viewable overlay PNG and saves it
locally, ready to send on Discord. Connects to the already-deployed app --
run "modal deploy modal_lefusion.py" first if you haven't redeployed since
the last fix.

Usage:
    python render_and_download.py Case_P001.nii.gz
    python render_and_download.py Case_P001.nii.gz 5    (force a specific slice)
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python render_and_download.py <case_filename>")
    print("Example: python render_and_download.py Case_P001.nii.gz")
    sys.exit(1)

case_id = sys.argv[1]
slice_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

render_fn = modal.Function.from_name("lefusion-emidec-poc", "render_synthetic_case")

print(f"Rendering {case_id}...")
result = render_fn.remote(case_id=case_id, slice_idx=slice_idx)

if "error" in result:
    print(f"Error: {result['error']}")
    sys.exit(1)

print(f"Image shape: {result['image_shape']}")
print(f"Mask shape: {result['mask_shape']}")
print(f"Unique mask labels found: {result['unique_mask_labels']}")
print(f"Mask slice used: {result['mask_slice_used']}")
print(f"Image slice used: {result['image_slice_used']}")

img_h, img_w, _ = result['image_shape']
mask_h, mask_w, _ = result['mask_shape']
if (img_h, img_w) != (mask_h, mask_w):
    print(f"\nWARNING: spatial dimensions do NOT match -- image is {img_h}x{img_w}, "
          f"mask is {mask_h}x{mask_w}. This means the mask is very likely NOT properly "
          f"aligned with the image (different field of view / resolution), not just a "
          f"depth difference. Worth investigating before treating this as a valid pair.")
print(f"Mask slice used: {result['mask_slice_used']}")
print(f"Image slice used: {result['image_slice_used']}")

out_name = case_id.replace(".nii.gz", "") + "_overlay.png"
with open(out_name, "wb") as f:
    f.write(result["png_bytes"])

print(f"\nSaved: {out_name}")
print("Ready to attach to Discord.")
