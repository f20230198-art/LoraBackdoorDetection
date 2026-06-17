#!/usr/bin/env python3
"""
Copy a locally-generated adapter bank to Google Drive.

Use this when you generated to fast local disk (LBD_OUTPUT_BASE=/content/output_<model>)
to avoid slow per-file writes to the mounted Drive. Run it once after generation:

    !LBD_MODEL=qwen \
     LBD_OUTPUT_BASE=/content/output_qwen \
     python colab/sync_output_to_drive.py

It copies /content/output_<model>  ->  <project on Drive>/output_<model>.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("."))
import config  # noqa: E402

src = config.OUTPUT_BASE  # whatever LBD_OUTPUT_BASE pointed generation at
dst = f"output_{config.MODEL}"  # canonical location under the project (on Drive)

if os.path.abspath(src) == os.path.abspath(dst):
    sys.exit(
        f"src and dst are the same ({src}). Set LBD_OUTPUT_BASE to a local path "
        "(e.g. /content/output_qwen) so there is something to copy."
    )
if not os.path.isdir(src):
    sys.exit(f"Source not found: {src}. Did generation run with LBD_OUTPUT_BASE={src}?")

print(f"Copying {src}  ->  {dst}")
shutil.copytree(src, dst, dirs_exist_ok=True)
n = sum(len(files) for _, _, files in os.walk(dst))
print(f"Done. {dst} now holds {n} files.")
