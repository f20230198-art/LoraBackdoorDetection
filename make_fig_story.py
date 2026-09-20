"""
Render fig_story.html to a tightly-cropped vector PDF for the paper.

Why this exists: the figure used to be a browser "print to PDF" of an A4 page, which
carried a date header, a file:// footer, and ~40% dead space, and the paper cropped it
with \includegraphics[trim=...,clip]. The AAAI-27 author kit forbids that:

    "Your graphics must be sized appropriately outside of LaTeX
     (do not use the 'clip' or 'trim' command)."

So we render at exactly the figure's own size instead. The output PDF *is* the drawing:
no header, no footer, no margins, nothing to crop. The paper then uses plain

    \includegraphics[width=\columnwidth]{figures/fig_story.pdf}

Text stays vector (selectable, sharp at any zoom), and the JS-drawn canvases render
because we use a real headless browser.

Usage:
    python make_fig_story.py                 # writes fig_story.pdf next to the html
    python make_fig_story.py --measure       # re-measure the ink box and report the
                                             # page size to use (run after editing the
                                             # HTML, then update PAGE_W / PAGE_H below)
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "fig_story.html")
OUT = os.path.join(HERE, "fig_story.pdf")

# The .fig block is 660 CSS px wide; 690 px covers its content plus bottom padding.
# Verified with --measure: ink occupies (22, 25) to (638, 672) CSS px.
PAGE_W, PAGE_H = 660, 690

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

PRINT_CSS = """<style id="printfix">
  @page {{ size: {w}px {h}px; margin: 0; }}
  html,body{{background:#fff !important;margin:0 !important;padding:0 !important;}}
  .wrap{{padding:0 !important;display:block !important;}}
  .fig{{width:660px !important;max-width:660px !important;margin:0 !important;}}
  @media print{{
    html,body{{background:#fff !important;
               -webkit-print-color-adjust:exact;print-color-adjust:exact;}}
  }}
</style>
"""


def find_browser():
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    sys.exit("No Chrome/Edge found. Edit BROWSERS in this script.")


def build_print_html(w, h):
    """Write a temp copy of the HTML with print CSS injected at the top."""
    src = open(HTML, encoding="utf-8", errors="ignore").read()
    fd, path = tempfile.mkstemp(suffix=".html", dir=HERE, prefix="_figprint_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(PRINT_CSS.format(w=w, h=h) + src)
    return path


def url(path):
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def measure(browser):
    """Screenshot tall, find the ink bounding box, and report the page size to use."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        sys.exit("--measure needs Pillow:  pip install pillow")

    tmp_html = build_print_html(700, 1400)
    png = os.path.join(tempfile.gettempdir(), "_fig_measure.png")
    try:
        subprocess.run([browser, "--headless", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=700,1400",
                        f"--screenshot={png}", url(tmp_html)],
                       check=True, capture_output=True)
        im = Image.open(png).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        box = ImageChops.difference(im, bg).getbbox()
        if not box:
            sys.exit("page rendered blank")
        l, t, r, b = (v / 2 for v in box)  # scale factor 2 -> CSS px
        print(f"ink box (CSS px): left={l:.0f} top={t:.0f} right={r:.0f} bottom={b:.0f}")
        print(f"content         : {r-l:.0f} x {b-t:.0f}")
        print(f"\nsuggested PAGE_W, PAGE_H = 660, {int(b) + 18}")
    finally:
        for p in (tmp_html, png):
            if os.path.exists(p):
                os.remove(p)


def render(browser):
    tmp_html = build_print_html(PAGE_W, PAGE_H)
    try:
        subprocess.run([browser, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer", f"--print-to-pdf={OUT}",
                        url(tmp_html)], check=True, capture_output=True)
    finally:
        if os.path.exists(tmp_html):
            os.remove(tmp_html)

    if not os.path.exists(OUT):
        sys.exit("browser did not write the PDF")

    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    try:
        from pypdf import PdfReader
        bx = PdfReader(OUT).pages[0].mediabox
        w = float(bx.right) - float(bx.left)
        h = float(bx.top) - float(bx.bottom)
        print(f"page: {w:.0f} x {h:.0f} pt = {w/72:.2f} x {h/72:.2f} in")
    except ImportError:
        pass
    print("\nCopy to figures/fig_story.pdf in the Overleaf project. The paper uses:")
    print(r"  \includegraphics[width=\columnwidth]{figures/fig_story.pdf}")


if __name__ == "__main__":
    b = find_browser()
    if "--measure" in sys.argv:
        measure(b)
    else:
        render(b)
