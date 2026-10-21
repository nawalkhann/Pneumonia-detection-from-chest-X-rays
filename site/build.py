"""Build the static project website into site/dist.

    python site/build.py --base https://example.com

On Vercel the base URL comes from the SITE_URL environment variable instead.

Each page body lives in site/pages/<name>.html. This script wraps it in the
shared layout (head, SEO tags, nav, footer), converts the chart and Grad-CAM
PNGs from outputs/ into resized WebP files, and writes sitemap.xml and robots.txt.
"""
import argparse
import datetime
import html
import json
import os
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DIST = SITE / "dist"
REPO_URL = "https://github.com/nawalkhann/Pneumonia-detection-from-chest-X-rays"
AUTHOR = "Nawal"
SITE_NAME = "Pneumonia Detection from Chest X-Rays"

PAGES = [
    dict(slug="", src="index", nav="Overview",
         title="Pneumonia Detection from Chest X-Rays with Deep Learning | 99% Recall CNN",
         description="A fine-tuned EfficientNet-B0 CNN that detects pneumonia in chest X-rays with 99.0% recall and 0.975 ROC-AUC on the Kaggle test set, with Grad-CAM heatmaps that show where the model looks.",
         keywords="pneumonia detection, chest x-ray, deep learning, CNN, transfer learning, EfficientNet, Grad-CAM, medical imaging AI, recall, Kaggle chest x-ray dataset",
         priority="1.0"),
    dict(slug="results", src="results", nav="Results",
         title="Results: 99% Recall Pneumonia Classifier on 624 Test X-Rays",
         description="Test-set results for the pneumonia CNN: recall, specificity, precision, ROC-AUC, confusion matrices and precision-recall curves on 624 real chest X-rays, with an honest look at the trade-offs.",
         keywords="pneumonia classifier results, confusion matrix, ROC curve, precision recall curve, sensitivity specificity, chest x-ray model evaluation",
         priority="0.9"),
    dict(slug="grad-cam", src="grad-cam", nav="Grad-CAM",
         title="Grad-CAM Heatmaps for Pneumonia X-Rays: Where the CNN Looks",
         description="Grad-CAM explainability for a pneumonia detection CNN: heatmaps on real chest X-rays for caught cases, misses, false alarms and clear lungs, and what they reveal about shortcut features.",
         keywords="Grad-CAM, explainable AI, XAI, heatmap, chest x-ray, CNN interpretability, medical AI explainability, saliency map",
         priority="0.9"),
    dict(slug="method", src="method", nav="Method",
         title="Method: Transfer Learning with EfficientNet-B0 on Chest X-Rays",
         description="How the pneumonia model was built: Kaggle chest X-ray data, patient-level validation split, EfficientNet-B0 transfer learning, class weights, augmentation and a recall-tuned decision threshold.",
         keywords="transfer learning, EfficientNet-B0, fine-tuning CNN, PyTorch, timm, class imbalance, data augmentation, patient-level split, chest x-ray dataset",
         priority="0.8"),
    dict(slug="why-recall-matters", src="why-recall-matters", nav="Why recall",
         title="Why Recall Matters in Medical AI: Tuning a Pneumonia Screening Model",
         description="Why a pneumonia screening model should be optimized for recall instead of accuracy, how to pick a decision threshold on validation data, and what the false-alarm cost looks like in practice.",
         keywords="recall vs precision, sensitivity, medical AI, screening model, decision threshold, false negatives, class imbalance, accuracy paradox",
         priority="0.8"),
]

IMAGES = {
    # name: (source png, widths)
    "gradcam-example": ("outputs/gradcam_example.png", [520, 1040]),
    "gradcam-grid": ("outputs/gradcam_grid.png", [680, 1352]),
    "confusion-matrices": ("outputs/confusion_matrices.png", [690, 1380]),
    "roc-pr-curves": ("outputs/roc_pr_curves.png", [750, 1500]),
}

CSS = """
:root{--bg:#fbfaf7;--fg:#1d2430;--muted:#5a6473;--line:#e3e0d8;--card:#fff;--accent:#1b4f93;--accent-soft:#e8eef8;--good:#1f7a4d;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--fg:#e6e9ee;--muted:#a3adbb;--line:#2a313b;--card:#1a2029;--accent:#8db4ff;--accent-soft:#1d2a40;--good:#6fd09f;--warn:#f0b35a}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
a{color:var(--accent)}
a:hover{text-decoration-thickness:2px}
.wrap{max-width:860px;margin:0 auto;padding:0 16px}
header.site{border-bottom:1px solid var(--line);background:var(--card)}
header.site .wrap{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 16px;padding-top:12px;padding-bottom:12px}
.brand{font-weight:700;color:var(--fg);text-decoration:none;display:flex;align-items:center;gap:8px}
.brand svg{flex:none}
nav ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:4px 14px}
nav a{color:var(--muted);text-decoration:none;font-size:15px;padding:4px 0;display:inline-block}
nav a[aria-current=page]{color:var(--fg);font-weight:600;border-bottom:2px solid var(--accent)}
main{padding:32px 0 48px}
h1{font-size:clamp(28px,5vw,40px);line-height:1.2;margin:0 0 12px;letter-spacing:-.01em}
h2{font-size:24px;line-height:1.3;margin:40px 0 12px}
h3{font-size:19px;margin:28px 0 8px}
.lede{font-size:19px;color:var(--muted);margin:0 0 24px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat b{display:block;font-size:28px;line-height:1.2;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:14px}
figure{margin:24px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{color:#5a6473;font-size:14px;padding:8px 4px 2px}
.table-scroll{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;background:var(--accent-soft)}
.note{background:var(--accent-soft);border-radius:10px;padding:14px 16px;margin:20px 0}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}
.btn{display:inline-block;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;border:1px solid var(--accent)}
.btn.primary{background:var(--accent);color:var(--bg)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h3{margin:0 0 6px;font-size:17px}
.card p{margin:0;color:var(--muted);font-size:15px}
.card a{text-decoration:none}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:14px;line-height:1.5}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em}
footer.site{border-top:1px solid var(--line);color:var(--muted);font-size:14px;padding:24px 0 40px}
footer.site p{margin:4px 0}
.next{margin-top:40px;padding-top:16px;border-top:1px solid var(--line)}
"""

LOGO = ('<svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#1b4f93"/>'
        '<path d="M16 6v20M16 11c-3 0-7 2-8 8-.5 3 1 5 3 5 3 0 5-3 5-8M16 11c3 0 7 2 8 8 .5 3-1 5-3 5-3 0-5-3-5-8" '
        'fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>')


# Google Search Console HTML-tag verification code, set as an environment variable on Vercel.
_gsv = os.environ.get("GOOGLE_SITE_VERIFICATION", "").strip()
VERIFY = f'\n<meta name="google-site-verification" content="{html.escape(_gsv)}">' if _gsv else ""


def page_url(base, slug):
    return f"{base}/{slug}" if slug else f"{base}/"


def render(page, body, base, today):
    url = page_url(base, page["slug"])
    nav = "".join(
        f'<li><a href="/{p["slug"]}"{" aria-current=page" if p is page else ""}>{p["nav"]}</a></li>'
        for p in PAGES)
    ld = {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "WebSite", "@id": f"{base}/#website", "url": f"{base}/", "name": SITE_NAME,
             "author": {"@type": "Person", "name": AUTHOR, "url": "https://github.com/nawalkhann"}},
            {"@type": "TechArticle" if page["slug"] else "WebPage", "@id": f"{url}#page", "url": url,
             "headline": page["title"], "name": page["title"], "description": page["description"],
             "isPartOf": {"@id": f"{base}/#website"}, "inLanguage": "en", "dateModified": today,
             "image": f"{base}/img/og.jpg",
             "author": {"@type": "Person", "name": AUTHOR}},
        ],
    }
    if not page["slug"]:
        ld["@graph"].append({
            "@type": "SoftwareSourceCode", "name": SITE_NAME, "codeRepository": REPO_URL,
            "programmingLanguage": "Python", "runtimePlatform": "PyTorch",
            "description": page["description"], "author": {"@type": "Person", "name": AUTHOR}})
    title = html.escape(page["title"])
    desc = html.escape(page["description"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="keywords" content="{html.escape(page["keywords"])}">
<meta name="author" content="{AUTHOR}">
<meta name="robots" content="index,follow">{VERIFY}
<link rel="canonical" href="{url}">
<meta property="og:type" content="{"article" if page["slug"] else "website"}">
<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{base}/img/og.jpg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{base}/img/og.jpg">
<meta name="theme-color" content="#1b4f93">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon.ico" sizes="32x32">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>{CSS}</style>
<script type="application/ld+json">{json.dumps(ld, separators=(",", ":"))}</script>
</head>
<body>
<header class="site"><div class="wrap">
<a class="brand" href="/">{LOGO}<span>Pneumonia X-Ray AI</span></a>
<nav aria-label="Main"><ul>{nav}</ul></nav>
</div></header>
<main class="wrap">
{body}
</main>
<footer class="site"><div class="wrap">
<p>Built by {AUTHOR}. Code, trained model and metrics on <a href="{REPO_URL}">GitHub</a>.</p>
<p>Data: Kermany et al., <i>Cell</i> 2018, via the Kaggle Chest X-Ray Images (Pneumonia) dataset. A portfolio project, not a medical device.</p>
</div></footer>
</body>
</html>
"""


def build_images(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, (src, widths) in IMAGES.items():
        im = Image.open(ROOT / src).convert("RGB")
        for w in widths:
            h = round(im.height * w / im.width)
            im.resize((w, h), Image.LANCZOS).save(out / f"{name}-{w}.webp", "WEBP", quality=80, method=6)
    # Social preview: Grad-CAM example cropped to the 1200x630 ratio, on a light background.
    ex = Image.open(ROOT / "outputs/gradcam_example.png").convert("RGB")
    top = ex.crop((0, 0, ex.width, ex.height // 2))
    og = Image.new("RGB", (1200, 630), (251, 250, 247))
    scale = 630 / top.height
    top = top.resize((round(top.width * scale), 630), Image.LANCZOS)
    og.paste(top, ((1200 - top.width) // 2, 0))
    og.save(out / "og.jpg", quality=85, optimize=True, progressive=True)


def build_icons(out):
    svg = LOGO.replace(' width="24" height="24"', "").replace(' aria-hidden="true"', ' xmlns="http://www.w3.org/2000/svg"')
    (out / "favicon.svg").write_text(svg)
    # Raster icons drawn with PIL so no extra tooling is needed.
    from PIL import ImageDraw
    big = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.rounded_rectangle((0, 0, 179, 179), radius=40, fill=(27, 79, 147))
    w = 11
    d.line((90, 34, 90, 146), fill="white", width=w)
    d.arc((40, 62, 92, 142), 90, 300, fill="white", width=w)
    d.arc((88, 62, 140, 142), 240, 90, fill="white", width=w)
    big.convert("RGB").save(out / "apple-touch-icon.png", optimize=True)
    big.resize((32, 32), Image.LANCZOS).save(out / "favicon.ico", sizes=[(32, 32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("SITE_URL"), help="Public site URL, e.g. https://example.com")
    args = ap.parse_args()
    if not args.base:
        ap.error("pass --base or set SITE_URL")
    base = args.base.rstrip("/")
    today = datetime.date.today().isoformat()

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    build_images(DIST / "img")
    build_icons(DIST)

    for p in PAGES:
        body = (SITE / "pages" / f"{p['src']}.html").read_text()
        name = "index.html" if not p["slug"] else f"{p['slug']}.html"
        (DIST / name).write_text(render(p, body, base, today))

    notfound = dict(slug="404", nav="", title="Page not found | " + SITE_NAME,
                    description="This page does not exist.", keywords="")
    body404 = ('<h1>Page not found</h1><p class="lede">That page does not exist or has moved.</p>'
               '<p><a class="btn primary" href="/">Go to the overview</a></p>')
    (DIST / "404.html").write_text(
        render(notfound, body404, base, today).replace('content="index,follow"', 'content="noindex"'))

    urls = "".join(
        f"<url><loc>{page_url(base, p['slug'])}</loc><lastmod>{today}</lastmod><priority>{p['priority']}</priority></url>"
        for p in PAGES)
    (DIST / "sitemap.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')
    (DIST / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {base}/sitemap.xml\n")
    # The Grad-CAM page links the full-resolution grid.
    shutil.copy(ROOT / "outputs/gradcam_grid.png", DIST / "img/gradcam-grid.png")
    print(f"built {len(PAGES)} pages into {DIST} for {base}")


if __name__ == "__main__":
    main()
