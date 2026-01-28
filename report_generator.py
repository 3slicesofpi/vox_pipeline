from jinja2 import Environment, FileSystemLoader
import datetime
import json

with open('setup/config.json', 'r') as f:
    config:dict = json.load(f)
report_cfg = config.get('report', {}) 

import os
import webbrowser
# ------------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------------

with open('export.json', 'r') as f:
    context:dict[dict] = json.load(f)

images = [
    {"file": "imgs/iso.png",   "label": "Isometric"},
    {"file": "imgs/top.png",   "label": "Top View"},
    {"file": "imgs/left.png",  "label": "Left Side"},
    {"file": "imgs/right.png", "label": "Right Side"},
    {"file": "imgs/back.png",  "label": "Back Side"},
    {"file": "imgs/front.png", "label": "Front Side"}
]

context["images"] = images
# ------------------------------------------------------------------
# 2. RENDER HTML
# ------------------------------------------------------------------

env = Environment(loader=FileSystemLoader("."))
template = env.get_template("setup/report.html")
html_output = template.render(context)

# ------------------------------------------------------------------
# 3. WRITE TO FILE
# ------------------------------------------------------------------

nowtext = datetime.datetime.now().strftime('%Y_%m_%d')
html_file = f"Report_{nowtext}.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_output)

# ------------------------------------------------------------------
# 3. WRITE TO FILE
# ------------------------------------------------------------------

print(f"HTML file generated: {html_file}")

print(f"Warning: NotImplementedError: Report assumes default values for SI units: cm, kg")

if report_cfg.get("open_result_html", False):
    _path = os.path.abspath(html_file)
    webbrowser.open(f"file:///{_path}")