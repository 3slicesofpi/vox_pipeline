from jinja2 import Environment, FileSystemLoader
import datetime
import json
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
template = env.get_template("report.html")

html_output = template.render(context)

# ------------------------------------------------------------------
# 3. WRITE TO FILE
# ------------------------------------------------------------------

output_file = f"Report_{datetime.datetime.now().strftime('%Y_%m_%d')}.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"Report generated: {output_file}")
print(f"Warning: NotImplementedError: Report assumes default values for SI units: cm, kg")