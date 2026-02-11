import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------
# Helpers
# ---------------------------

def build_manifest(container_id: int, L: float, W: float, H: float, packages: list[dict]) -> dict:
    """
    Build manifest.json EXACTLY like your visualiser expects.
    Matches keys and structure from your uploaded manifest.json.
    """
    return {
        "Container_ID": int(container_id),
        "Dimensions": {
            "Length": float(L),
            "Width": float(W),
            "Height": float(H),
        },
        "Packages": packages
    }


def validate_packages(packages: list[dict], container_dims: dict, payload_limit: float | None):
    """
    Basic validation checks (non-blocking warnings)
    """
    Lc = container_dims["Length"]
    Wc = container_dims["Width"]
    Hc = container_dims["Height"]

    total_weight = 0.0
    oversized = 0

    for p in packages:
        total_weight += float(p["Weight"])
        L = float(p["Dimensions"]["Length"])
        W = float(p["Dimensions"]["Width"])
        H = float(p["Dimensions"]["Height"])

        # Check if any dimension exceeds container (simple check; rotation not considered here)
        if L > Lc or W > Wc or H > Hc:
            oversized += 1

    return total_weight, oversized


def stack_positions(packages_raw: list[dict], z_start: float = 0.15, max_height: float | None = None):
    """
    Create simple placeholder positions:
    - All packages at x=0, y=0
    - z increases by package height (stacking vertically)
    This makes the output immediately visualisable.
    """
    z_cursor = z_start
    packages = []

    for p in packages_raw:
        H = float(p["Dimensions"]["Height"])
        if max_height is not None and (z_cursor + H) > max_height:
            # if stacking exceeds container height, keep it but warn later
            pass

        packages.append({
            "Package_ID": int(p["Package_ID"]),
            "Dimensions": {
                "Length": float(p["Dimensions"]["Length"]),
                "Width": float(p["Dimensions"]["Width"]),
                "Height": float(p["Dimensions"]["Height"]),
            },
            "Weight": float(p["Weight"]),
            "Position": {
                "x": 0.0,
                "y": 0.0,
                "z": float(z_cursor)
            }
        })
        z_cursor += H

    return packages


def explode_qty_from_csv(df: pd.DataFrame) -> list[dict]:
    """
    CSV columns supported (case-insensitive):
      package_id, package_type, length, width, height, weight, qty
    Returns a list (without positions), one entry per unit.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"package_id", "length", "width", "height", "weight", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(list(missing))}")

    has_type = "package_type" in df.columns

    items = []
    for _, row in df.iterrows():
        qty = int(row["qty"])
        for _ in range(qty):
            item = {
                "Package_ID": int(row["package_id"]),
                "Dimensions": {
                    "Length": float(row["length"]),
                    "Width": float(row["width"]),
                    "Height": float(row["height"]),
                },
                "Weight": float(row["weight"]),
            }
            if has_type and pd.notna(row["package_type"]):
                item["Package_Type"] = str(row["package_type"]).strip()
            items.append(item)

    return items

def upsert_typedata_container(typedata_container: dict, container_type: str,
                             L: float, W: float, H: float,
                             tare_weight: float, payload_limit: float,
                             use_buffer: bool, bL: float, bW: float, bH: float) -> dict:
    """
    typedata_container.json format (friend-provided):
      {
        "<key>": {
           "Container_Type": "<same or different>",
           "Dimensions": {"Length":..., "Width":..., "Height":...},
           "DimensionBuffer": {...} (optional),
           "Weight": <tare>,
           "WeightLimit": <payload>
        },
        ...
      }
    We'll use key == Container_Type for consistency.
    """
    entry = {
        "Container_Type": container_type,
        "Dimensions": {"Length": float(L), "Width": float(W), "Height": float(H)},
        "Weight": float(tare_weight),
        "WeightLimit": float(payload_limit),
    }
    if use_buffer:
        entry["DimensionBuffer"] = {"Length": float(bL), "Width": float(bW), "Height": float(bH)}

    typedata_container[container_type] = entry
    return typedata_container


def upsert_typedata_package(typedata_package: dict, package_type: str,
                            L: float, W: float, H: float,
                            weight: float, color: str,
                            weight_limit: float | None) -> dict:
    """
    typedata_package.json format (friend-provided) AND your rule:
      - dictionary key MUST equal Package_Type.

      {
        "<Package_Type>": {
           "Package_Type": "<Package_Type>",
           "Dimensions": {...},
           "Weight": ...,
           "WeightLimit": ... (optional),
           "Color": ...
        },
        ...
      }
    """
    entry = {
        "Package_Type": package_type,  # MUST equal dict key
        "Dimensions": {"Length": float(L), "Width": float(W), "Height": float(H)},
        "Weight": float(weight),
        "Color": color
    }
    if weight_limit is not None:
        entry["WeightLimit"] = float(weight_limit)

    typedata_package[package_type] = entry
    return typedata_package


def infer_types_from_packages(packages_raw: list[dict], prefix: str = "PKG_", default_color: str = "Brown") -> dict:
    """
    Build typedata_package entries from current packages list.
    Each unique Package_ID becomes a Package_Type = f"{prefix}{id}".
    Enforces key == Package_Type.
    """
    typed = {}
    seen = set()
    for p in packages_raw:
        pid = int(p["Package_ID"])
        if pid in seen:
            continue
        seen.add(pid)
        pkg_type = f"{prefix}{pid}"
        typed[pkg_type] = {
            "Package_Type": pkg_type,
            "Dimensions": {
                "Length": float(p["Dimensions"]["Length"]),
                "Width": float(p["Dimensions"]["Width"]),
                "Height": float(p["Dimensions"]["Height"]),
            },
            "Weight": float(p["Weight"]),
            "Color": default_color
        }
    return typed


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Input → Manifest Generator", layout="wide")
st.title("📦 Smart Container Loading — Input Page ")

# st.caption("This generates a manifest.json in the EXACT schema your visualiser already reads.")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Keep typedata dictionaries in session state
if "typedata_container" not in st.session_state:
    st.session_state.typedata_container = {}

if "typedata_package" not in st.session_state:
    st.session_state.typedata_package = {}

# Default container presets (based on your project docs, but you can override in UI)
presets = {
    "40ft High Cube (sample)": {"L": 12.0, "W": 2.3, "H": 2.45},
    "20ft (sample)": {"L": 5.9, "W": 2.35, "H": 2.39},
    "Custom": None
}

colA, colB = st.columns(2)

with colA:
    st.subheader("1) Container Setup")
    preset_name = st.selectbox("Container preset", list(presets.keys()), index=0)

    if preset_name != "Custom":
        L_default = presets[preset_name]["L"]
        W_default = presets[preset_name]["W"]
        H_default = presets[preset_name]["H"]
    else:
        L_default, W_default, H_default = 12.0, 2.3, 2.45

    container_id = st.number_input("Container_ID", min_value=1, value=1, step=1)
    L = st.number_input("Container Length (m)", min_value=0.1, value=float(L_default), step=0.01)
    W = st.number_input("Container Width (m)", min_value=0.1, value=float(W_default), step=0.01)
    H = st.number_input("Container Height (m)", min_value=0.1, value=float(H_default), step=0.01)

with colB:
    st.subheader("2) Constraints (for validation + rules)")
    payload_limit = st.number_input("Payload limit (kg) (optional)", min_value=0.0, value=0.0, step=10.0)
    pallet_max_weight = st.number_input("Max weight per pallet (kg) (optional)", min_value=0.0, value=0.0, step=10.0)

    st.markdown("*Stacking rule by weight (optional):*")
    heavy_threshold = st.number_input("Heavy threshold (kg)", min_value=0.0, value=100.0, step=1.0)
    heavy_max_layers = st.number_input("Max layers for heavy items", min_value=1, value=3, step=1)

    st.markdown("*Placeholder positioning:*")
    z_start = st.number_input("Start z (m)", min_value=0.0, value=0.15, step=0.01)

st.divider()

# ---------------------------
# NEW: typedata_container controls (friend json)
# ---------------------------
st.subheader("🧱 Container Type Catalog (typedata_container.json)")

cc1, cc2, cc3 = st.columns(3)
with cc1:
    container_type_name = st.text_input("Container_Type name (typedata key)", value="40ft High Cube Container")
with cc2:
    tare_weight = st.number_input("Container tare weight (kg)", min_value=0.0, value=0.0, step=10.0)
with cc3:
    payload_for_typedata = st.number_input("Container WeightLimit / payload (kg)", min_value=-1.0, value=26660.0, step=10.0)

use_buffer = st.checkbox("Include DimensionBuffer in typedata_container.json?", value=True)
b1, b2, b3 = st.columns(3)
with b1:
    bL = st.number_input("Buffer Length (m)", min_value=0.0, value=0.03, step=0.01)
with b2:
    bW = st.number_input("Buffer Width (m)", min_value=0.0, value=0.05, step=0.01)
with b3:
    bH = st.number_input("Buffer Height (m)", min_value=0.0, value=0.17, step=0.01)

if st.button("✅ Add/Update container type into typedata_container"):
    st.session_state.typedata_container = upsert_typedata_container(
        st.session_state.typedata_container,
        container_type=container_type_name,
        L=L, W=W, H=H,
        tare_weight=tare_weight,
        payload_limit=payload_for_typedata,
        use_buffer=use_buffer,
        bL=bL, bW=bW, bH=bH
    )
    st.success(f"Saved container type: {container_type_name}")

st.json(st.session_state.typedata_container)

st.divider()

# ---------------------------
# NEW: Pallet type (typedata_package) + your rule key==Package_Type
# ---------------------------
st.subheader("🟧 Pallet Type (typedata_package.json)")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    pallet_type_name = st.text_input("Pallet Package_Type (key must match)", value="Loscam Wooden Pallet - Asia")
with pc2:
    pallet_color = st.text_input("Pallet Color", value="Orange")
with pc3:
    pallet_weight = st.number_input("Pallet Weight (kg)", min_value=0.0, value=28.0, step=1.0)
with pc4:
    pallet_limit = st.number_input("Pallet WeightLimit (kg)", min_value=-1.0, value=1000.0, step=10.0)

pl1, pl2, pl3 = st.columns(3)
with pl1:
    pallet_L = st.number_input("Pallet Length (m)", min_value=0.1, value=1.20, step=0.01)
with pl2:
    pallet_W = st.number_input("Pallet Width (m)", min_value=0.1, value=1.00, step=0.01)
with pl3:
    pallet_H = st.number_input("Pallet Height (m)", min_value=0.01, value=0.156, step=0.01)

if st.button("✅ Save/Update pallet into typedata_package"):
    st.session_state.typedata_package = upsert_typedata_package(
        st.session_state.typedata_package,
        package_type=pallet_type_name,  # key == Package_Type enforced
        L=pallet_L, W=pallet_W, H=pallet_H,
        weight=pallet_weight,
        color=pallet_color,
        weight_limit=pallet_limit
    )
    st.success(f"Saved pallet type: {pallet_type_name} (key == Package_Type ✅)")

st.divider()

st.subheader("3) Input Packages: CSV Upload OR Manual Entry")

tab1, tab2 = st.tabs(["📄 Upload CSV", "✍️ Manual Entry"])

packages_raw: list[dict] = []

with tab1:
    st.write("CSV columns required: package_id, length, width, height, weight, qty")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df)

        try:
            packages_raw = explode_qty_from_csv(df)
            st.success(f"Loaded {len(packages_raw)} packages from CSV (qty expanded).")
        except Exception as e:
            st.error(str(e))

with tab2:
    if "manual_list" not in st.session_state:
        st.session_state.manual_list = []

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pid = st.number_input("Package_ID", min_value=0, value=1, step=1)
            pweight = st.number_input("Weight (kg)", min_value=0.0, value=10.0, step=0.1)
        with c2:
            pl = st.number_input("Length (m)", min_value=0.01, value=0.4, step=0.01)
            pw = st.number_input("Width (m)", min_value=0.01, value=0.3, step=0.01)
        with c3:
            ph = st.number_input("Height (m)", min_value=0.01, value=0.2, step=0.01)
            qty = st.number_input("Quantity", min_value=1, value=1, step=1)

        add_btn = st.form_submit_button("Add to list")

    if add_btn:
        for _ in range(int(qty)):
            st.session_state.manual_list.append({
                "Package_ID": int(pid),
                "Dimensions": {"Length": float(pl), "Width": float(pw), "Height": float(ph)},
                "Weight": float(pweight),
            })
        st.success(f"Added {int(qty)} package(s). Total manual: {len(st.session_state.manual_list)}")

    if st.session_state.manual_list:
        st.write("Manual packages (first 20 shown):")
        st.dataframe(pd.DataFrame(st.session_state.manual_list).head(20))

        if st.button("Clear manual list"):
            st.session_state.manual_list = []
            st.rerun()

    # Use manual list if no CSV
    if not packages_raw and st.session_state.manual_list:
        packages_raw = st.session_state.manual_list

st.divider()

# ---------------------------
# NEW: typedata_package generator (from Package_IDs) with key==Package_Type
# ---------------------------
st.subheader("🧩 Package Type Catalog (typedata_package.json)")
st.caption("Auto-create package types from the packages list. Keys will be Package_Type = prefix + Package_ID.")

tcol1, tcol2, tcol3 = st.columns(3)
with tcol1:
    type_prefix = st.text_input("Package_Type prefix", value="PKG_")
with tcol2:
    default_color = st.text_input("Default Color", value="Brown")
with tcol3:
    include_weight_limit = st.checkbox("Include WeightLimit for auto types?", value=False)

auto_wlimit = None
if include_weight_limit:
    auto_wlimit = st.number_input("Auto WeightLimit value (kg)", min_value=0.0, value=0.0, step=1.0)

if st.button("⚡ Generate/Update typedata_package from packages list"):
    if not packages_raw:
        st.warning("No packages loaded yet.")
    else:
        auto_types = infer_types_from_packages(packages_raw, prefix=type_prefix, default_color=default_color)

        # merge into session typedata_package (and optionally include weightlimit)
        for k, v in auto_types.items():
            if auto_wlimit is not None:
                v["WeightLimit"] = float(auto_wlimit)
            st.session_state.typedata_package[k] = v

        st.success(f"Generated/updated {len(auto_types)} package types. (key == Package_Type ✅)")

st.json(st.session_state.typedata_package)

st.divider()

# ---------------------------
# Apply rules (simple) + manifest generation
# ---------------------------

if packages_raw:
    heavy_count = sum(1 for p in packages_raw if float(p["Weight"]) >= float(heavy_threshold))

    packages_ready = stack_positions(packages_raw, z_start=float(z_start), max_height=float(H))

    manifest = build_manifest(container_id=int(container_id), L=float(L), W=float(W), H=float(H), packages=packages_ready)

    total_weight, oversized = validate_packages(
        packages_ready,
        {"Length": float(L), "Width": float(W), "Height": float(H)},
        payload_limit if payload_limit > 0 else None
    )

    st.subheader("4) Summary & Generate manifest.json")
    st.write(f"*Packages loaded:* {len(packages_ready)}")
    st.write(f"*Heavy items (≥ {heavy_threshold} kg):* {heavy_count}")
    st.write(f"*Total weight:* {total_weight:.2f} kg")

    if payload_limit > 0 and total_weight > payload_limit:
        st.warning("Total weight exceeds payload limit — you should split load or reject in optimiser.")

    if oversized > 0:
        st.warning(f"{oversized} package(s) exceed container dimensions (simple check).")

    if st.button("✅ Generate output/manifest.json"):
        outpath = output_dir / "manifest.json"
        outpath.write_text(json.dumps(manifest, indent=2))
        st.success(f"Saved: {outpath.resolve()}")

    st.code(json.dumps(manifest, indent=2), language="json")

else:
    st.info("Upload a CSV or add manual packages to generate the manifest.")

st.divider()

# ---------------------------
# NEW: Save typedata files
# ---------------------------
st.subheader("💾 Save JSON files (typedata_container.json + typedata_package.json)")

s1, s2 = st.columns(2)
with s1:
    if st.button("💾 Save output/typedata_container.json"):
        outpath = output_dir / "typedata_container.json"
        outpath.write_text(json.dumps(st.session_state.typedata_container, indent=4))
        st.success(f"Saved: {outpath.resolve()}")

with s2:
    if st.button("💾 Save output/typedata_package.json"):
        outpath = output_dir / "typedata_package.json"
        outpath.write_text(json.dumps(st.session_state.typedata_package, indent=4))
        st.success(f"Saved: {outpath.resolve()}")

st.caption("✅ Rule enforced: typedata_package dictionary key == Package_Type.")