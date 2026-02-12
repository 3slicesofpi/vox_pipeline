import json
from pathlib import Path

import pandas as pd
import streamlit as st


#Helpers

def build_manifest(container_id: int, L: float, W: float, H: float, packages: list[dict]):
    """
    Build manifest.json in the schema the visualiser expects.
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
        total_weight += float(p.get("Weight", 0.0))
        L = float(p["Dimensions"]["Length"])
        W = float(p["Dimensions"]["Width"])
        H = float(p["Dimensions"]["Height"])

        #Check if any dimension exceeds container (simple check; rotation not considered here)
        if L > Lc or W > Wc or H > Hc:
            oversized += 1

    return total_weight, oversized


def stack_positions(packages_raw: list[dict], z_start: float = 0.15, max_height: float | None = None):
    """
    Create simple placeholder positions:
    1:All packages at x=0, y=0
    2: z increases by package height (stacking vertically)

    """
    z_cursor = z_start
    packages = []

    for p in packages_raw:
        H = float(p["Dimensions"]["Height"])
        if max_height is not None and (z_cursor + H) > max_height:
            #keep it but warn later
            pass

        out = {
            "Package_ID": int(p["Package_ID"]),
            "Package_Type": str(p["Package_Type"]).strip(),  #Package_Type (used by optimizer)
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
        }
        packages.append(out)
        z_cursor += H

    return packages


def explode_qty_from_csv(df: pd.DataFrame, pallet_type_name: str | None = None) -> tuple[list[dict], int]:
    """
    CSV columns required (case-insensitive):
      package_id, package_type, weight, length, width, height, qty

    Returns:
      (packages_raw, pallet_qty_in_csv)

    packages_raw: list of NON-pallet cargo items (one per unit, Qty expanded)
    pallet_qty_in_csv: number of pallets detected in CSV (if pallet_type_name provided)

    Also ensures every expanded unit gets a UNIQUE Package_ID.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"package_id", "package_type", "weight", "length", "width", "height", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(list(missing))}")

    packages_raw: list[dict] = []
    pallet_qty = 0

    #unique Package_ID counter for expanded units (avoid repeated IDs in manifest)
    uid = 1

    for _, row in df.iterrows():
        ptype = str(row["package_type"]).strip()
        qty = int(row["qty"])

        #If this row is pallets, count them but do NOT add to package list
        if pallet_type_name and ptype == pallet_type_name:
            pallet_qty += qty
            continue

        for _ in range(qty):
            packages_raw.append({
                "Package_ID": uid,
                "Package_Type": ptype,
                "Dimensions": {
                    "Length": float(row["length"]),
                    "Width": float(row["width"]),
                    "Height": float(row["height"])
                },
                "Weight": float(row["weight"]),
            })
            uid += 1

    return packages_raw, pallet_qty


def upsert_typedata_container(typedata_container: dict, container_type: str,
                             L: float, W: float, H: float,
                             tare_weight: float, payload_limit: float,
                             use_buffer: bool, bL: float, bW: float, bH: float) -> dict:
    """
    typedata_container.json format:
      {
        "<key>": {
           "Container_Type": "<same>",
           "Dimensions": {"Length":..., "Width":..., "Height":...},
           "DimensionBuffer": {...} (optional),
           "Weight": <tare>,
           "WeightLimit": <payload>
        },
        ...
      }
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
    typedata_package.json format:
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


def infer_types_from_packages(packages_raw: list[dict], default_color: str = "Brown") -> dict:
    """
    Build typedata_package entries from current packages list.
    Each unique Package_Type becmoes a typedata entry.
    Enforces key == Package_Type.
    """
    typed = {}
    seen = set()

    for p in packages_raw:
        ptype = str(p["Package_Type"]).strip()
        if ptype in seen:
            continue
        seen.add(ptype)

        typed[ptype] = {
            "Package_Type": ptype,
            "Dimensions": {
                "Length": float(p["Dimensions"]["Length"]),
                "Width": float(p["Dimensions"]["Width"]),
                "Height": float(p["Dimensions"]["Height"]),
            },
            "Weight": float(p["Weight"]),
            "Color": default_color
        }

    return typed



#Streamlit UI
#____________________

st.set_page_config(page_title="Input → Manifest Generator", layout="wide")
st.title("📦 Smart Container Loading — Input Page")
st.caption("This generates json files optimizer and visualizer requires")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

#Keep typedata dictionaries in session state
if "typedata_container" not in st.session_state:
    st.session_state.typedata_container = {}

if "typedata_package" not in st.session_state:
    st.session_state.typedata_package = {}

#Default container presets (override in UI)
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
    st.subheader("2) Constraints")
    payload_limit = st.number_input("Payload limit (kg) (optional)", min_value=0.0, value=0.0, step=10.0)
    pallet_max_weight = st.number_input("Max weight per pallet (kg) (optional)", min_value=0.0, value=0.0, step=10.0)

    st.markdown("**Stacking rule by weight (optional):**")
    heavy_threshold = st.number_input("Heavy threshold (kg)", min_value=0.0, value=100.0, step=1.0)
    heavy_max_layers = st.number_input("Max layers for heavy items", min_value=1, value=3, step=1)

    st.markdown("**Placeholder positioning (for input manifest only):**")
    z_start = st.number_input("Start z (m)", min_value=0.0, value=0.15, step=0.01)

st.divider()


# Container type catalog (typedata_container.json)
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


#Pallet type (typedata_package.json)

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
        package_type=pallet_type_name,
        L=pallet_L, W=pallet_W, H=pallet_H,
        weight=pallet_weight,
        color=pallet_color,
        weight_limit=pallet_limit
    )
    st.success(f"Saved pallet type: {pallet_type_name} (key == Package_Type ✅)")

st.divider()


#Package input
#__________________________

st.subheader("3) Input Packages: CSV Upload OR Manual Entry")
st.caption("IMPORTANT: Pallet rows in CSV (Package_Type == pallet type) will be detected and EXCLUDED from cargo list.")

tab1, tab2 = st.tabs(["📄 Upload CSV", "✍️ Manual Entry"])

packages_raw: list[dict] = []
pallets_in_csv = 0

with tab1:
    st.write("CSV columns required: `package_id, package_type, weight, length, width, height, qty`")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df)

        try:
            packages_raw, pallets_in_csv = explode_qty_from_csv(df, pallet_type_name=pallet_type_name)
            st.success(f"Loaded {len(packages_raw)} cargo packages from CSV (qty expanded).")
            if pallets_in_csv > 0:
                st.info(
                    f"Detected {pallets_in_csv} pallet(s) in CSV (Package_Type='{pallet_type_name}'). "
                    f"Excluded from cargo list so pallets act as base platforms in optimizer."
                )
        except Exception as e:
            st.error(str(e))

with tab2:
    if "manual_list" not in st.session_state:
        st.session_state.manual_list = []

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ptype = st.text_input("Package_Type", value="Medium Carton Box - Standard Packing Size")
            pweight = st.number_input("Weight (kg)", min_value=0.0, value=10.0, step=0.1)
        with c2:
            pl = st.number_input("Length (m)", min_value=0.01, value=0.4, step=0.01)
            pw = st.number_input("Width (m)", min_value=0.01, value=0.3, step=0.01)
        with c3:
            ph = st.number_input("Height (m)", min_value=0.01, value=0.2, step=0.01)
            qty = st.number_input("Quantity", min_value=1, value=1, step=1)

        add_btn = st.form_submit_button("Add to list")

    if add_btn:
        #unique IDs for manual list too
        next_id = (max([p["Package_ID"] for p in st.session_state.manual_list]) + 1) if st.session_state.manual_list else 1
        for _ in range(int(qty)):
            st.session_state.manual_list.append({
                "Package_ID": int(next_id),
                "Package_Type": str(ptype).strip(),
                "Dimensions": {"Length": float(pl), "Width": float(pw), "Height": float(ph)},
                "Weight": float(pweight),
            })
            next_id += 1
        st.success(f"Added {int(qty)} package(s). Total manual: {len(st.session_state.manual_list)}")

    if st.session_state.manual_list:
        st.write("Manual packages (first 50 shown):")
        st.dataframe(pd.DataFrame(st.session_state.manual_list).head(50))

        if st.button("Clear manual list"):
            st.session_state.manual_list = []
            st.rerun()

    #Use manual list if no CSV
    if not packages_raw and st.session_state.manual_list:
        packages_raw = st.session_state.manual_list

st.divider()


#auto-generate typedata_package for cargo types (from Package_Type)
# ____________________________________________________________________

st.subheader("🧩 Package Type Catalog (typedata_package.json)")
st.caption("Auto-create/Update typedata_package entries from the current cargo packages list (key == Package_Type).")

tcol1, tcol2 = st.columns(2)
with tcol1:
    default_color = st.text_input("Default Color for auto types", value="Brown")
with tcol2:
    include_weight_limit = st.checkbox("Include WeightLimit for auto types?", value=False)

auto_wlimit = None
if include_weight_limit:
    auto_wlimit = st.number_input("Auto WeightLimit value (kg)", min_value=0.0, value=0.0, step=1.0)

if st.button("⚡ Generate/Update typedata_package from cargo packages list"):
    if not packages_raw:
        st.warning("No cargo packages loaded yet.")
    else:
        auto_types = infer_types_from_packages(packages_raw, default_color=default_color)

        for k, v in auto_types.items():
            if auto_wlimit is not None:
                v["WeightLimit"] = float(auto_wlimit)
            st.session_state.typedata_package[k] = v

        st.success(f"Generated/updated {len(auto_types)} cargo Package_Type entries. (key == Package_Type ✅)")

st.json(st.session_state.typedata_package)

st.divider()


#Generate manifest.json (for optimizer to consume)

if packages_raw:
    heavy_count = sum(1 for p in packages_raw if float(p["Weight"]) >= float(heavy_threshold))

    #Placeholder positions only (optimizer will overwrite)
    packages_ready = stack_positions(packages_raw, z_start=float(z_start), max_height=float(H))

    manifest = build_manifest(container_id=int(container_id), L=float(L), W=float(W), H=float(H), packages=packages_ready)

    total_weight, oversized = validate_packages(
        packages_ready,
        {"Length": float(L), "Width": float(W), "Height": float(H)},
        payload_limit if payload_limit > 0 else None
    )

    st.subheader("4) Summary & Generate manifest.json (for optimizer)")
    st.write(f"**Cargo packages loaded:** {len(packages_ready)}")
    st.write(f"**Pallets detected in CSV (excluded from cargo):** {pallets_in_csv}")
    st.write(f"**Heavy items (≥ {heavy_threshold} kg):** {heavy_count}")
    st.write(f"**Total cargo weight:** {total_weight:.2f} kg")

    if payload_limit > 0 and total_weight > payload_limit:
        st.warning("Total cargo weight exceeds payload limit — you should split load or reject in optimiser.")

    if oversized > 0:
        st.warning(f"{oversized} cargo package(s) exceed container dimensions (simple check).")

    if st.button("✅ Generate output/manifest.json"):
        outpath = output_dir / "manifest.json"
        outpath.write_text(json.dumps(manifest, indent=2))
        st.success(f"Saved: {outpath.resolve()}")

    st.code(json.dumps(manifest, indent=2), language="json")
else:
    st.info("Upload a CSV or add manual packages to generate the manifest.")

st.divider()


#Save typedata files

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

st.caption("✅ Rule enforced: typedata_package dictionary key == Package_Type. Pallet rows in CSV are excluded from cargo manifest.")