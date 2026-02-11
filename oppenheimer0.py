import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Data models
# -----------------------------
@dataclass
class Dims:
    L: float
    W: float
    H: float

@dataclass
class Item:
    package_id: int
    package_type: str
    dims: Dims
    weight: float

@dataclass
class PlacedItem(Item):
    x: float
    y: float
    z: float
    rotated: bool  # True if L/W swapped

# -----------------------------
# Helpers: load typedata
# -----------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def get_dims_from_typedata(td_pkg: dict, package_type: str) -> Dims:
    td = td_pkg.get(package_type)
    if not td or "Dimensions" not in td:
        raise ValueError(f"Package_Type '{package_type}' not found (or missing Dimensions) in typedata_package.json")
    d = td["Dimensions"]
    return Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))

def get_weightlimit_from_typedata(td_pkg: dict, package_type: str, default: float = 1000.0) -> float:
    td = td_pkg.get(package_type, {})
    wl = td.get("WeightLimit", default)
    return float(wl)

# -----------------------------
# Step 1: normalize input packages
# -----------------------------
def normalize_input_packages(
    manifest_in: dict,
    pkg_prefix: str = "PKG_"
) -> List[Item]:
    """
    Your Streamlit manifest may not have Package_Type.
    We set Package_Type = f"{pkg_prefix}{Package_ID}" so it matches your auto-generated typedata_package entries.
    """
    items: List[Item] = []

    for p in manifest_in.get("Packages", []):
        pid = int(p["Package_ID"])
        ptype = p.get("Package_Type") or f"{pkg_prefix}{pid}"

        # Prefer explicit dims/weight in manifest if present; else allow visualiser to pull from typedata.
        d = p.get("Dimensions", {})
        if d:
            dims = Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))
        else:
            # We'll resolve dims later via typedata if needed; placeholder for now
            dims = Dims(0.0, 0.0, 0.0)

        weight = float(p.get("Weight", 0.0))

        items.append(Item(package_id=pid, package_type=ptype, dims=dims, weight=weight))

    return items

# -----------------------------
# Step 2: assign packages to pallets (weight-based binning)
# -----------------------------
def assign_to_pallets_by_weight(
    items: List[Item],
    pallet_weight_limit: float
) -> List[List[Item]]:
    """
    Simple first-fit decreasing (FFD) on weight.
    """
    items_sorted = sorted(items, key=lambda x: x.weight, reverse=True)
    pallets: List[List[Item]] = []
    pallet_weights: List[float] = []

    for it in items_sorted:
        placed = False
        for i in range(len(pallets)):
            if pallet_weights[i] + it.weight <= pallet_weight_limit:
                pallets[i].append(it)
                pallet_weights[i] += it.weight
                placed = True
                break
        if not placed:
            pallets.append([it])
            pallet_weights.append(it.weight)

    return pallets

# -----------------------------
# Step 3: pack items on each pallet (simple 3D shelf heuristic)
# -----------------------------
def try_orientations(dims: Dims) -> List[Tuple[Dims, bool]]:
    """
    Only allow horizontal rotation (swap L/W). Height unchanged.
    """
    return [
        (dims, False),
        (Dims(dims.W, dims.L, dims.H), True)
    ]

def pack_on_pallet_shelf_3d(
    items: List[Item],
    pallet_dims: Dims,
    pallet_height: float,
    container_height: float,
    heavy_threshold: float = 100.0,
    heavy_max_layers: int = 3
) -> List[PlacedItem]:
    """
    Packs items on a pallet footprint:
    - Fill rows along x; when x runs out, move y; when y runs out, start new layer (z increases).
    - Heavy items first (so they tend to go lower layers).
    - Heavy items are restricted to layers < heavy_max_layers.
    """
    # Sort: heavy first, then by volume descending
    def vol(it: Item) -> float:
        return it.dims.L * it.dims.W * it.dims.H

    heavy = [it for it in items if it.weight >= heavy_threshold]
    light = [it for it in items if it.weight < heavy_threshold]
    heavy.sort(key=vol, reverse=True)
    light.sort(key=vol, reverse=True)
    ordered = heavy + light

    placed: List[PlacedItem] = []

    x_cursor = 0.0
    y_cursor = 0.0
    z_cursor = pallet_height

    row_depth = 0.0          # how much y this row consumes (max width in row)
    layer_height = 0.0       # max height in layer
    layer_index = 0

    max_L = pallet_dims.L
    max_W = pallet_dims.W

    for it in ordered:
        # Resolve missing dims (if any) – should ideally be provided already by input
        if it.dims.L <= 0 or it.dims.W <= 0 or it.dims.H <= 0:
            raise ValueError(f"Item {it.package_id} has missing/zero Dimensions. Provide dims in manifest or resolve from typedata.")

        # Heavy stacking cap
        if it.weight >= heavy_threshold and layer_index >= heavy_max_layers:
            # push to next pallet by failing here
            raise ValueError(
                f"Heavy item (id={it.package_id}, w={it.weight}) exceeded heavy_max_layers={heavy_max_layers}. "
                f"Need more pallets or relax rule."
            )

        placed_this = False

        for dims_try, rotated in try_orientations(it.dims):
            # If doesn't fit in current row, try moving to next row
            if x_cursor + dims_try.L > max_L:
                # new row
                x_cursor = 0.0
                y_cursor += row_depth
                row_depth = 0.0

            # If doesn't fit in footprint, start new layer
            if y_cursor + dims_try.W > max_W:
                # new layer
                x_cursor = 0.0
                y_cursor = 0.0
                z_cursor += layer_height
                layer_height = 0.0
                row_depth = 0.0
                layer_index += 1

            # Check height against container
            if z_cursor + dims_try.H > container_height:
                continue

            # Final fit check
            if (x_cursor + dims_try.L <= max_L) and (y_cursor + dims_try.W <= max_W):
                placed.append(
                    PlacedItem(
                        package_id=it.package_id,
                        package_type=it.package_type,
                        dims=dims_try,
                        weight=it.weight,
                        x=x_cursor,
                        y=y_cursor,
                        z=z_cursor,
                        rotated=rotated
                    )
                )
                # advance x in row
                x_cursor += dims_try.L
                row_depth = max(row_depth, dims_try.W)
                layer_height = max(layer_height, dims_try.H)
                placed_this = True
                break

        if not placed_this:
            raise ValueError(f"Could not place item {it.package_id} on pallet with shelf heuristic.")

    return placed

# -----------------------------
# Step 4: place pallets on container floor (grid)
# -----------------------------
def place_pallets_in_container_grid(
    num_pallets: int,
    container_dims: Dims,
    pallet_dims: Dims,
    gap: float = 0.02
) -> List[Tuple[float, float]]:
    """
    Places pallets on container floor starting at (0,0) in a simple grid.
    """
    positions: List[Tuple[float, float]] = []
    x = 0.0
    y = 0.0
    row_max_depth = 0.0

    for _ in range(num_pallets):
        # wrap to next row if needed
        if x + pallet_dims.L > container_dims.L:
            x = 0.0
            y += row_max_depth + gap
            row_max_depth = 0.0

        # if still doesn't fit, fail (need different arrangement or fewer pallets)
        if y + pallet_dims.W > container_dims.W:
            raise ValueError("Not enough floor space in container to place all pallets with current grid layout.")

        positions.append((x, y))
        x += pallet_dims.L + gap
        row_max_depth = max(row_max_depth, pallet_dims.W)

    return positions

# -----------------------------
# Build optimized manifest
# -----------------------------
def build_optimized_manifest(
    manifest_in_path: str,
    typedata_package_path: str,
    out_path: str,
    pallet_type: str,
    pkg_prefix: str = "PKG_",
    heavy_threshold: float = 100.0,
    heavy_max_layers: int = 3
):
    manifest_in = load_json(manifest_in_path)
    td_pkg = load_json(typedata_package_path)

    # Container dims from manifest (your input page includes these)
    cd = manifest_in.get("Dimensions", {})
    container_dims = Dims(float(cd["Length"]), float(cd["Width"]), float(cd["Height"]))

    # Pallet dims / limits from typedata
    pallet_dims = get_dims_from_typedata(td_pkg, pallet_type)
    pallet_height = pallet_dims.H
    pallet_weight_limit = get_weightlimit_from_typedata(td_pkg, pallet_type, default=1000.0)

    # Normalize package list
    items = normalize_input_packages(manifest_in, pkg_prefix=pkg_prefix)

    # Assign items into pallets by weight limit
    pallet_groups = assign_to_pallets_by_weight(items, pallet_weight_limit=pallet_weight_limit)

    # Pack each pallet group
    packed_pallets: List[List[PlacedItem]] = []
    for group in pallet_groups:
        packed = pack_on_pallet_shelf_3d(
            group,
            pallet_dims=Dims(pallet_dims.L, pallet_dims.W, pallet_dims.H),
            pallet_height=pallet_height,
            container_height=container_dims.H,
            heavy_threshold=heavy_threshold,
            heavy_max_layers=heavy_max_layers
        )
        packed_pallets.append(packed)

    # Place pallets on container floor
    pallet_xy = place_pallets_in_container_grid(
        num_pallets=len(packed_pallets),
        container_dims=container_dims,
        pallet_dims=pallet_dims
    )

    # Build output packages list:
    # - include pallets at z=0
    # - include packages at z>=pallet_height with offsets
    out_packages: List[dict] = []

    # Pallet IDs (arbitrary unique IDs in a safe range)
    pallet_id_start = 900000

    for idx, (packed_items, (px, py)) in enumerate(zip(packed_pallets, pallet_xy)):
        pallet_id = pallet_id_start + idx

        # Add pallet object
        out_packages.append({
            "Package_ID": pallet_id,
            "Package_Type": pallet_type,
            "Position": {"x": round(px, 5), "y": round(py, 5), "z": 0}
        })

        # Add packages on top of pallet
        for it in packed_items:
            out_packages.append({
                "Package_ID": it.package_id,
                "Package_Type": it.package_type,
                "Dimensions": {"Length": it.dims.L, "Width": it.dims.W, "Height": it.dims.H},
                "Weight": it.weight,
                "Position": {
                    "x": round(px + it.x, 5),
                    "y": round(py + it.y, 5),
                    "z": round(it.z, 5)  # already includes pallet height base
                }
            })

    manifest_out = {
        "Container_ID": manifest_in.get("Container_ID", 1),
        # keep Container_Type if it exists, else omit (exec.py can still use Dimensions)
        **({"Container_Type": manifest_in["Container_Type"]} if "Container_Type" in manifest_in else {}),
        "Dimensions": {"Length": container_dims.L, "Width": container_dims.W, "Height": container_dims.H},
        "Packages": out_packages
    }

    with open(out_path, "w") as f:
        json.dump(manifest_out, f, indent=2)

    return {
        "num_packages": len(items),
        "num_pallets": len(packed_pallets),
        "out_path": out_path
    }

# -----------------------------
# CLI runner
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Heuristic optimizer: manifest_in.json -> manifest_optimized.json")
    ap.add_argument("--in", dest="manifest_in", required=True, help="Input manifest.json from Streamlit")
    ap.add_argument("--typedata-package", dest="td_pkg", help="typedata_package.json (must include pallet type + PKG_ types)", default="setup/typedata_package.json")
    ap.add_argument("--out", dest="out", help="Output optimized manifest.json", default="output/manifest_optimized.json")
    ap.add_argument("--pallet-type", dest="pallet_type", required=True, help="Pallet Package_Type string (must match typedata)")
    ap.add_argument("--pkg-prefix", dest="pkg_prefix", default="PKG_", help="Prefix used for Package_Type mapping (default PKG_)")
    ap.add_argument("--heavy-threshold", dest="heavy_threshold", type=float, default=100.0)
    ap.add_argument("--heavy-max-layers", dest="heavy_max_layers", type=int, default=3)

    args = ap.parse_args()

    info = build_optimized_manifest(
        manifest_in_path=args.manifest_in,
        typedata_package_path=args.td_pkg,
        out_path=args.out,
        pallet_type=args.pallet_type,
        pkg_prefix=args.pkg_prefix,
        heavy_threshold=args.heavy_threshold,
        heavy_max_layers=args.heavy_max_layers
    )
    print("Done:", info)