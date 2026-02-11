import json
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
# JSON helpers
# -----------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# typedata helpers
# -----------------------------
def td_get_dims(td_pkg: dict, package_type: str) -> Optional[Dims]:
    entry = td_pkg.get(package_type)
    if not entry:
        return None
    d = entry.get("Dimensions")
    if not d:
        return None
    return Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))


def td_get_weight(td_pkg: dict, package_type: str) -> Optional[float]:
    entry = td_pkg.get(package_type)
    if not entry:
        return None
    w = entry.get("Weight")
    return float(w) if w is not None else None


def td_get_weightlimit(td_pkg: dict, package_type: str, default: float = 1000.0) -> float:
    entry = td_pkg.get(package_type, {})
    wl = entry.get("WeightLimit", default)
    return float(wl)


# -----------------------------
# Orientation helper
# -----------------------------
def orientations_horizontal(d: Dims) -> List[Tuple[Dims, bool]]:
    """Allow horizontal rotation only (swap L/W). Height stays the same."""
    return [(d, False), (Dims(d.W, d.L, d.H), True)]


# -----------------------------
# Normalize input items
# -----------------------------
def normalize_input_items(
    manifest_in: dict,
    td_pkg: dict,
    pallet_type: str,
) -> List[Item]:
    """
    Reads packages from input manifest. Ensures every cargo item has:
      - Package_Type
      - Dimensions
      - Weight
    Uses typedata_package.json as fallback if missing.
    Ignores entries that are pallets (Package_Type == pallet_type).
    """
    items: List[Item] = []

    for p in manifest_in.get("Packages", []):
        pid = int(p.get("Package_ID", 0))
        ptype = p.get("Package_Type")

        # If Package_Type missing, we cannot reliably resolve it.
        if not ptype:
            raise ValueError(
                f"Input manifest package id={pid} is missing Package_Type. "
                f"Fix Streamlit CSV parser to include Package_Type in manifest."
            )

        # Ignore pallets accidentally included as cargo
        if ptype.strip() == pallet_type.strip():
            continue

        # Resolve dimensions
        d = p.get("Dimensions")
        if d:
            dims = Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))
        else:
            td_dims = td_get_dims(td_pkg, ptype)
            if not td_dims:
                raise ValueError(
                    f"Missing Dimensions for Package_Type='{ptype}' (pid={pid}). "
                    f"Provide Dimensions in manifest OR define it in typedata_package.json."
                )
            dims = td_dims

        # Resolve weight
        w = p.get("Weight")
        if w is not None:
            weight = float(w)
        else:
            td_w = td_get_weight(td_pkg, ptype)
            if td_w is None:
                raise ValueError(
                    f"Missing Weight for Package_Type='{ptype}' (pid={pid}). "
                    f"Provide Weight in manifest OR define it in typedata_package.json."
                )
            weight = float(td_w)

        items.append(Item(package_id=pid, package_type=ptype, dims=dims, weight=weight))

    return items


# -----------------------------
# Shelf packer (single pallet)
# -----------------------------
def pack_one_pallet_shelf(
    items: List[Item],
    pallet_dims: Dims,
    pallet_height: float,
    container_height: float,
    heavy_threshold: float = 100.0,
    heavy_max_layers: int = 3
) -> Tuple[List[PlacedItem], List[Item]]:
    """
    Attempts to pack as MANY items as possible onto ONE pallet footprint using a shelf strategy.
    Returns:
      - placed items (with x,y,z)
      - remaining items (not placed)
    IMPORTANT: This does NOT throw if some can't fit; it just leaves them in remaining.
    It only throws if an item is physically impossible to ever place (larger than pallet footprint / too tall).
    """

    # Sort: heavy first, then by volume desc (helps stability + fit)
    def volume(it: Item) -> float:
        return it.dims.L * it.dims.W * it.dims.H

    ordered = sorted(items, key=lambda it: (it.weight < heavy_threshold, -volume(it)))

    placed: List[PlacedItem] = []
    remaining: List[Item] = []

    # Cursor state
    x_cursor = 0.0
    y_cursor = 0.0
    z_cursor = pallet_height
    row_depth = 0.0
    layer_height = 0.0
    layer_index = 0

    for it in ordered:
        # Quick “ever possible” checks
        if it.dims.H + pallet_height > container_height:
            raise ValueError(
                f"Item pid={it.package_id} type='{it.package_type}' is too tall for container "
                f"even on pallet (itemH={it.dims.H}, palletH={pallet_height}, containerH={container_height})."
            )

        # If item footprint larger than pallet in both orientations, impossible
        fits_any = False
        for d_try, _ in orientations_horizontal(it.dims):
            if d_try.L <= pallet_dims.L and d_try.W <= pallet_dims.W:
                fits_any = True
                break
        if not fits_any:
            raise ValueError(
                f"Item pid={it.package_id} type='{it.package_type}' footprint too large for pallet. "
                f"Item={it.dims.L}x{it.dims.W}, Pallet={pallet_dims.L}x{pallet_dims.W}."
            )

        # Heavy stacking cap (layers)
        if it.weight >= heavy_threshold and layer_index >= heavy_max_layers:
            # can't place this heavy item on this pallet under rules
            remaining.append(it)
            continue

        placed_this = False

        # Try both orientations at current cursor, with normal shelf progression
        for dims_try, rotated in orientations_horizontal(it.dims):
            # Copy cursor state to test placement
            x = x_cursor
            y = y_cursor
            z = z_cursor
            rd = row_depth
            lh = layer_height
            li = layer_index

            # If doesn't fit in current row, move to next row
            if x + dims_try.L > pallet_dims.L:
                x = 0.0
                y += rd
                rd = 0.0

            # If doesn't fit in footprint, start new layer
            if y + dims_try.W > pallet_dims.W:
                x = 0.0
                y = 0.0
                z += lh
                lh = 0.0
                rd = 0.0
                li += 1

            # Heavy layer cap after moving layer
            if it.weight >= heavy_threshold and li >= heavy_max_layers:
                continue

            # Height check
            if z + dims_try.H > container_height:
                continue

            # Final footprint check
            if x + dims_try.L <= pallet_dims.L and y + dims_try.W <= pallet_dims.W:
                # Commit placement
                placed.append(
                    PlacedItem(
                        package_id=it.package_id,
                        package_type=it.package_type,
                        dims=dims_try,
                        weight=it.weight,
                        x=x,
                        y=y,
                        z=z,
                        rotated=rotated
                    )
                )

                # Update global cursors based on committed placement
                x_cursor = x + dims_try.L
                y_cursor = y
                z_cursor = z
                row_depth = max(rd, dims_try.W)
                layer_height = max(lh, dims_try.H)
                layer_index = li

                placed_this = True
                break

        if not placed_this:
            remaining.append(it)

    return placed, remaining


# -----------------------------
# Place pallets in container grid
# -----------------------------
def place_pallets_grid(
    num_pallets: int,
    container_dims: Dims,
    pallet_dims: Dims,
    gap: float = 0.02
) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    x = 0.0
    y = 0.0
    row_max_depth = 0.0

    for _ in range(num_pallets):
        # wrap row
        if x + pallet_dims.L > container_dims.L:
            x = 0.0
            y += row_max_depth + gap
            row_max_depth = 0.0

        if y + pallet_dims.W > container_dims.W:
            raise ValueError(
                "Not enough container floor space to place all required pallets. "
                "Reduce pallet count, use another container, or implement smarter pallet floor layout."
            )

        positions.append((x, y))
        x += pallet_dims.L + gap
        row_max_depth = max(row_max_depth, pallet_dims.W)

    return positions


# -----------------------------
# Main optimizer build
# -----------------------------
def build_optimized_manifest(
    manifest_in_path: str,
    typedata_package_path: str,
    out_path: str,
    pallet_type: str,
    heavy_threshold: float = 100.0,
    heavy_max_layers: int = 3,
    pallet_gap: float = 0.02
) -> dict:
    manifest_in = load_json(manifest_in_path)
    td_pkg = load_json(typedata_package_path)

    # Container dims
    cd = manifest_in.get("Dimensions")
    if not cd:
        raise ValueError("Input manifest missing Dimensions.")
    container_dims = Dims(float(cd["Length"]), float(cd["Width"]), float(cd["Height"]))

    # Pallet dims/limits from typedata
    pallet_dims = td_get_dims(td_pkg, pallet_type)
    if not pallet_dims:
        raise ValueError(f"Pallet type '{pallet_type}' missing from typedata_package.json (or missing Dimensions).")

    pallet_height = pallet_dims.H

    # Normalize cargo items (ignore pallet entries)
    cargo_items = normalize_input_items(manifest_in, td_pkg, pallet_type=pallet_type)
    if not cargo_items:
        raise ValueError("No cargo items found to optimize (did you accidentally mark everything as pallets?).")

    # ---------
    # NEW LOGIC:
    # Pack as many items as possible per pallet, then spill leftovers into new pallets,
    # until all cargo is placed OR a physical impossibility is found.
    # ---------
    all_pallet_loads: List[List[PlacedItem]] = []
    remaining = cargo_items[:]
    safety_counter = 0

    while remaining:
        safety_counter += 1
        if safety_counter > 5000:
            raise RuntimeError("Safety stop: too many iterations while creating pallets (unexpected loop).")

        placed, remaining2 = pack_one_pallet_shelf(
            remaining,
            pallet_dims=pallet_dims,
            pallet_height=pallet_height,
            container_height=container_dims.H,
            heavy_threshold=heavy_threshold,
            heavy_max_layers=heavy_max_layers
        )

        if not placed:
            # If nothing placed on a fresh pallet, remaining items are impossible under constraints
            impossible = remaining[0]
            raise ValueError(
                f"Could not place any remaining items onto a new pallet. Example item pid={impossible.package_id} "
                f"type='{impossible.package_type}'. Check pallet size, container height, and item dimensions."
            )

        all_pallet_loads.append(placed)
        remaining = remaining2

    # Place pallets on container floor
    pallet_xy = place_pallets_grid(
        num_pallets=len(all_pallet_loads),
        container_dims=container_dims,
        pallet_dims=pallet_dims,
        gap=pallet_gap
    )

    # Build manifest output
    out_packages: List[dict] = []
    pallet_id_start = 900000  # unique-ish IDs for pallet objects

    for idx, (pallet_load, (px, py)) in enumerate(zip(all_pallet_loads, pallet_xy)):
        pallet_id = pallet_id_start + idx

        # Pallet object at z=0
        out_packages.append({
            "Package_ID": pallet_id,
            "Package_Type": pallet_type,
            # Include dims/weight optional, but helpful:
            "Dimensions": {"Length": pallet_dims.L, "Width": pallet_dims.W, "Height": pallet_dims.H},
            "Position": {"x": round(px, 5), "y": round(py, 5), "z": 0.0}
        })

        # Cargo on top
        for it in pallet_load:
            out_packages.append({
                "Package_ID": it.package_id,
                "Package_Type": it.package_type,
                "Dimensions": {"Length": it.dims.L, "Width": it.dims.W, "Height": it.dims.H},
                "Weight": it.weight,
                "Position": {
                    "x": round(px + it.x, 5),
                    "y": round(py + it.y, 5),
                    "z": round(it.z, 5)
                }
            })

    manifest_out = {
        "Container_ID": manifest_in.get("Container_ID", 1),
        **({"Container_Type": manifest_in["Container_Type"]} if "Container_Type" in manifest_in else {}),
        "Dimensions": {"Length": container_dims.L, "Width": container_dims.W, "Height": container_dims.H},
        "Packages": out_packages
    }

    save_json(out_path, manifest_out)

    return {
        "cargo_items": len(cargo_items),
        "pallets_used": len(all_pallet_loads),
        "out_path": out_path
    }


# -----------------------------
# CLI runner
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Heuristic optimizer: input manifest -> optimized manifest (pallet-based)")
    ap.add_argument("--in", dest="manifest_in", required=True, help="Input manifest.json (from Streamlit)")
    ap.add_argument("--typedata-package", dest="td_pkg", required=True, help="typedata_package.json")
    ap.add_argument("--out", dest="out", required=True, help="Output optimized manifest.json")
    ap.add_argument("--pallet-type", dest="pallet_type", required=True, help="Exact pallet Package_Type string")
    ap.add_argument("--heavy-threshold", dest="heavy_threshold", type=float, default=100.0)
    ap.add_argument("--heavy-max-layers", dest="heavy_max_layers", type=int, default=3)
    ap.add_argument("--pallet-gap", dest="pallet_gap", type=float, default=0.02)

    args = ap.parse_args()

    info = build_optimized_manifest(
        manifest_in_path=args.manifest_in,
        typedata_package_path=args.td_pkg,
        out_path=args.out,
        pallet_type=args.pallet_type,
        heavy_threshold=args.heavy_threshold,
        heavy_max_layers=args.heavy_max_layers,
        pallet_gap=args.pallet_gap
    )

    print("✅ Optimizer complete:", info)