import json
from dataclasses import dataclass
from typing import List, Tuple, Optional



#Data models

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
    rotated: bool


@dataclass
class StackSpot:
    x: float
    y: float
    z: float   # top surface height (in metres)
    L: float
    W: float



#Save and load JSON files

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


#typedata helpers
def td_get_dims(td_pkg: dict, package_type: str):
    entry = td_pkg.get(package_type)
    if not entry:
        return None
    d = entry.get("Dimensions")
    if not d:
        return None
    return Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))


def td_get_weight(td_pkg: dict, package_type: str):
    entry = td_pkg.get(package_type)
    if not entry:
        return None
    w = entry.get("Weight")
    return float(w) if w is not None else None



#Orientation change

def orientations_horizontal(d: Dims):
    #This allows horizontal rotation only (swap L/W). Height stays the same.
    return [(d, False), (Dims(d.W, d.L, d.H), True)]



#Normalize input items

def normalize_input_items(manifest_in: dict, td_pkg: dict, pallet_type: str):
    
    #Ensures each package item has Package_Type, Dimensions, Weight.
    #Ignores pallet entries (Package_Type == pallet_type).
    
    items: List[Item] = []
    pallet_type = str(pallet_type).strip()

    for p in manifest_in.get("Packages", []):
        pid = int(p.get("Package_ID", 0))
        ptype = p.get("Package_Type")

        if not ptype:
            raise ValueError(
                f"Input manifest package id={pid} is missing Package_Type. "
                f"Fix Streamlit to include Package_Type in manifest."
            )
        ptype = str(ptype).strip()

        #Ignore pallets if they accidentally appear as packages
        if ptype == pallet_type:
            continue

        #Dimensions
        d = p.get("Dimensions")
        if d:
            dims = Dims(float(d["Length"]), float(d["Width"]), float(d["Height"]))
        else:
            td_dims = td_get_dims(td_pkg, ptype)
            if not td_dims:
                raise ValueError(
                    f"Missing Dimensions for Package_Type='{ptype}' (pid={pid}). "
                    f"Provide Dimensions in manifest OR typedata_package.json."
                )
            dims = td_dims

        #Weight
        w = p.get("Weight")
        if w is not None:
            weight = float(w)
        else:
            td_w = td_get_weight(td_pkg, ptype)
            if td_w is None:
                raise ValueError(
                    f"Missing Weight for Package_Type='{ptype}' (pid={pid}). "
                    f"Provide Weight in manifest OR typedata_package.json."
                )
            weight = float(td_w)

        items.append(Item(package_id=pid, package_type=ptype, dims=dims, weight=weight))

    return items



#Support-based pallet packing (no more floating)

def pack_one_pallet_supported(
    items: List[Item],
    pallet_dims: Dims,
    pallet_height: float,
    container_height: float,
    heavy_threshold: float = 100.0,
    heavy_max_layers: int = 3,
):
    
    #Packs onto ONE pallet with strict support:
     #Base layer: shelf packing at z = pallet_height
      #Upper layers: ONLY stack directly on top of an existing box footprint
    #Prevents floating.
    

    def volume(it: Item):
        return it.dims.L * it.dims.W * it.dims.H

    ordered = sorted(items, key=lambda it: (it.weight < heavy_threshold, -volume(it)))

    placed: List[PlacedItem] = []
    remaining: List[Item] = []

    #Base shelf state
    z_base = pallet_height
    x_cursor = 0.0
    y_cursor = 0.0
    row_depth = 0.0

    stack_spots: List[StackSpot] = []

    def approx_layer_index(z: float, x: float, y: float) -> int:
        return sum(
            1 for p in placed
            if abs(p.x - x) < 1e-6 and abs(p.y - y) < 1e-6 and p.z < z + 1e-9
        )

    for it in ordered:
        #Impossible checks
        if it.dims.H + pallet_height > container_height:
            raise ValueError(f"Item pid={it.package_id} too tall even on pallet.")

        fits_any = any(
            d_try.L <= pallet_dims.L and d_try.W <= pallet_dims.W
            for d_try, _ in orientations_horizontal(it.dims)
        )
        if not fits_any:
            raise ValueError(f"Item pid={it.package_id} footprint too large for pallet.")

        placed_this = False

        #1) Base layer (on pallet)
        for dims_try, rotated in orientations_horizontal(it.dims):
            x = x_cursor
            y = y_cursor

            #wrap row if needed
            if x + dims_try.L > pallet_dims.L:
                x = 0.0
                y = y_cursor + row_depth

            #no space on pallet base
            if y + dims_try.W > pallet_dims.W:
                continue

            #height check
            if z_base + dims_try.H > container_height:
                continue

            #place
            placed.append(PlacedItem(
                package_id=it.package_id,
                package_type=it.package_type,
                dims=dims_try,
                weight=it.weight,
                x=x,
                y=y,
                z=z_base,
                rotated=rotated
            ))

            #add a stack spot above it
            stack_spots.append(StackSpot(
                x=x, y=y, z=z_base + dims_try.H, L=dims_try.L, W=dims_try.W
            ))

            #commit shelf cursor update
            if x == 0.0 and (x_cursor + dims_try.L > pallet_dims.L):
                # we wrapped to new row
                y_cursor = y
                row_depth = 0.0

            x_cursor = x + dims_try.L
            row_depth = max(row_depth, dims_try.W)

            placed_this = True
            break

        if placed_this:
            continue

        #2) Stacking (must be supported by a box below)
        stack_spots.sort(key=lambda s: s.z)  # lowest first

        for dims_try, rotated in orientations_horizontal(it.dims):
            for si, spot in enumerate(stack_spots):
                layer_idx = approx_layer_index(spot.z, spot.x, spot.y)
                if it.weight >= heavy_threshold and layer_idx >= heavy_max_layers:
                    continue

                #must fit within supporting footprint
                if dims_try.L <= spot.L and dims_try.W <= spot.W:
                    if spot.z + dims_try.H <= container_height:
                        placed.append(PlacedItem(
                            package_id=it.package_id,
                            package_type=it.package_type,
                            dims=dims_try,
                            weight=it.weight,
                            x=spot.x,
                            y=spot.y,
                            z=spot.z,
                            rotated=rotated
                        ))

                        # replace spot with new spot on top
                        stack_spots[si] = StackSpot(
                            x=spot.x, y=spot.y, z=spot.z + dims_try.H,
                            L=dims_try.L, W=dims_try.W
                        )
                        placed_this = True
                        break
            if placed_this:
                break

        if not placed_this:
            remaining.append(it)

    return placed, remaining



#Better pallet floor placement (now uses width and length, tries rotation, centers grid)

def plan_pallet_floor(
    num_pallets: int,
    container_L: float,
    container_W: float,
    pallet_L: float,
    pallet_W: float,
    gap: float
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    
    #Improved pallet floor planner:
      # tries both pallet orientations
      # searches for a rows/cols arrangement that fits all pallets
      # prefers using MORE rows (better width utilization)
      # then prefers using LESS total length (compact)
      # centers the resulting grid in the container
    

    def max_counts(Lp: float, Wp: float) -> Tuple[int, int]:
        cols = int((container_L + gap) // (Lp + gap))
        rows = int((container_W + gap) // (Wp + gap))
        return cols, rows

    def used_length(cols: int, Lp: float) -> float:
        return cols * Lp + (cols - 1) * gap

    def used_width(rows: int, Wp: float) -> float:
        return rows * Wp + (rows - 1) * gap

    best = None
    best_score = None  #tuple for sorting

    #Try both pallet orientations
    for (Lp, Wp) in [(pallet_L, pallet_W), (pallet_W, pallet_L)]:
        max_cols, max_rows = max_counts(Lp, Wp)
        if max_cols <= 0 or max_rows <= 0:
            continue

        #Try row counts from max_rows downwards (prefer more rows)
        for rows in range(max_rows, 0, -1):
            cols_needed = (num_pallets + rows - 1) // rows  # ceil(num/rows)
            if cols_needed <= 0:
                continue
            if cols_needed > max_cols:
                continue

            # Valid layout
            L_used = used_length(cols_needed, Lp)
            W_used = used_width(rows, Wp)

            if L_used > container_L + 1e-9 or W_used > container_W + 1e-9:
                continue

            #Scoring:
            #1) maximize rows
            #2) minimize used length
            #3) minimize wasted slots (cols*rows - num)
            waste = cols_needed * rows - num_pallets
            score = (-rows, L_used, waste)

            if best_score is None or score < best_score:
                best_score = score
                best = (Lp, Wp, cols_needed, rows)

    if best is None:
        #fallback: compute absolute max capacity and explain
        c1, r1 = max_counts(pallet_L, pallet_W)
        c2, r2 = max_counts(pallet_W, pallet_L)
        cap1, cap2 = c1 * r1, c2 * r2
        raise ValueError(
            f"Not enough container floor space: need {num_pallets} pallets, "
            f"max is {max(cap1, cap2)} with pallet {pallet_L}x{pallet_W}."
        )

    Lp, Wp, cols, rows = best

    #Center the grid
    L_used = used_length(cols, Lp)
    W_used = used_width(rows, Wp)

    offset_x = max(0.0, (container_L - L_used) / 2.0)
    offset_y = max(0.0, (container_W - W_used) / 2.0)

    positions = []
    placed = 0
    for ry in range(rows):
        for cx in range(cols):
            if placed >= num_pallets:
                break
            x = offset_x + cx * (Lp + gap)
            y = offset_y + ry * (Wp + gap)
            positions.append((x, y))
            placed += 1
        if placed >= num_pallets:
            break

    return positions, (Lp, Wp)


# Main optimizer build
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

    cd = manifest_in.get("Dimensions")
    if not cd:
        raise ValueError("Input manifest missing Dimensions.")
    container_dims = Dims(float(cd["Length"]), float(cd["Width"]), float(cd["Height"]))

    pallet_dims = td_get_dims(td_pkg, pallet_type)
    if not pallet_dims:
        raise ValueError(f"Pallet type '{pallet_type}' missing from typedata_package.json (or missing Dimensions).")

    pallet_height = pallet_dims.H

    cargo_items = normalize_input_items(manifest_in, td_pkg, pallet_type=pallet_type)
    if not cargo_items:
        raise ValueError("No cargo items found to optimize.")

    # Pack onto as many pallets as needed
    all_pallet_loads: List[List[PlacedItem]] = []
    remaining = cargo_items[:]
    safety_counter = 0

    while remaining:
        safety_counter += 1
        if safety_counter > 5000:
            raise RuntimeError("Safety stop: too many iterations.")

        placed, remaining2 = pack_one_pallet_supported(
            remaining,
            pallet_dims=pallet_dims,
            pallet_height=pallet_height,
            container_height=container_dims.H,
            heavy_threshold=heavy_threshold,
            heavy_max_layers=heavy_max_layers
        )

        if not placed:
            impossible = remaining[0]
            raise ValueError(
                f"Could not place any remaining items onto a new pallet. Example pid={impossible.package_id}, "
                f"type='{impossible.package_type}'."
            )

        all_pallet_loads.append(placed)
        remaining = remaining2

    #Improved pallet floor placement
    pallet_xy, (Lp_used, Wp_used) = plan_pallet_floor(
        num_pallets=len(all_pallet_loads),
        container_L=container_dims.L,
        container_W=container_dims.W,
        pallet_L=pallet_dims.L,
        pallet_W=pallet_dims.W,
        gap=pallet_gap
    )

    #Build output manifest
    out_packages: List[dict] = []
    pallet_id_start = 900000

    for idx, (pallet_load, (px, py)) in enumerate(zip(all_pallet_loads, pallet_xy)):
        pallet_id = pallet_id_start + idx

        out_packages.append({
            "Package_ID": pallet_id,
            "Package_Type": pallet_type,
            "Dimensions": {"Length": pallet_dims.L, "Width": pallet_dims.W, "Height": pallet_dims.H},
            "Position": {"x": round(px, 5), "y": round(py, 5), "z": 0.0}
        })

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
        "Dimensions": {"Length": container_dims.L, "Width": container_dims.W, "Height": container_dims.H},
        "Packages": out_packages
    }

    save_json(out_path, manifest_out)

    return {
        "cargo_items": len(cargo_items),
        "pallets_used": len(all_pallet_loads),
        "out_path": out_path,
        "pallet_floor_used_LW": (Lp_used, Wp_used)
    }




#CLI runner
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Heuristic optimizer: input manifest -> optimized manifest (pallet-based)")
    ap.add_argument("--in", dest="manifest_in", required=True)
    ap.add_argument("--typedata-package", dest="td_pkg", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--pallet-type", dest="pallet_type", required=True)
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

    print("OK: Optimizer complete:", info)