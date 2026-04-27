"""
Core internals for biological cell cascade diagrams.

Contains geometry helpers, drawing primitives, network construction,
layout computation, and color utilities shared by cascade_plot and
contrast_cascade_plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle, Arc, FancyBboxPatch, FancyArrowPatch


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

EDGE_LW_MIN = 2.4
EDGE_LW_MAX = 11.0

SENDER_PALETTE = [
    (0.85, 0.92, 0.78, 0.30),
    (0.95, 0.85, 0.72, 0.30),
    (0.80, 0.85, 0.95, 0.30),
    (0.95, 0.80, 0.90, 0.30),
]

TYPE_RGBA = {
    "receptor": (0.20, 0.65, 0.32, 0.90),
    "tf":       (0.75, 0.40, 0.15, 0.90),
    "ligand":   (0.15, 0.50, 0.72, 0.90),
    "gene":     (0.60, 0.20, 0.55, 0.90),
    "seed":     (0.45, 0.15, 0.45, 0.90),
}

EDGE_RGBA = {
    "upstream_r_tf":   (0.45, 0.45, 0.45, 0.35),
    "upstream_tf_lig": (0.45, 0.45, 0.45, 0.35),
    "lig_rec":         (0.20, 0.50, 0.72, 0.40),
    "rec_tf":          (0.20, 0.65, 0.32, 0.40),
    "tf_gene":         (0.60, 0.20, 0.55, 0.40),
}

DEFAULT_RECV_COLORS = dict(
    face=(0.72, 0.82, 0.96, 0.10),
    edge=(0.25, 0.45, 0.75, 0.75),
    arc=(0.25, 0.45, 0.75, 0.85),
    nuc_face=(0.50, 0.64, 0.88, 0.62),
    nuc_edge=(0.16, 0.30, 0.72, 0.92),
    label=(0.15, 0.35, 0.65),
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — geometry
# ═══════════════════════════════════════════════════════════════════════════

def arc_positions(cx, cy, r, n, theta_start_deg, theta_end_deg):
    if n == 0:
        return []
    angles = np.linspace(np.radians(theta_start_deg),
                         np.radians(theta_end_deg), max(n, 1))
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]


def arc_positions_ellipse(cx, cy, rx, ry, n, theta_start_deg, theta_end_deg):
    if n == 0:
        return []
    angles = np.linspace(np.radians(theta_start_deg),
                         np.radians(theta_end_deg), max(n, 1))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]


def vertical_positions(cx, cy, n, half_span):
    if n == 0:
        return []
    if n == 1:
        return [(cx, cy)]
    ys = np.linspace(cy - half_span, cy + half_span, n)
    return [(cx, y) for y in ys]


def grid_positions_in_ellipse(cx, cy, a, b, n):
    if n == 0:
        return []
    aspect = a / b if b > 0 else 1.0
    n_cols = max(1, int(np.ceil(np.sqrt(n * aspect))))
    n_rows = max(1, int(np.ceil(n / n_cols)))
    pts = []
    for r in np.linspace(-b * 0.85, b * 0.85, n_rows):
        hw = a * np.sqrt(max(0, 1 - (r / b) ** 2)) * 0.85 if b > 0 else a * 0.85
        row_n = min(n_cols, n - len(pts))
        if row_n <= 0:
            break
        for c in np.linspace(-hw, hw, max(row_n, 1)):
            pts.append((cx + c, cy + r))
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    return pts[:n]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — color
# ═══════════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def lighten(hex_str, amount=0.5):
    rgb = hex_to_rgb(hex_str)
    return tuple(v + (1 - v) * amount for v in rgb)


def parse_rgba(rgba):
    if isinstance(rgba, str):
        inner = rgba.strip().removeprefix("rgba(").removesuffix(")")
        parts = [float(x) for x in inner.split(",")]
        return (parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0, parts[3])
    return tuple(float(v) for v in rgba[:4])


def score_to_rgba(score, vmax, base_rgb, alpha=0.85):
    """Map a score in [0, vmax] to a white → base_rgb gradient."""
    if not np.isfinite(score) or vmax <= 0:
        return (1.0, 1.0, 1.0, alpha)
    t = clamp(score / vmax, 0.0, 1.0)
    r = 1.0 + t * (base_rgb[0] - 1.0)
    g = 1.0 + t * (base_rgb[1] - 1.0)
    b = 1.0 + t * (base_rgb[2] - 1.0)
    return (r, g, b, alpha)


def score_to_grey(score, vmax, alpha=0.65):
    """Map a score in [0, vmax] to a white → medium grey gradient for edges."""
    if not np.isfinite(score) or vmax <= 0:
        return (1.0, 1.0, 1.0, alpha)
    t = clamp(score / vmax, 0.0, 1.0)
    grey = 1.0 - t * 0.55   # 1.0 → 0.45 (lighter than before)
    return (grey, grey, grey, alpha)


def delta_to_rgba(delta, vmax=2.0, alpha=0.7, scheme="temperature"):
    """Diverging color map for contrast diagrams."""
    if not np.isfinite(delta):
        return (1, 1, 1, 0.5)
    norm = max(min(delta / vmax, 1.0), -1.0)
    _GAMMA = 0.80
    if scheme == "sex":
        if norm >= 0:
            t = norm ** _GAMMA
            r = 1.0 + t * (0.75 - 1.0)
            g = 1.0 + t * (0.20 - 1.0)
            b = 1.0 + t * (0.10 - 1.0)
        else:
            t = (-norm) ** _GAMMA
            r = 1.0 + t * (0.20 - 1.0)
            g = 1.0 + t * (0.50 - 1.0)
            b = 1.0 + t * (0.90 - 1.0)
    else:
        if norm >= 0:
            t = norm ** _GAMMA
            r = (1.0 - t)
            g = (1.0 - t)
            b = 1.0
        else:
            t = (-norm) ** _GAMMA
            r = 1.0
            g = (1.0 - t)
            b = (1.0 - t)
    return (r, g, b, alpha)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — naming
# ═══════════════════════════════════════════════════════════════════════════

def celltype_of(node_name: str) -> str:
    return node_name.rsplit("::", 1)[-1] if "::" in node_name else ""


def short_label(node_name: str) -> str:
    base = node_name.split("::")[0]
    return base.replace("_TF", "").replace("_tf", "").replace("_receptor", "")


# ═══════════════════════════════════════════════════════════════════════════
# Network construction
# ═══════════════════════════════════════════════════════════════════════════

def build_networks(multicell_obj, results, *,
                   cell_type, seeds, ligand_cells,
                   top_ligand_n, top_receptor_n, top_tf_n,
                   before_top_n, per_celltype, verbose):
    """Build and mutually filter the five edge tables.

    Parameters
    ----------
    multicell_obj : single multicell object
    results : DataFrame with columns 'node', 'score'.
    """
    from recon.plot import sankey_paths as _sp

    nets = _sp.build_partial_networks(
        multicell_obj=multicell_obj,
        results=results,
        cell_type=cell_type,
        seeds=seeds or [],
        ligand_cells=ligand_cells or [],
        top_ligand_n=top_ligand_n,
        top_receptor_n=top_receptor_n,
        top_tf_n=top_tf_n,
        before_top_n=before_top_n,
        per_celltype=per_celltype,
        include_before_cells=True,
        verbose=verbose,
    )

    _tf_map = {}
    for net_df in [nets[0], nets[1], nets[3], nets[4]]:
        if "tf_clean" in net_df.columns and "tf" in net_df.columns:
            for clean, orig in zip(net_df["tf_clean"], net_df["tf"]):
                _tf_map[clean] = orig

    _rec_map = {}
    if "receptor_clean" in nets[2].columns and "receptor" in nets[2].columns:
        for clean, orig in zip(nets[2]["receptor_clean"], nets[2]["receptor"]):
            _rec_map[clean] = orig

    def _fmt(df, s, t):
        return (df[[s, t, "weight"]]
                .rename(columns={s: "source", t: "target", "weight": "value"})
                .copy())

    if "receptor_clean" in nets[0].columns:
        br_bt = _fmt(nets[0], "receptor_clean", "tf_clean")
    elif "tf_clean" in nets[0].columns:
        br_bt = _fmt(nets[0], "receptor", "tf_clean")
    else:
        br_bt = _fmt(nets[0], "receptor", "tf")

    bt_l = _fmt(nets[1], "tf_clean", "gene")
    l_r  = _fmt(nets[2], "ligand",   "receptor_clean")

    if "tf_clean" in nets[3].columns:
        r_t = _fmt(nets[3],
                   "receptor_clean" if "receptor_clean" in nets[3].columns else "receptor",
                   "tf_clean")
    else:
        r_t = _fmt(nets[3], "receptor", "tf")

    t_g = _fmt(nets[4], "tf_clean", "gene")

    br_bt = br_bt.copy()
    br_bt["target"] = br_bt["target"].map(_tf_map).fillna(br_bt["target"])
    bt_l = bt_l.copy()
    bt_l["source"] = bt_l["source"].map(_tf_map).fillna(bt_l["source"])
    r_t = r_t.copy()
    r_t["target"] = r_t["target"].map(_tf_map).fillna(r_t["target"])
    t_g = t_g.copy()
    t_g["source"] = t_g["source"].map(_tf_map).fillna(t_g["source"])

    r_t   = r_t[r_t["source"].isin(l_r["target"])]
    t_g   = t_g[t_g["source"].isin(r_t["target"])]
    br_bt = br_bt[br_bt["target"].isin(bt_l["source"])]

    return {
        "upstream_r_tf":   br_bt,
        "upstream_tf_lig": bt_l,
        "lig_rec":         l_r,
        "rec_tf":          r_t,
        "tf_gene":         t_g,
    }


def build_networks_contrast(multicell_objs, results, score_dict, *,
                            cell_type, seeds, ligand_cells,
                            top_ligand_n, top_receptor_n, top_tf_n,
                            before_top_n, per_celltype, verbose):
    """Build networks for contrast mode (two-run comparison)."""
    results_fake = results.copy()
    results_fake["score"] = results_fake["node"].map(
        lambda n: abs(score_dict.get(n, 0.0))
    )
    for idx in results_fake[results_fake["score"] == 0.0].index:
        node = results_fake.loc[idx, "node"]
        node_replaced = "::".join(node.rsplit("-", 1))
        if node_replaced in score_dict:
            results_fake.at[idx, "score"] = abs(score_dict[node_replaced])

    multicell_unique = multicell_objs[list(multicell_objs.keys())[0]]
    multicell_unique.multiplexes["cell_communication"]["layers"][0] = pd.concat(
        [multicell_objs[c].multiplexes["cell_communication"]["layers"][0]
         for c in multicell_objs],
        ignore_index=True,
    )
    multicell_unique.multiplexes["cell_communication"]["layers"][0].drop_duplicates(
        ["source", "target"], inplace=True
    )

    return build_networks(
        multicell_unique, results_fake,
        cell_type=cell_type, seeds=seeds, ligand_cells=ligand_cells,
        top_ligand_n=top_ligand_n, top_receptor_n=top_receptor_n,
        top_tf_n=top_tf_n, before_top_n=before_top_n,
        per_celltype=per_celltype, verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Node set collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_node_sets(edges, seeds=None, cell_type=None):
    """Extract ordered node lists per layer from the filtered edge tables.

    Parameters
    ----------
    seeds : list of str or None
        Raw seed gene names (e.g. ['Nfkb1', 'Tnf']).
    cell_type : str or None
        Receiver cell type name, used for seed expansion.
    """
    br_bt = edges["upstream_r_tf"]
    bt_l  = edges["upstream_tf_lig"]
    l_r   = edges["lig_rec"]
    r_t   = edges["rec_tf"]
    t_g   = edges["tf_gene"]

    sender_recs  = list(pd.unique(br_bt["source"])) if len(br_bt) else []
    sender_tfs   = list(pd.unique(br_bt["target"])) if len(br_bt) else []
    ligands      = list(pd.unique(l_r["source"]))   if len(l_r)   else []
    recv_recs    = list(pd.unique(r_t["source"]))    if len(r_t)   else []
    recv_tfs     = list(pd.unique(r_t["target"]))    if len(r_t)   else []
    recv_genes   = list(pd.unique(t_g["target"]))    if len(t_g)   else []

    # Expand seeds to node format
    seed_nodes = []
    if seeds and cell_type:
        seed_nodes = [f"{s}::{cell_type}" for s in seeds]

    # Deduplicate: TF arc vs gene grid (TF wins)
    tf_set = set(recv_tfs)
    recv_genes = [n for n in recv_genes if n not in tf_set]

    # Deduplicate: seeds vs recv_genes (seeds win)
    seed_set = set(seed_nodes)
    recv_genes = [n for n in recv_genes if n not in seed_set]

    # Deduplicate across sender/receiver (receiver wins)
    recv_rec_set = set(recv_recs)
    recv_tf_set  = set(recv_tfs)
    sender_recs  = [n for n in sender_recs if n not in recv_rec_set]
    sender_tfs   = [n for n in sender_tfs if n not in recv_tf_set]
    ligands      = [n for n in ligands if n not in recv_rec_set]

    # Track bare-name collisions for labelling
    tf_bare   = {short_label(n) for n in recv_tfs}
    gene_bare = {short_label(n) for n in recv_genes}
    seed_bare = {short_label(n) for n in seed_nodes}
    name_collisions = tf_bare & (gene_bare | seed_bare)

    return {
        "sender_recs": sender_recs,
        "sender_tfs":  sender_tfs,
        "ligands":     ligands,
        "recv_recs":   recv_recs,
        "recv_tfs":    recv_tfs,
        "recv_genes":  recv_genes,
        "seed_nodes":  seed_nodes,
        "name_collisions": name_collisions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Geometry layout
# ═══════════════════════════════════════════════════════════════════════════

def compute_layout(nodes, edges, sender_ct_order, sender_cy_by_ct,
                   sender_cx, sender_r, extracell_x_lo, extracell_x_hi,
                   recv_cx, recv_cy, recv_r, nuc_cx):
    """Assign (x, y) positions for every node."""
    positions = {}
    RECV_RY_RATIO = 0.85

    for ct in sender_ct_order:
        ct_recs = [n for n in nodes["sender_recs"] if celltype_of(n) == ct]
        sy = sender_cy_by_ct[ct]
        span = clamp(len(ct_recs) * 20 + 15, 44, 130)
        for node, pt in zip(ct_recs, arc_positions(
                sender_cx, sy, sender_r, len(ct_recs), 180 - span, 180 + span)):
            positions[node] = pt

    for ct in sender_ct_order:
        ct_tfs = [n for n in nodes["sender_tfs"] if celltype_of(n) == ct]
        sy = sender_cy_by_ct[ct]
        span = clamp(len(ct_tfs) * 25 + 15, 48, 162)
        for node, pt in zip(ct_tfs, arc_positions(
                sender_cx, sy, 0.48 * sender_r, len(ct_tfs), -span, span)):
            positions[node] = pt

    mid_x = (extracell_x_lo + extracell_x_hi) / 2
    for ct in sender_ct_order:
        ct_ligs = [n for n in nodes["ligands"] if celltype_of(n) == ct]
        sy = sender_cy_by_ct[ct]
        hs = min(0.85 * sender_r, len(ct_ligs) * 0.35)
        for node, pt in zip(ct_ligs, vertical_positions(mid_x, sy, len(ct_ligs), hs)):
            positions[node] = pt

    l_r = edges["lig_rec"]

    def _receptor_sender_y(rec_node):
        incoming = l_r.loc[l_r["target"] == rec_node, "source"].tolist()
        ys = [sender_cy_by_ct[celltype_of(lig)]
              for lig in incoming if celltype_of(lig) in sender_cy_by_ct]
        return float(np.mean(ys)) if ys else 0.0

    recv_recs_sorted = sorted(nodes["recv_recs"],
                              key=_receptor_sender_y, reverse=True)
    span = clamp(len(recv_recs_sorted) * 18 + 20, 60, 162)
    for node, pt in zip(recv_recs_sorted, arc_positions_ellipse(
            recv_cx, recv_cy, recv_r, recv_r * RECV_RY_RATIO,
            len(recv_recs_sorted), 180 - span, 180 + span)):
        positions[node] = pt

    span = clamp(len(nodes["recv_tfs"]) * 20 + 15, 52, 120)
    for node, pt in zip(nodes["recv_tfs"], arc_positions_ellipse(
            recv_cx, recv_cy, 0.65 * recv_r, 0.65 * recv_r * RECV_RY_RATIO * 1.25,
            len(nodes["recv_tfs"]), 180 - span, 180 + span)):
        positions[node] = pt

    # Nucleus: recv_genes + seed_nodes together in grid
    # Drawn nucleus semi-axes are 0.4*recv_r and 0.4*recv_r*RECV_RY_RATIO
    # Use 90% of that to keep a margin
    nuc_a = 0.40 * recv_r * 0.90
    nuc_b = 0.40 * recv_r * RECV_RY_RATIO * 0.90
    nucleus_nodes = nodes["recv_genes"] + nodes["seed_nodes"]
    for node, pt in zip(nucleus_nodes,
                        grid_positions_in_ellipse(nuc_cx, recv_cy, nuc_a, nuc_b,
                                                  len(nucleus_nodes))):
        positions[node] = pt

    nodes["recv_recs"] = recv_recs_sorted
    return positions


# ═══════════════════════════════════════════════════════════════════════════
# Z-score normalization
# ═══════════════════════════════════════════════════════════════════════════

def zscore_layer(color_dict, node_list):
    present = [n for n in node_list if n in color_dict]
    if len(present) < 2:
        return
    vals = np.array([color_dict[n] for n in present])
    mx = np.abs(vals).max()
    if mx < 1e-9:
        return
    for n, v in zip(present, vals):
        color_dict[n] = v / mx


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers — cells
# ═══════════════════════════════════════════════════════════════════════════

def sender_colors(ct_idx, ct, celltype_colors):
    if celltype_colors and ct in celltype_colors:
        h = celltype_colors[ct]
        rgb = hex_to_rgb(h)
        return {
            "fill":  (*lighten(h, 0.72), 0.30),
            "edge":  (*rgb, 0.7),
            "arc":   (*rgb, 0.85),
            "label": rgb,
        }
    return {
        "fill":  SENDER_PALETTE[ct_idx % len(SENDER_PALETTE)],
        "edge":  (0.35, 0.35, 0.35, 0.7),
        "arc":   (0.35, 0.35, 0.35, 0.85),
        "label": (0.20, 0.20, 0.20),
    }


def receiver_colors(cell_type, celltype_colors):
    if celltype_colors and cell_type and cell_type in celltype_colors:
        h = celltype_colors[cell_type]
        rgb = hex_to_rgb(h)
        return {
            "face":     (*lighten(h, 0.82), 0.30),
            "edge":     (*rgb, 0.75),
            "arc":      (*rgb, 0.85),
            "nuc_face": (*lighten(h, 0.45), 0.62),
            "nuc_edge": (*rgb, 0.92),
            "label":    rgb,
        }
    return {
        **DEFAULT_RECV_COLORS,
        "face": (*DEFAULT_RECV_COLORS["face"][:3], 0.30),
    }


def compute_edge_lw_ranges(edges):
    ranges = {}
    for key, df in edges.items():
        if len(df):
            vmin = float(df["value"].min())
            vmax = float(df["value"].max())
            ranges[key] = (vmin, vmax if vmax > vmin else vmin + 1e-9)
    return ranges


def compute_node_radii(all_nodes, score_dict, base_r, node_size_by_weight,
                       group_lists):
    if not node_size_by_weight or not all_nodes:
        return {n: base_r for n in all_nodes}

    radii = {}
    for grp in group_lists:
        if not grp:
            continue
        raw = np.array([max(score_dict.get(n, 0.0), 0.0) for n in grp])
        sqrt_raw = np.sqrt(raw)
        mean_sqrt = float(sqrt_raw.mean()) if sqrt_raw.mean() > 0 else 1.0
        cap = min(2.0, max(0.5, 2.2 / np.sqrt(max(len(grp), 1))))
        for n, sq in zip(grp, sqrt_raw):
            radii[n] = float(np.clip(
                base_r * (sq / mean_sqrt),
                base_r * 0.4,
                base_r * cap,
            ))

    # Any node not in a group gets base_r
    for n in all_nodes:
        if n not in radii:
            radii[n] = base_r

    return radii


def boost_values_for_color(color_dict, vmax, min_fraction=0.25):
    boosted = color_dict.copy()
    for k, v in boosted.items():
        if v == 0:
            continue
        sign = np.sign(v)
        normed = min(abs(v) / vmax, 1.0)
        rescaled = min_fraction + normed * (1.0 - min_fraction)
        boosted[k] = sign * rescaled * vmax
    return boosted


# ═══════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════════════

def draw_extracellular_region_per_ct(ax, x_lo, x_hi, sender_ct_order,
                                     sender_cy_by_ct, sender_r, fontsize,
                                     celltype_colors):
    if not sender_ct_order:
        return
    for i, ct in enumerate(sender_ct_order):
        sy = sender_cy_by_ct[ct]
        ylo = sy - sender_r * 1.05
        yhi = sy + sender_r * 1.05
        if celltype_colors and ct in celltype_colors:
            rgb = hex_to_rgb(celltype_colors[ct])
            face = (*rgb, 0.25)
            edge = (*rgb, 0.35)
        else:
            pal = SENDER_PALETTE[i % len(SENDER_PALETTE)]
            face = (*pal[:3], 0.25)
            edge = (*pal[:3], 0.35)
        ax.add_patch(FancyBboxPatch(
            (x_lo - 0.15, ylo), (x_hi - x_lo + 0.3), (yhi - ylo),
            boxstyle="round,pad=0.08",
            facecolor=face, edgecolor=edge,
            linewidth=0.8, linestyle="--", zorder=0,
        ))
    top_y = max(sender_cy_by_ct[ct] + sender_r * 1.05
                for ct in sender_ct_order)
    ax.text((x_lo + x_hi) / 2, top_y + 0.25, "extracellular",
            ha="center", va="bottom", fontsize=fontsize,
            color=(0.30, 0.55, 0.50), style="italic", zorder=7)


def draw_sender_cell(ax, cx, cy, r, cols, label, fontsize):
    ax.add_patch(Circle((cx, cy), r,
                        facecolor=cols["fill"], edgecolor=cols["edge"],
                        linewidth=3, zorder=1))
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r,
                     angle=0, theta1=110, theta2=250,
                     color=cols["arc"], lw=3.5, zorder=2))
    ax.text(cx, cy + r + 0.66, label, ha="center", va="top",
            fontsize=fontsize + 10, fontweight="bold", color=cols["label"],
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
            zorder=7)


def draw_receiver_cell(ax, cx, cy, r, ry_ratio, nuc_cx, cols, label, fontsize):
    rx, ry = r, r * ry_ratio
    ax.add_patch(mpatches.Ellipse(
        (cx, cy), 2 * rx, 2 * ry,
        facecolor=cols["face"], edgecolor=cols["edge"],
        linewidth=3, zorder=1,
    ))
    ax.add_patch(Arc((cx, cy), 2 * rx, 2 * ry,
                     angle=0, theta1=100, theta2=260,
                     color=cols["arc"], lw=3.5, zorder=2))
    ax.add_patch(mpatches.Ellipse(
        (nuc_cx, cy), 0.8 * rx, 0.8 * ry,
        facecolor=cols["nuc_face"], edgecolor=cols["nuc_edge"],
        linewidth=3, linestyle="--", zorder=1,
    ))
    ax.text(cx, cy + ry + 0.22, label, ha="center", va="bottom",
            fontsize=fontsize + 10, color=cols["label"],
            fontweight="bold", zorder=7)


def draw_single_edge(ax, src, tgt, value, layer_key, rad,
                     positions, lw_ranges, color_fn,
                     score_filter_dict=None, thresholds=None):
    """Draw a single directed edge: black outline underneath, lighter color on top."""
    if src not in positions or tgt not in positions:
        return
    if score_filter_dict is not None and thresholds is not None:
        threshold = thresholds.get(layer_key, 0.0)
        if abs(score_filter_dict.get(src, 0.0)) < threshold:
            return

    sx, sy = positions[src]
    tx, ty = positions[tgt]
    col = color_fn(src, layer_key)

    if layer_key in lw_ranges:
        vmin, vmax = lw_ranges[layer_key]
        t = (value - vmin) / (vmax - vmin)
        lw = EDGE_LW_MIN + t * (EDGE_LW_MAX - EDGE_LW_MIN)
    else:
        lw = (EDGE_LW_MIN + EDGE_LW_MAX) / 2

    hw = 0.06 + 0.015 * lw
    hl = 0.05 + 0.012 * lw

    # Black outline underneath — strong and visible
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (tx, ty),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"->,head_width={hw:.3f},head_length={hl:.3f}",
        color=(0.0, 0.0, 0.0, 0.7),
        linewidth=lw + 1.5, zorder=2,
    ))
    # Lighter color on top
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (tx, ty),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"->,head_width={hw:.3f},head_length={hl:.3f}",
        color=(*col[:3], 1.0), linewidth=lw, zorder=3,
    ))


def draw_single_node(ax, node_name, role, cell_cx, cell_cy, positions, node_radii,
                     default_r, fill_fn, halo, show_labels, fontsize,
                     name_collisions):
    if node_name not in positions:
        return
    nx, ny = positions[node_name]

    draw_role = "receptor" if role == "recv_receptor" else role

    fill = fill_fn(node_name, draw_role)
    r = node_radii.get(node_name, default_r)

    if halo:
        tc = TYPE_RGBA.get(draw_role, (0.7, 0.7, 0.7, 0.9))
        ax.add_patch(Circle((nx, ny), r * 1.55,
                            facecolor=(*tc[:3], 0.55),
                            edgecolor=(0.10, 0.10, 0.10, 0.45),
                            linewidth=0.5, zorder=4))

    ax.add_patch(Circle((nx, ny), r,
                        facecolor=fill,
                        edgecolor=(0.12, 0.12, 0.12, 0.85),
                        linewidth=0.9, zorder=5))

    if show_labels:
        lbl = short_label(node_name)
        if lbl in name_collisions:
            lbl = f"{lbl} ({'TF' if draw_role == 'tf' else 'gene'})"

        cy_ref = cell_cy if cell_cy is not None else 0.0
        if cell_cx is not None:
            if nx <= cell_cx and abs(ny - cy_ref) < r * 2:
                ha = "left"
                label_x = nx + r + 0.07
            elif nx <= cell_cx:
                ha = "right"
                label_x = nx - r - 0.07
            else:
                ha = "left"
                label_x = nx + r + 0.07
        else:
            label_x = nx + r + 0.08
            ha = "left"

        fs = fontsize * 1.10
        if role == "recv_receptor":
            fs = fontsize * 1.30

        fw = "bold" if draw_role in ("receptor", "ligand", "seed") else "normal"

        ax.text(label_x, ny, lbl, fontsize=fs, fontweight=fw,
                ha=ha, va="center",
                color=(0.10, 0.10, 0.10), zorder=6, clip_on=True)


# ═══════════════════════════════════════════════════════════════════════════
# Legend helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_type_legend(ax, fontsize, show_seeds=False):
    def _type_handle(role, label):
        tc = TYPE_RGBA[role][:3]
        return mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=11,
            markerfacecolor=(*tc, 0.7),
            markeredgecolor=tc, markeredgewidth=3.2, label=label,
        )
    handles = [
        _type_handle("receptor", "Receptor"),
        _type_handle("tf",       "TF"),
        _type_handle("ligand",   "Ligand"),
        _type_handle("gene",     "Gene"),
    ]
    if show_seeds:
        handles.append(_type_handle("seed", "Seed"))
    ax.legend(handles=handles, loc="upper right", fontsize=fontsize - 1,
              framealpha=0.75, edgecolor="lightgray")


def draw_contrast_legend(ax, scheme, delta_vmax, normalized, fontsize,
                         show_seeds=False):
    def _type_handle(role, label):
        tc = TYPE_RGBA[role][:3]
        return mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=11,
            markerfacecolor=(0.95, 0.95, 0.95, 1.0),
            markeredgecolor=tc, markeredgewidth=3.2, label=label,
        )
    handles = [
        _type_handle("receptor", "Receptor"),
        _type_handle("tf",       "TF"),
        _type_handle("ligand",   "Ligand"),
        _type_handle("gene",     "Gene"),
    ]
    if show_seeds:
        handles.append(_type_handle("seed", "Seed"))

    if scheme == "sex":
        contrast_labels = [
            ("A-enriched (hi)",  -delta_vmax),
            ("neutral",           0.0),
            ("B-enriched (lo)",   delta_vmax),
        ]
    else:
        contrast_labels = [
            ("warm-enriched (hi)",  -delta_vmax),
            ("neutral",              0.0),
            ("cold-enriched (lo)",   delta_vmax),
        ]

    handles += [
        mlines.Line2D([], [], lw=2, label=lbl,
                      color=parse_rgba(delta_to_rgba(v, delta_vmax, 0.8,
                                                     scheme=scheme)))
        for lbl, v in contrast_labels
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=fontsize - 1,
              framealpha=0.75, edgecolor="lightgray")
    if normalized:
        ax.text(0.99, 0.01, "receiver scores z-scored per layer",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=max(7, fontsize - 4), color="gray", style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# Shared geometry setup & orchestration
# ═══════════════════════════════════════════════════════════════════════════

def compute_global_geometry(nodes):
    sender_ct_order = list(dict.fromkeys(
        celltype_of(n) for n in nodes["ligands"]
    ))
    n_senders = max(len(sender_ct_order), 1)

    SENDER_R  = min(2.4, 10.0 / n_senders) * 1.20
    V_GAP     = SENDER_R * 0.45
    total_h   = n_senders * 2 * SENDER_R + (n_senders - 1) * V_GAP
    RECV_R    = max(2.8, total_h / 2.0) * 0.80
    RECV_RY_RATIO = 0.85
    BUFFER    = 0.6

    if n_senders == 1:
        sender_ys = [0.0]
    else:
        sender_ys = list(np.linspace(
            -(total_h / 2) + SENDER_R,
            (total_h / 2) - SENDER_R,
            n_senders,
        ))
    sender_cy_by_ct = {ct: sender_ys[i] for i, ct in enumerate(sender_ct_order)}

    sender_cx      = SENDER_R + 0.5
    extracell_x_lo = sender_cx + SENDER_R + BUFFER
    extracell_x_hi = extracell_x_lo + 2.8
    recv_cx        = extracell_x_hi + BUFFER + RECV_R
    recv_cy        = sender_ys[-1] + SENDER_R * 1.10 - RECV_R
    NUC_CX         = recv_cx + 0.10 * RECV_R

    return {
        "sender_ct_order": sender_ct_order,
        "n_senders": n_senders,
        "SENDER_R": SENDER_R,
        "RECV_R": RECV_R,
        "RECV_RY_RATIO": RECV_RY_RATIO,
        "total_h": total_h,
        "sender_cy_by_ct": sender_cy_by_ct,
        "sender_cx": sender_cx,
        "extracell_x_lo": extracell_x_lo,
        "extracell_x_hi": extracell_x_hi,
        "recv_cx": recv_cx,
        "recv_cy": recv_cy,
        "NUC_CX": NUC_CX,
    }


def setup_figure(geo, figsize=None):
    x_min = -geo["SENDER_R"] * 0.3
    x_max = geo["recv_cx"] + geo["RECV_R"] + 1.5
    y_max = max(geo["total_h"] / 2 + geo["SENDER_R"] + 0.8,
                geo["RECV_R"] * geo["RECV_RY_RATIO"] + 1.5)
    if figsize is None:
        fw = max(14, (x_max - x_min) * 1.1)
        fh = max(8, 2 * y_max * 0.9)
        figsize = (fw, fh)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    return fig, ax, y_max


def draw_cells_and_edges(ax, nodes, edges, geo, *,
                         cell_type, celltype_colors, celltype_display_names,
                         label_fontsize, node_radii, node_type_halo,
                         show_labels, show_seeds, fill_fn, edge_color_fn,
                         seed_label=None, seed_label_fontsize=10,
                         score_filter_dict=None, lfc_thresholds=None):
    g = geo

    for i, ct in enumerate(g["sender_ct_order"]):
        sy = g["sender_cy_by_ct"][ct]
        cols = sender_colors(i, ct, celltype_colors)
        ct_label = (celltype_display_names or {}).get(ct, ct)
        draw_sender_cell(ax, g["sender_cx"], sy, g["SENDER_R"], cols,
                         ct_label, label_fontsize)

    rcols = receiver_colors(cell_type, celltype_colors)
    recv_label = ((celltype_display_names or {}).get(cell_type, cell_type)
                  if cell_type else "Receiver")
    draw_receiver_cell(ax, g["recv_cx"], g["recv_cy"], g["RECV_R"],
                       g["RECV_RY_RATIO"], g["NUC_CX"], rcols, recv_label,
                       label_fontsize)

    if seed_label is not None:
        ax.text(g["NUC_CX"], g["recv_cy"], seed_label,
                ha="center", va="center",
                fontsize=seed_label_fontsize + 10,
                fontweight="bold", zorder=11)

    positions = compute_layout(
        nodes, edges, g["sender_ct_order"], g["sender_cy_by_ct"],
        g["sender_cx"], g["SENDER_R"], g["extracell_x_lo"], g["extracell_x_hi"],
        g["recv_cx"], g["recv_cy"], g["RECV_R"], g["NUC_CX"],
    )

    lw_ranges = compute_edge_lw_ranges(edges)

    edge_configs = [
        ("upstream_r_tf",   0.30),
        ("upstream_tf_lig", 0.20),
        ("lig_rec",         0.10),
        ("rec_tf",          0.25),
    ]
    for layer_key, rad in edge_configs:
        for _, row in edges[layer_key].iterrows():
            draw_single_edge(
                ax, row["source"], row["target"], row["value"],
                layer_key, rad, positions, lw_ranges, edge_color_fn,
                score_filter_dict=score_filter_dict,
                thresholds=lfc_thresholds,
            )

    NODE_R = max(0.22, g["RECV_R"] * 0.055)
    node_draw_specs = [
        (nodes["sender_recs"], "receptor",      g["sender_cx"], None),
        (nodes["sender_tfs"],  "tf",            g["sender_cx"], None),
        (nodes["ligands"],     "ligand",        None,           None),
        (nodes["recv_recs"],   "recv_receptor", g["recv_cx"],   g["recv_cy"]),
        (nodes["recv_tfs"],    "tf",            g["recv_cx"],   g["recv_cy"]),
        (nodes["recv_genes"],  "gene",          g["recv_cx"],   g["recv_cy"]),
    ]
    if show_seeds:
        node_draw_specs.append(
            (nodes["seed_nodes"], "seed", g["recv_cx"], g["recv_cy"]),
        )
    for node_list, role, cell_cx, cell_cy in node_draw_specs:
        for n in node_list:
            draw_single_node(
                ax, n, role, cell_cx, cell_cy, positions, node_radii,
                NODE_R, fill_fn, node_type_halo,
                show_labels, label_fontsize, nodes["name_collisions"],
            )

    return positions


def add_section_labels(ax, geo, y_max, label_fontsize):
    g = geo
    for x, lbl in [(g["sender_cx"], "Sender cells"),
                   ((g["extracell_x_lo"] + g["extracell_x_hi"]) / 2, "Ligands"),
                   (g["recv_cx"], "Receiver")]:
        ax.text(x, -y_max + 0.3, lbl, ha="center", va="bottom",
                fontsize=label_fontsize, color="gray", style="italic")