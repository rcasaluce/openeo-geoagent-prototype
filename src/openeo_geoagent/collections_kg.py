#!/usr/bin/env python
"""
KNOWLEDGE GRAPHS

Generate a "knowledge graph" (JSON) for collections exposed by an openEO/CDSE backend.

For each collection:
- fetch metadata via describe_collection
- extract band information (name, common_name, wavelength, etc.)
- assign logical roles (RED, GREEN, BLUE, NIR, SWIR1, SWIR2, RED_EDGE, VV, VH, etc.)
- classify sensor type (optical / sar / other)
- automatically determine which standard indices are supported
  using an index catalog (INDEX_DEFS) and the available logical roles.

Output: collections_kg.json
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple

import openeo

# ---------------------------------------------------------------------
# 1) Index catalog (scientifically sensible, but generic)
# ---------------------------------------------------------------------

INDEX_DEFS = {
    # --------------------------------------------------
    # Base vegetation
    # --------------------------------------------------
    "NDVI": {
        "label": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "required_roles": ["NIR", "RED"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "greenness", "biomass"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
        },
    },
    "EVI": {
        "label": "Enhanced Vegetation Index",
        "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
        "required_roles": ["BLUE", "RED", "NIR"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "greenness", "biomass"],
        "role_wavelength_nm_ranges": {
            "BLUE": [400, 520],
            "RED": [600, 700],
            "NIR": [750, 900],
        },
    },
    "RGB": {
        "label": "True color RGB",
        "formula": "Stack of RED, GREEN, BLUE reflectances",
        "required_roles": ["RED", "GREEN", "BLUE"],
        "sensor_type": "optical",
        "phenomenon": ["visual", "true_color", "rgb"],
        "role_wavelength_nm_ranges": {
            "BLUE": [400, 520],
            "GREEN": [500, 600],
            "RED": [600, 700],
        },
    },

    # --------------------------------------------------
    # Advanced vegetation / soil-adjusted
    # --------------------------------------------------
    "SAVI": {
        "label": "Soil Adjusted Vegetation Index",
        "formula": "(NIR - RED) * (1 + L) / (NIR + RED + L), L≈0.5",
        "required_roles": ["NIR", "RED"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "soil_adjusted", "sparse_canopy"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
        },
    },
    "OSAVI": {
        "label": "Optimized Soil Adjusted Vegetation Index",
        "formula": "(NIR - RED) * 1.16 / (NIR + RED + 0.16)",
        "required_roles": ["NIR", "RED"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "soil_adjusted", "sparse_canopy"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
        },
    },
    "MSAVI": {
        "label": "Modified Soil Adjusted Vegetation Index",
        "formula": "(2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED))) / 2",
        "required_roles": ["NIR", "RED"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "soil_adjusted", "sparse_canopy"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
        },
    },
    "MSAVI2": {
        "label": "Modified Soil Adjusted Vegetation Index 2",
        "formula": "(2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED))) / 2",
        "required_roles": ["NIR", "RED"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "soil_adjusted", "sparse_canopy"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
        },
    },
    "GNDVI": {
        "label": "Green Normalized Difference Vegetation Index",
        "formula": "(NIR - GREEN) / (NIR + GREEN)",
        "required_roles": ["NIR", "GREEN"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "chlorophyll", "stress"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "GREEN": [500, 600],
        },
    },
    "NDRE": {
        "label": "Normalized Difference Red Edge Index",
        "formula": "(NIR - RED_EDGE) / (NIR + RED_EDGE)",
        "required_roles": ["NIR", "RED_EDGE"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "chlorophyll", "stress", "red_edge"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED_EDGE": [700, 750],
        },
    },
    "ARVI": {
        "label": "Atmospherically Resistant Vegetation Index",
        "formula": "(NIR - (2*RED - BLUE)) / (NIR + (2*RED - BLUE))",
        "required_roles": ["NIR", "RED", "BLUE"],
        "sensor_type": "optical",
        "phenomenon": ["vegetation", "greenness", "atmosphere_resistant"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "RED": [600, 700],
            "BLUE": [400, 520],
        },
    },

    # --------------------------------------------------
    # Water / wetness
    # --------------------------------------------------
    "NDWI": {
        "label": "Normalized Difference Water Index (Green/NIR variant)",
        "formula": "(GREEN - NIR) / (GREEN + NIR)",
        "required_roles": ["GREEN", "NIR"],
        "sensor_type": "optical",
        "phenomenon": ["water", "wetness", "vegetation_water_content"],
        "role_wavelength_nm_ranges": {
            "GREEN": [500, 600],
            "NIR": [750, 900],
        },
    },
    "MNDWI": {
        "label": "Modified Normalized Difference Water Index",
        "formula": "(GREEN - SWIR1) / (GREEN + SWIR1)",
        "required_roles": ["GREEN", "SWIR1"],
        "sensor_type": "optical",
        "phenomenon": ["water", "surface_water", "urban_water"],
        "role_wavelength_nm_ranges": {
            "GREEN": [500, 600],
            "SWIR1": [1500, 1800],
        },
    },
    "AWEI_SH": {
        "label": "Automated Water Extraction Index (shadow version)",
        "formula": "4*(GREEN - SWIR1) - (0.25*NIR + 2.75*SWIR2)",
        "required_roles": ["GREEN", "NIR", "SWIR1", "SWIR2"],
        "sensor_type": "optical",
        "phenomenon": ["water", "surface_water", "shadows"],
        "role_wavelength_nm_ranges": {
            "GREEN": [500, 600],
            "NIR": [750, 900],
            "SWIR1": [1500, 1800],
            "SWIR2": [2000, 2500],
        },
    },
    "AWEI_NSH": {
        "label": "Automated Water Extraction Index (no-shadow version)",
        "formula": "BLUE + 2.5*GREEN - 1.5*(NIR + SWIR1) - 0.25*SWIR2",
        "required_roles": ["BLUE", "GREEN", "NIR", "SWIR1", "SWIR2"],
        "sensor_type": "optical",
        "phenomenon": ["water", "surface_water"],
        "role_wavelength_nm_ranges": {
            "BLUE": [400, 520],
            "GREEN": [500, 600],
            "NIR": [750, 900],
            "SWIR1": [1500, 1800],
            "SWIR2": [2000, 2500],
        },
    },

    # --------------------------------------------------
    # Snow / cryosphere
    # --------------------------------------------------
    "NDSI": {
        "label": "Normalized Difference Snow Index",
        "formula": "(GREEN - SWIR1) / (GREEN + SWIR1)",
        "required_roles": ["GREEN", "SWIR1"],
        "sensor_type": "optical",
        "phenomenon": ["snow", "cryosphere", "snow_cover"],
        "role_wavelength_nm_ranges": {
            "GREEN": [500, 600],
            "SWIR1": [1500, 1800],
        },
    },

    # --------------------------------------------------
    # Fire / burned area
    # --------------------------------------------------
    "NBR": {
        "label": "Normalized Burn Ratio",
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "required_roles": ["NIR", "SWIR2"],
        "sensor_type": "optical",
        "phenomenon": ["fire", "burned_area", "severity"],
        "role_wavelength_nm_ranges": {
            "NIR": [750, 900],
            "SWIR2": [2000, 2500],
        },
    },
    "NBR2": {
        "label": "Normalized Burn Ratio 2",
        "formula": "(SWIR1 - SWIR2) / (SWIR1 + SWIR2)",
        "required_roles": ["SWIR1", "SWIR2"],
        "sensor_type": "optical",
        "phenomenon": ["fire", "burned_area", "moisture"],
        "role_wavelength_nm_ranges": {
            "SWIR1": [1500, 1800],
            "SWIR2": [2000, 2500],
        },
    },
    "BAI": {
        "label": "Burned Area Index",
        "formula": "1 / ((RED - 0.1)**2 + (NIR - 0.06)**2)",
        "required_roles": ["RED", "NIR"],
        "sensor_type": "optical",
        "phenomenon": ["fire", "burned_area"],
        "role_wavelength_nm_ranges": {
            "RED": [600, 700],
            "NIR": [750, 900],
        },
    },

    # --------------------------------------------------
    # Urban / built-up / bare soil
    # --------------------------------------------------
    "NDBI": {
        "label": "Normalized Difference Built-up Index",
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "required_roles": ["SWIR1", "NIR"],
        "sensor_type": "optical",
        "phenomenon": ["urban", "built_up", "impervious"],
        "role_wavelength_nm_ranges": {
            "SWIR1": [1500, 1800],
            "NIR": [750, 900],
        },
    },
    "BSI": {
        "label": "Bare Soil Index",
        "formula": "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        "required_roles": ["SWIR1", "RED", "NIR", "BLUE"],
        "sensor_type": "optical",
        "phenomenon": ["bare_soil", "degradation", "desertification"],
        "role_wavelength_nm_ranges": {
            "SWIR1": [1500, 1800],
            "RED": [600, 700],
            "NIR": [750, 900],
            "BLUE": [400, 520],
        },
    },

    # --------------------------------------------------
    # SAR / radar
    # --------------------------------------------------
    "VV_VH_RATIO": {
        "label": "VV/VH backscatter ratio",
        "formula": "VV / VH (in linear space)",
        "required_roles": ["VV", "VH"],
        "sensor_type": "sar",
        "phenomenon": ["structure", "surface", "roughness"],
        "role_wavelength_nm_ranges": {},
    },
    "RVI": {
        "label": "Radar Vegetation Index (simplified VV/VH version)",
        "formula": "4 * VH / (VV + VH)",
        "required_roles": ["VV", "VH"],
        "sensor_type": "sar",
        "phenomenon": ["vegetation", "structure", "volume_scattering"],
        "role_wavelength_nm_ranges": {},
    },
}

# ---------------------------------------------------------------------
# 2) KG data models
# ---------------------------------------------------------------------


@dataclass
class BandProfile:
    band_id: str
    common_name: Optional[str] = None
    center_wavelength_nm: Optional[float] = None
    full_width_half_max_nm: Optional[float] = None
    description: Optional[str] = None
    gsd: Optional[float] = None
    data_type: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    roles: List[str] = field(default_factory=list)


@dataclass
class CollectionProfile:
    collection_id: str
    title: Optional[str]
    description: Optional[str]
    platform: Optional[str]
    instrument: Optional[str]
    license: Optional[str]
    keywords: List[str]
    sensor_type: str  # "optical", "sar", "other"
    temporal_extent: Optional[List[Optional[str]]]
    spatial_extent: Optional[Dict[str, Optional[float]]]
    native_resolution: Optional[Dict[str, Optional[float]]]
    crs: Any
    bands: List[BandProfile]
    logical_roles: Dict[str, str]
    supported_indices: List[str]


# ---------------------------------------------------------------------
# 3) Utilities to normalize wavelengths and assign logical roles
# ---------------------------------------------------------------------


def _normalize_wavelength_nm(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return None
        # If the value is in µm (e.g. 0.665), convert it to nm
        if v < 50:
            return v * 1000.0
        return v
    return None


def _extract_band_list(raw_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Look for band definitions in this priority order:

    1. summaries.eo:bands / summaries.bands  (often richer: wavelength, common_name, etc.)
    2. cube:dimensions / bands               (fallback, often just a list of names)
    """

    # 1) summaries.eo:bands / summaries.bands  -> preferred source
    summ = raw_meta.get("summaries", {}) or {}
    eo_bands = summ.get("eo:bands") or summ.get("bands")
    if isinstance(eo_bands, list) and eo_bands:
        out: List[Dict[str, Any]] = []
        for b in eo_bands:
            if isinstance(b, dict):
                name = b.get("name") or b.get("id")
                if not name:
                    continue
                band_info = dict(b)
                band_info.setdefault("name", name)
                out.append(band_info)
        if out:
            return out

    # 2) cube:dimensions / bands  -> fallback
    dims = raw_meta.get("cube:dimensions", {}) or {}
    for _dim_name, dim in dims.items():
        if isinstance(dim, dict) and dim.get("type") == "bands":
            values = dim.get("values")
            if isinstance(values, list) and values:
                out = []
                for v in values:
                    if isinstance(v, dict):
                        name = v.get("name") or v.get("id")
                        if not name:
                            continue
                        band_info = dict(v)
                        band_info.setdefault("name", name)
                        out.append(band_info)
                    elif isinstance(v, str):
                        out.append({"name": v})
                if out:
                    return out

    return []


def _logical_roles_for_band(name: str, info: Dict[str, Any]) -> List[str]:
    """
    Assign one or more logical roles to a band based on:
    - name
    - common_name / eo:common_name
    - description
    - wavelength / eo:center_wavelength
    """
    roles: List[str] = []
    name_l = name.lower()
    cn = str(info.get("common_name") or info.get("eo:common_name") or "").lower()
    desc = str(info.get("description") or "").lower()

    wl_nm: Optional[float] = None
    for k in ("center_wavelength", "wavelength", "wavelength_nm", "eo:center_wavelength"):
        if k in info:
            wl_nm = _normalize_wavelength_nm(info[k])
            break

    # SAR roles
    for tok in (name_l, cn, desc):
        if "vv" in tok and "vh" not in tok:
            roles.append("VV")
        if "vh" in tok and "vv" not in tok:
            roles.append("VH")
        if "hv" in tok:
            roles.append("HV")
        if "hh" in tok:
            roles.append("HH")

    # SCL / mask
    if "scl" in name_l or "scene classification" in desc or "classification" in desc:
        roles.append("SCL")

    # RED_EDGE from names
    if any(t in cn for t in ["re", "rededge", "red_edge"]) or "red edge" in desc:
        roles.append("RED_EDGE")
    if "rededge" in name_l or "red_edge" in name_l:
        roles.append("RED_EDGE")

    # Optical roles from names / common_name
    for tok in (cn, name_l):
        if not tok:
            continue
        if tok.startswith("red"):
            roles.append("RED")
        if tok.startswith("green"):
            roles.append("GREEN")
        if tok.startswith("blue"):
            roles.append("BLUE")
        if tok.startswith("nir") or "nir " in tok:
            roles.append("NIR")
        if "swir" in tok:
            # Distinguish SWIR1 and SWIR2 if wavelength is available
            if wl_nm:
                if 1400 <= wl_nm <= 1900:
                    roles.append("SWIR1")
                elif 2000 <= wl_nm <= 2600:
                    roles.append("SWIR2")
                else:
                    roles.append("SWIR")
            else:
                roles.append("SWIR")

        # Sentinel-2 style red-edge common names (nir08, re1, re2, swir16, swir22, etc.)
        if any(tok.startswith(x) for x in ["re1", "re2", "re3", "re4", "rededge1", "rededge2"]):
            roles.append("RED_EDGE")

    # Optical roles from wavelengths (fallback)
    if wl_nm is not None:
        if 430 <= wl_nm <= 520 and "BLUE" not in roles:
            roles.append("BLUE")
        if 520 < wl_nm <= 600 and "GREEN" not in roles:
            roles.append("GREEN")
        if 600 < wl_nm <= 700 and "RED" not in roles:
            roles.append("RED")
        # red-edge ~700–750 nm
        if 700 <= wl_nm <= 750 and "RED_EDGE" not in roles and "NIR" not in roles:
            roles.append("RED_EDGE")
        if 750 < wl_nm <= 900 and "NIR" not in roles:
            roles.append("NIR")
        if 1400 <= wl_nm <= 1900 and "SWIR1" not in roles:
            roles.append("SWIR1")
        if 2000 <= wl_nm <= 2600 and "SWIR2" not in roles:
            roles.append("SWIR2")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _sensor_type_from_bands(bands_meta: List[Dict[str, Any]]) -> str:
    """
    bands_meta: list of raw band dictionaries (as exposed by STAC/openEO metadata).
    """
    names = [str(b.get("name") or "").lower() for b in bands_meta]
    if any(n in {"vv", "vh", "hh", "hv"} for n in names):
        return "sar"
    # Coarse heuristic: optical if any band has wavelength in visible/NIR/SWIR ranges
    for b in bands_meta:
        wl_nm = None
        for k in ("center_wavelength", "wavelength", "wavelength_nm", "eo:center_wavelength"):
            if k in b:
                wl_nm = _normalize_wavelength_nm(b[k])
                break
        if wl_nm and 350 <= wl_nm <= 2600:
            return "optical"
    return "other"


# ---------------------------------------------------------------------
# 4) Collection ↔ supported index matching
# ---------------------------------------------------------------------


def _compute_logical_mapping(bands: List[BandProfile]) -> Dict[str, str]:
    """
    Choose one band for each logical role (RED, NIR, SWIR1, etc.).
    If multiple bands share the same role, choose the one with wavelength
    closest to the typical center (when available).
    """
    IDEAL = {
        "BLUE": 480,
        "GREEN": 560,
        "RED": 660,
        "RED_EDGE": 720,
        "NIR": 840,
        "SWIR1": 1600,
        "SWIR2": 2200,
    }

    role_candidates: Dict[str, List[Tuple[BandProfile, float]]] = {}
    for b in bands:
        wl = b.center_wavelength_nm
        for r in b.roles:
            if r in ("VV", "VH", "HH", "HV", "SCL", "SWIR"):
                continue
            ideal = IDEAL.get(r)
            if ideal and wl:
                score = abs(wl - ideal)
            else:
                score = 0.0
            role_candidates.setdefault(r, []).append((b, score))

    mapping: Dict[str, str] = {}
    for r, cand in role_candidates.items():
        cand_sorted = sorted(cand, key=lambda x: x[1])
        mapping[r] = cand_sorted[0][0].band_id

    # Generic SWIR -> SWIR1 / SWIR2 fallbacks
    if any("SWIR1" in b.roles for b in bands):
        swir1 = [b for b in bands if "SWIR1" in b.roles][0]
        mapping.setdefault("SWIR1", swir1.band_id)
    if any("SWIR2" in b.roles for b in bands):
        swir2 = [b for b in bands if "SWIR2" in b.roles][0]
        mapping.setdefault("SWIR2", swir2.band_id)

    if "SWIR1" not in mapping:
        swir_generic = [b for b in bands if "SWIR" in b.roles]
        if swir_generic:
            mapping["SWIR1"] = swir_generic[0].band_id

    # SAR polarizations + SCL
    for pol in ("VV", "VH", "HH", "HV", "SCL"):
        cand = [b for b in bands if pol in b.roles]
        if cand:
            mapping[pol] = cand[0].band_id

    return mapping


def _get_band_wavelength(bands: List[BandProfile], band_id: str) -> Optional[float]:
    for b in bands:
        if b.band_id == band_id:
            return b.center_wavelength_nm
    return None


def _is_index_supported_for_collection(
    index_id: str,
    sensor_type: str,
    logical_roles: Dict[str, str],
    bands: List[BandProfile],
) -> bool:
    idx = INDEX_DEFS[index_id]
    if idx["sensor_type"] != sensor_type:
        return False

    for role in idx["required_roles"]:
        if role not in logical_roles:
            return False

    ranges = idx.get("role_wavelength_nm_ranges") or {}
    for role, (lo, hi) in ranges.items():
        band_id = logical_roles.get(role)
        if not band_id:
            return False
        wl = _get_band_wavelength(bands, band_id)
        if wl is None:
            return False
        if not (lo <= wl <= hi):
            return False

    return True


# ---------------------------------------------------------------------
# 5) Main ETL
# ---------------------------------------------------------------------


def build_kg(
    endpoint: str = "https://openeo.dataspace.copernicus.eu",
    max_collections: Optional[int] = None,
) -> Dict[str, Any]:
    conn = openeo.connect(endpoint)
    coll_list = conn.list_collections()
    profiles: List[CollectionProfile] = []

    for i, c in enumerate(coll_list):
        cid = c["id"]
        if max_collections is not None and i >= max_collections:
            break

        print(f"[{i+1}/{len(coll_list)}] Processing collection {cid} ...")
        meta = conn.describe_collection(cid)

        raw = getattr(meta, "metadata", None) or getattr(meta, "_metadata", None)
        if not isinstance(raw, dict):
            try:
                raw = dict(meta)  # VisualDict -> dict
            except TypeError:
                raw = meta

        # -----------------------------------------------------------------
        # Extract band information
        # -----------------------------------------------------------------
        bands_meta = _extract_band_list(raw)
        if not bands_meta:
            print("  -> no bands metadata, skipping.")
            continue

        bands: List[BandProfile] = []
        for b in bands_meta:
            name = str(b.get("name") or b.get("id") or "").strip()
            if not name:
                continue

            cn = b.get("common_name") or b.get("eo:common_name")
            desc = b.get("description")

            wl_nm = None
            for k in ("center_wavelength", "wavelength", "wavelength_nm", "eo:center_wavelength"):
                if k in b:
                    wl_nm = _normalize_wavelength_nm(b[k])
                    break

            fwhm_nm = None
            if "eo:full_width_half_max" in b:
                fwhm_nm = _normalize_wavelength_nm(b["eo:full_width_half_max"])

            gsd = b.get("gsd")
            data_type = b.get("type")
            scale = b.get("scale")
            offset = b.get("offset")

            roles = _logical_roles_for_band(name, b)

            bands.append(
                BandProfile(
                    band_id=name,
                    common_name=cn,
                    center_wavelength_nm=wl_nm,
                    full_width_half_max_nm=fwhm_nm,
                    description=desc,
                    gsd=gsd,
                    data_type=data_type,
                    scale=scale,
                    offset=offset,
                    roles=roles,
                )
            )

        # -----------------------------------------------------------------
        # Per-collection info: sensor type, extents, etc.
        # -----------------------------------------------------------------
        sensor_type = _sensor_type_from_bands(bands_meta)

        dims = raw.get("cube:dimensions", {}) or {}
        t_dim = dims.get("t", {})
        temporal_extent = t_dim.get("extent")

        x_dim = dims.get("x", {})
        y_dim = dims.get("y", {})
        spatial_extent: Optional[Dict[str, Optional[float]]] = None
        if x_dim.get("extent") and y_dim.get("extent"):
            try:
                spatial_extent = {
                    "west": x_dim["extent"][0],
                    "east": x_dim["extent"][1],
                    "south": y_dim["extent"][0],
                    "north": y_dim["extent"][1],
                }
            except Exception:
                spatial_extent = None

        native_resolution = {
            "x": x_dim.get("step"),
            "y": y_dim.get("step"),
        }

        crs = x_dim.get("reference_system") or y_dim.get("reference_system")

        title = raw.get("title") or c.get("title")
        description = raw.get("description") or c.get("description")

        props = raw.get("properties", {}) or {}
        platform = raw.get("platform") or props.get("platform")
        instrument = raw.get("instrument") or props.get("instrument")
        license_ = raw.get("license") or props.get("license")
        keywords = raw.get("keywords") or props.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = [keywords]

        logical_roles = _compute_logical_mapping(bands)

        supported_indices: List[str] = []
        for idx_id in INDEX_DEFS.keys():
            if _is_index_supported_for_collection(
                index_id=idx_id,
                sensor_type=sensor_type,
                logical_roles=logical_roles,
                bands=bands,
            ):
                supported_indices.append(idx_id)

        prof = CollectionProfile(
            collection_id=cid,
            title=title,
            description=description,
            platform=platform,
            instrument=instrument,
            license=license_,
            keywords=keywords,
            sensor_type=sensor_type,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            native_resolution=native_resolution,
            crs=crs,
            bands=bands,
            logical_roles=logical_roles,
            supported_indices=supported_indices,
        )
        profiles.append(prof)

    out: Dict[str, Any] = {
        "endpoint": endpoint,
        "indices_catalog": INDEX_DEFS,
        "collections": [
            {
                **{
                    k: v
                    for k, v in asdict(p).items()
                    if k != "bands"
                },
                "bands": [asdict(b) for b in p.bands],
            }
            for p in profiles
        ],
    }
    return out


if __name__ == "__main__":
    kg = build_kg()
    with open("collections_kg.json", "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print("Written collections_kg.json")
