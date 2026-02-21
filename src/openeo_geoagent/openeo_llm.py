
# nl_to_openeo.py — WITH MULTI INDEXES + GENERIC BAND-MATH + MULTI-ASSET/MULTI-BAND
# ===============================================================
# NL → DSL → openEO ProcessBuilder-based process graph
#
# Pipeline:
#   1) LLM: NL → DSLRequest (senza dettagli di bande)
#   2) Validazione DSL + arricchimento usando KG (band mapping, allowed indices)
#   3) Costruzione process graph openEO con ProcessBuilder
#
# Estensioni:
#   - supporto per più indici nello stesso process graph (multi-output)
#   - supporto per band-math generico via DSL.band_math (espressioni sui ruoli logici)
#   - supporto per packing output:
#       * "multi_asset": un save_result per indice/band-math (job multi-asset)
#       * "multi_band" : stacking di più indici raster come bande in un singolo asset
#
#

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Union, Tuple, Set
import json, re, ast, os
from pathlib import Path
from datetime import datetime

# Expert EO helpers (temporal mosaicking, spatial filters, morphology, temporal filters, UDF)
# definiti in eo_expert_ops.py e richiamati qui dal DSL.
from .eo_expert_ops import (
    ExpertOps,
    apply_expert_ops,
    get_time_dimension_name,
)


import logging
logger = logging.getLogger(__name__)

# LangChain
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# pydantic v2
from pydantic import BaseModel, Field, field_validator, ValidationError

# DataCube per costruire il grafo in modo connection-less
from openeo.processes import save_result as oe_save_result

# DataCube del client openEO
from openeo.rest.datacube import DataCube

# openEO client
try:
    import openeo
except Exception:
    openeo = None


# ==========================================================
# 0) LLM key handling
# ==========================================================

from pathlib import Path
import os

def _read_api_key() -> str:
    """
    Lookup order:
      1) env TOKEN_GEMINI
      2) local file ./GEMINI
      3) local file ./.keys/GEMINI
    """
    k = os.getenv("TOKEN_GEMINI")
    if k and k.strip():
        return k.strip()

    p1 = Path("GEMINI")
    if p1.exists():
        val = p1.read_text(encoding="utf-8").strip()
        if val:
            return val

    p2 = Path(".keys") / "GEMINI"
    if p2.exists():
        val = p2.read_text(encoding="utf-8").strip()
        if val:
            return val

    raise RuntimeError(
        "Missing Gemini API key. Set TOKEN_GEMINI or create a 'GEMINI' file "
        "in the current folder (or '.keys/GEMINI')."
    )


# ==========================================================
# 1) DSL (schema for the LLM)
# ==========================================================

IndexName = Literal[
    # base
    "NDVI", "NDWI", "NDSI", "EVI", "RGB",
    # vegetation / soil-adjusted
    "SAVI", "OSAVI", "MSAVI", "MSAVI2", "GNDVI", "NDRE", "ARVI",
    # water
    "MNDWI", "AWEI_SH", "AWEI_NSH",
    # fire
    "NBR", "NBR2", "BAI",
    # urban / soil
    "NDBI", "BSI",
    # SAR
    "VV_VH_RATIO", "RVI",
]

AggregationName = Literal[
    "mean",
    "median",
    "max",
    "min",
    "monthly_pct_area",   # percent area in AOI con index > threshold per periodo (fraction [0..1])
    "trend_yearly",       # linear trend su medie annuali
]

OutputPacking = Literal[
    "multi_asset",  # un save_result per indice/band-math (multi-asset)
    "multi_band",   # stacking di più indici raster come bande in un singolo asset multi-banda
]

# FIX 8: Global temporal aggregation schema (aggregate_temporal_period on final outputs)
TemporalAggPeriod = Literal["dekad", "month", "year"]
TemporalAggReducer = Literal["mean", "median", "max", "min"]


class TemporalAggregation(BaseModel):
    period: TemporalAggPeriod = Field(..., description="Temporal period for aggregate_temporal_period (dekad|month|year).")
    reducer: TemporalAggReducer = Field("mean", description="Reducer to apply within each temporal period (mean|median|max|min).")


class IndexTask(BaseModel):
    """
    Task per indici "predefiniti" (NDVI, NBR, ecc.).
    name: uno dei nomi in IndexName
    """
    name: IndexName = Field(..., description="Index to compute or product (e.g. RGB)")
    threshold: Optional[float] = Field(None, description="Boolean threshold (e.g. NDSI>0.4)")
    agg: Optional[AggregationName] = Field(None, description="Requested aggregation/analysis")
    period: Optional[str] = Field(None, description="Period for aggregate_temporal_period (e.g. 'month')" )

    # blocco opzionale per operazioni avanzate (definite in eo_expert_ops.ExpertOps)
    expert: Optional[ExpertOps] = Field(
        default=None,
        description=(
            "Optional advanced per-index pipeline: temporal mosaicking, "
            "spatial filters, morphology on masks, temporal smoothing/outlier removal, UDFs."
        ),
    )


class BandMathTask(BaseModel):
    """
    Task per band-math generico definito dall'utente/LLM.
    Esempi di expression:
        "(NIR - SWIR1) / (NIR + SWIR1)"
        "(RED + GREEN + BLUE) / 3"
    """
    name: str = Field(..., description="Logical name of the derived index/band (e.g. 'NDMI', 'custom_index_1').")
    expression: str = Field(
        ...,
        description=(
            "Band-math expression using logical roles (e.g. 'RED', 'NIR', 'SWIR1', 'VV', 'VH'). "
            "Only arithmetic operators +, -, *, /, ** and parentheses are allowed."
        ),
    )
    threshold: Optional[float] = Field(
        None,
        description="Optional threshold (e.g. for monthly_pct_area)."
    )
    agg: Optional[AggregationName] = Field(
        None,
        description="Optional aggregation/analysis (same semantics as IndexTask.agg)."
    )
    period: Optional[str] = Field(
        None,
        description="Period for aggregate_temporal(_period) (e.g. 'month', 'year')."
    )

    expert: Optional[ExpertOps] = Field(
        default=None,
        description="Optional advanced pipeline as in IndexTask.expert (see eo_expert_ops.ExpertOps).",
    )


class TimeRange(BaseModel):
    start: str
    end: str


class CloudMask(BaseModel):
    max: Optional[float] = Field(
        None,
        description=(
            "max cloud cover threshold [0..1]; "
            "if set, cloud_cover_property <= max*100 will be applied at load_collection properties."
        ),
    )
    mask: Optional[str] = Field(None, description="qualitative expression (e.g. 'basic_s2_scl')" )
    property: Optional[str] = Field(
        None,
        description=(
            "Optional explicit metadata property for cloud cover (e.g. 'eo:cloud_cover'). "
            "If set, it overrides KG.cloud_cover_property for the BASE collection."
        ),
    )


class DSLRequest(BaseModel):
    """
    DSL principale:
      - collections: esattamente UNA collection per process graph
      - temporal: uno o più intervalli
      - indices: lista di IndexTask (indici predefiniti) -> opzionale
      - band_math: lista opzionale di BandMathTask (indici custom)
      - temporal_aggregation: (FIX 8) opzionale aggregazione globale aggregate_temporal_period sul/i risultati finali
      - output_packing: 'multi_asset' (default) oppure 'multi_band'
    """
    collections: List[str] = Field(..., description="STAC/back-end collection IDs")
    temporal: List[TimeRange]

    prefer_precomputed_indices: bool = Field(
        default=False,
        description=(
            "If true, and if KG.indices_catalog provides a precomputed source for an index "
            "(e.g. NDVI), load it from the preferred collection/band instead of computing it "
            "from the base collection."
        ),
    )

    output_formats: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional per-task output format override. Keys are task names (e.g. 'NDVI', 'RGB', "
            "or custom band_math names), values are openEO formats like 'GTiff', 'NetCDF', 'PNG', 'JSON'. "
            "If omitted, formats are inferred."
        ),
    )

    # FIX 2: indices non più obbligatorio
    indices: List[IndexTask] = Field(
        default_factory=list,
        description="Optional list of predefined indices to compute."
    )

    band_math: Optional[List[BandMathTask]] = Field(
        default=None,
        description="Optional list of band-math tasks defined by algebraic expressions."
    )

    # FIX 8: global aggregate_temporal_period on final output cubes (per-output apply)
    temporal_aggregation: Optional[TemporalAggregation] = Field(
        default=None,
        description="Optional global aggregate_temporal_period applied to final output cube(s).",
    )

    cloud: Optional[CloudMask] = None

    resolution: Optional[float] = Field(
        None,
        description="Target spatial resolution in meters. None = keep native backend resolution.",
    )
    crs: Optional[str] = Field(
        None,
        description="Target CRS (e.g. 'EPSG:32632'). None = keep native backend CRS.",
    )
    output_format: Optional[str] = Field(
        None,
        description=(
            "Optional explicit default output format (e.g. 'GTiff', 'PNG', 'JSON', 'NetCDF'). "
            "If omitted, a sensible default will be chosen for each index/band-math task."
        ),
    )
    output_packing: OutputPacking = Field(
        "multi_asset",
        description=(
            "How to pack outputs: "
            "'multi_asset' (one save_result per index/band-math, job multi-asset) or "
            "'multi_band' (stack raster indices as bands in a single asset, when possible)."
        ),
    )

    @field_validator("collections")
    @classmethod
    def _check_collections(cls, v: List[str]):
        if not v:
            raise ValueError("At least one collection ID is required.")
        if len(v) != 1:
            raise ValueError(
                f"Exactly one collection is supported per process graph, got {len(v)}: {v}. "
                "Split the request into separate graphs (one per collection)."
            )
        return v

    @field_validator("temporal")
    @classmethod
    def _check_dates(cls, v: List[TimeRange]):
        if not v:
            raise ValueError("At least one temporal range is required.")
        for t in v:
            if not t.start or not t.end:
                raise ValueError("Temporal ranges must have both 'start' and 'end' dates.")
            if len(t.start) < 10 or len(t.end) < 10:
                raise ValueError("Dates must be in 'YYYY-MM-DD' format.")
        return v


# ==========================================================
# 2) FIX 8 guard: apply global temporal aggregation ONLY if time dimension exists
# ==========================================================

def _has_time_dimension(cube: DataCube) -> bool:
    try:
        _ = get_time_dimension_name(cube)
        return True
    except Exception:
        return False


def _apply_global_temporal_aggregation(
    cube: DataCube,
    ta: Optional[TemporalAggregation],
) -> DataCube:
    """
    FIX 8 (correct): apply aggregate_temporal_period ONLY if the cube still has a time dimension.
    Prevents nonsense graphs like: reduce_dimension(t) -> aggregate_temporal_period(month).
    """
    if ta is None:
        return cube

    if not _has_time_dimension(cube):
        return cube

    reducer_name = ta.reducer

    def reducer(data, context=None, _r=reducer_name):
        return getattr(data, _r)()

    # Fail loud: temporal_aggregation was explicit user intent.
    return cube.aggregate_temporal_period(period=ta.period, reducer=reducer)


# ==========================================================
# 3) Output format extraction + smart per-task defaults
# ==========================================================

_FMT_ALIASES = {
    "geotiff": "GTiff",
    "geo tiff": "GTiff",
    "gtiff": "GTiff",
    "tiff": "GTiff",
    "tif": "GTiff",
    "netcdf": "NetCDF",
    "nc": "NetCDF",
    "png": "PNG",
    "json": "JSON",
}

def _canonicalize_format_token(tok: str) -> Optional[str]:
    if not tok:
        return None
    t = tok.strip().lower()
    t = t.replace(".", "").replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    return _FMT_ALIASES.get(t)

def _extract_default_output_format_from_text(text: str) -> Optional[str]:
    """
    Detect global format requests like:
      - "salva in NetCDF"
      - "output geotiff"
      - "export as json"
    """
    s = (text or "").lower()
    m = re.search(r"\b(salva(?:re)?|output|export|formato|format)\b.*\b(netcdf|nc|geotiff|gtiff|tiff|tif|png|json)\b", s)
    if m:
        return _canonicalize_format_token(m.group(2))
    m2 = re.search(r"\b(netcdf|geotiff|gtiff|tiff|tif|png|json)\b", s)
    if m2 and ("salv" in s or "output" in s or "export" in s or "formato" in s or "format" in s):
        return _canonicalize_format_token(m2.group(1))
    return None

def _extract_per_task_output_formats_from_text(text: str, task_names: List[str]) -> Dict[str, str]:
    """
    Detect per-task formats like:
      - "NDVI in NetCDF, RGB in PNG"
      - "salva NDVI geotiff"
    """
    s = (text or "")
    out: Dict[str, str] = {}

    if not task_names:
        return out

    escaped = sorted({re.escape(n) for n in task_names if isinstance(n, str) and n.strip()}, key=len, reverse=True)
    if not escaped:
        return out

    tasks_group = r"(" + "|".join(escaped) + r")"

    for m in re.finditer(tasks_group + r".{0,40}\b(netcdf|nc|geotiff|gtiff|tiff|tif|png|json)\b",
                         s, flags=re.IGNORECASE | re.DOTALL):
        task = m.group(1)
        fmt = _canonicalize_format_token(m.group(2))
        if fmt:
            out[task.upper() if task.upper() in {t.upper() for t in task_names} else task] = fmt

    name_map = {t.upper(): t for t in task_names}
    normalized: Dict[str, str] = {}
    for k, v in out.items():
        kk = name_map.get(str(k).upper(), str(k))
        normalized[kk] = v
    return normalized

def _choose_sensible_default_format_for_single_task(task_name: str, agg: Optional[AggregationName], is_rgb: bool) -> str:
    if is_rgb:
        return "PNG"
    if agg in {"monthly_pct_area", "trend_yearly"}:
        return "JSON"
    return "GTiff"

def _derive_output_formats(dsl: DSLRequest, instruction: str) -> Dict[str, str]:
    """
    - If user mentions formats in text -> use them (global and/or per task)
    - Otherwise -> per-task defaults
    """
    task_names: List[str] = []
    idx_meta: Dict[str, Tuple[Optional[AggregationName], bool]] = {}

    for t in dsl.indices:
        task_names.append(t.name)
        idx_meta[t.name] = (t.agg, t.name == "RGB")

    if dsl.band_math:
        for bm in dsl.band_math:
            task_names.append(bm.name)
            idx_meta[bm.name] = (bm.agg, False)

    per_task = _extract_per_task_output_formats_from_text(instruction, task_names)
    default_fmt = _extract_default_output_format_from_text(instruction)

    out: Dict[str, str] = {}
    for name in task_names:
        if name in per_task:
            out[name] = per_task[name]
            continue
        if default_fmt:
            out[name] = default_fmt
            continue
        agg, is_rgb = idx_meta.get(name, (None, False))
        out[name] = _choose_sensible_default_format_for_single_task(name, agg, is_rgb)

    return out


# ==========================================================
# 4) Cloud property/max extraction from user text + precomputed indices hint
# ==========================================================

def _extract_cloud_property_from_text(text: str) -> Optional[str]:
    """
    Try to infer an explicit cloud-cover metadata property from the instruction.
    Handles:
      - explicit 'eo:cloud_cover'
      - English pattern: properties filter on "X"
      - Italian-ish patterns: filtro (proprietà) su "X"
    """
    s = text or ""

    # Most common / standard case
    if re.search(r"\beo:cloud_cover\b", s, flags=re.IGNORECASE):
        return "eo:cloud_cover"

    # English: properties filter on "X"
    m = re.search(r'properties\s+filter\s+on\s+"([^"]+)"', s, flags=re.IGNORECASE)
    if m:
        prop = (m.group(1) or "").strip()
        return prop or None

    # Italian: filtro (di) proprietà su "X"
    m = re.search(
        r'filtro\s+(?:di\s+)?propriet[àa]\s+(?:su|sulla)\s+["\']([^"\']+)["\']',
        s,
        flags=re.IGNORECASE,
    )
    if m:
        prop = (m.group(1) or "").strip()
        return prop or None

    # Generic: filter on "X" (last resort, conservative)
    m = re.search(r'\bfilter\s+on\s+"([^"]+)"', s, flags=re.IGNORECASE)
    if m:
        prop = (m.group(1) or "").strip()
        # avoid catching band names etc. Only accept STAC-like keys containing ':'.
        if ":" in prop:
            return prop
    return None


def _extract_cloud_max_from_text(text: str) -> Optional[float]:
    """
    Understand:
      - "cloud cover <= 50%"
      - "nuvole 50%"
      - "copertura nuvolosa 20 percento"
    Returns fraction [0..1].
    """
    s = (text or "").lower()

    m = re.search(
        r"\b(cloud|nuvol|copertura)\w*\b.{0,30}\b(\d{1,3})\s*%|\b(\d{1,3})\s*%\b.{0,30}\b(cloud|nuvol|copertura)\w*\b",
        s
    )
    if m:
        pct = m.group(2) or m.group(3)
        try:
            v = float(pct) / 100.0
            return max(0.0, min(1.0, v))
        except Exception:
            return None

    m2 = re.search(r"\b(cloud|max cloud|nuvole|max nuvole|copertura)\w*\b.{0,30}\b(0\.\d+|1\.0|0)\b", s)
    if m2:
        try:
            v = float(m2.group(2))
            return max(0.0, min(1.0, v))
        except Exception:
            return None

    return None

def _text_requests_precomputed_indices(text: str) -> bool:
    s = (text or "").lower()
    return (
        "precomput" in s
        or "pre-calcol" in s
        or "precalcol" in s
        or "terracope" in s
        or "terrascope" in s
        or "TERRASCOPE_S2_NDVI" in text
        or "NDVI_10M" in text
    )


# ==========================================================
# 5) Knowledge Graph loader (dynamic bands & indices)
# ==========================================================

_KG: Optional[Dict[str, Any]] = None
_KG_COLLECTIONS: Dict[str, Dict[str, Any]] = {}
_INDEX_DEFS: Dict[str, Any] = {}

DEFAULT_KG_PATH = "collections_kg.json"


def _load_kg(path: Optional[str] = None) -> None:
    """
    Carica il knowledge graph da JSON (una sola volta per processo).
    Usa:
      - env OPENEO_COLLECTIONS_KG_PATH
      - oppure percorso passato
      - altrimenti 'collections_kg.json' nella cwd
    """
    global _KG, _KG_COLLECTIONS, _INDEX_DEFS
    if _KG is not None:
        return

    cfg_path = (
        path
        or os.getenv("OPENEO_COLLECTIONS_KG_PATH")
        or DEFAULT_KG_PATH
    )
    kg_path = Path(cfg_path)
    if not kg_path.exists():
        raise RuntimeError(
            f"Knowledge graph JSON '{kg_path}' not found. "
            "Generate it first with collections_kg.py."
        )

    with kg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid KG structure in '{kg_path}': expected JSON object.")

    collections = data.get("collections") or []
    if not isinstance(collections, list):
        raise RuntimeError("Invalid KG: 'collections' must be a list.")

    _KG = data
    _KG_COLLECTIONS = {
        str(c.get("collection_id")): c for c in collections if c.get("collection_id")
    }

    catalog = data.get("indices_catalog") or {}
    if isinstance(catalog, dict):
        _INDEX_DEFS = catalog
    else:
        _INDEX_DEFS = {}


def _get_collection_profile(collection_id: str) -> Dict[str, Any]:
    _load_kg()
    prof = _KG_COLLECTIONS.get(collection_id)
    if not prof:
        raise ValueError(
            f"Collection '{collection_id}' is not present in the knowledge graph. "
            f"Run collections_kg.py again or check the collection ID."
        )
    return prof


def _get_index_defs() -> Dict[str, Any]:
    _load_kg()
    return _INDEX_DEFS


# ==========================================================
# 5A) Precomputed index lookup via KG.indices_catalog
# ==========================================================

def _get_precomputed_source_for_index(index_name: str) -> Optional[Dict[str, str]]:
    """
    Expected KG.indices_catalog entry examples:
      "NDVI": { "preferred_collection": "TERRASCOPE_S2_NDVI_V2", "preferred_band": "NDVI_10M" }
    """
    defs = _get_index_defs()
    if not isinstance(defs, dict):
        return None
    src = defs.get(index_name)
    if not isinstance(src, dict):
        return None

    coll = src.get("preferred_collection") or src.get("collection_id")
    band = src.get("preferred_band") or src.get("band_id")

    if isinstance(coll, str) and coll.strip() and isinstance(band, str) and band.strip():
        return {"collection_id": coll.strip(), "band": band.strip()}
    return None


# ==========================================================
# 5B) KG helpers (single-band-only collections)
# ==========================================================

def _collection_single_band_only(collection_id: str) -> bool:
    """
    FIX 4: alcuni dataset supportano load_collection con UNA sola banda alla volta.
    Rilevamento non invasivo dalla description nel KG.
    """
    prof = _get_collection_profile(collection_id)
    desc = (prof.get("description") or "")
    d = desc.lower()
    return ("only supports loading one band at a time" in d) or ("loading one band at a time" in d)


# ==========================================================
# 6) Dynamic band & index lookup via KG
# ==========================================================

def _get_band_mapping(collection_id: str) -> Dict[str, str]:
    prof = _get_collection_profile(collection_id)
    logical_roles = dict(prof.get("logical_roles") or {})

    # FIX 3: se logical_roles è vuoto, fallback band_id->band_id
    if not logical_roles:
        for b in (prof.get("bands") or []):
            if not isinstance(b, dict):
                continue
            band_id = b.get("band_id")
            if isinstance(band_id, str) and band_id.strip():
                role = band_id.strip()
                logical_roles[role] = role
                up = role.upper()
                if up != role and up not in logical_roles:
                    logical_roles[up] = role

    if "SWIR" not in logical_roles:
        if "SWIR1" in logical_roles:
            logical_roles["SWIR"] = logical_roles["SWIR1"]
        elif "SWIR2" in logical_roles:
            logical_roles["SWIR"] = logical_roles["SWIR2"]

    sensor_type = prof.get("sensor_type")
    if sensor_type == "sar":
        vv = logical_roles.get("VV")
        vh = logical_roles.get("VH")
        if vv and "RED" not in logical_roles:
            logical_roles["RED"] = vv
        if vh and "GREEN" not in logical_roles:
            logical_roles["GREEN"] = vh
        if vv and "BLUE" not in logical_roles:
            logical_roles["BLUE"] = vv

    return logical_roles


def _get_cloud_cover_property(collection_id: str) -> Optional[str]:
    prof = _get_collection_profile(collection_id)
    prop = prof.get("cloud_cover_property")
    if prop is None:
        return None
    prop = str(prop).strip()
    return prop or None


def _get_classification_band_mapping(collection_id: str) -> Dict[str, str]:
    prof = _get_collection_profile(collection_id)
    class_roles = dict(prof.get("classification_roles") or {})
    logical_roles = dict(prof.get("logical_roles") or {})
    if "SCL" in logical_roles and "SCL" not in class_roles:
        class_roles["SCL"] = logical_roles["SCL"]
    return class_roles


def _get_allowed_indices(collection_id: str) -> List[IndexName]:
    prof = _get_collection_profile(collection_id)
    supported = prof.get("supported_indices") or []
    if not isinstance(supported, list):
        supported = []

    allowed_all = {
        "NDVI", "NDWI", "NDSI", "EVI", "RGB",
        "SAVI", "OSAVI", "MSAVI", "MSAVI2", "GNDVI", "NDRE", "ARVI",
        "MNDWI", "AWEI_SH", "AWEI_NSH",
        "NBR", "NBR2", "BAI",
        "NDBI", "BSI",
        "VV_VH_RATIO", "RVI",
    }
    allowed = [idx for idx in supported if idx in allowed_all]

    if not allowed:
        raise ValueError(
            f"Collection '{collection_id}' has no indices supported by the current DSL "
            f"according to the knowledge graph."
        )
    return allowed  # type: ignore[return-value]


def _is_index_supported(collection_id: str, index: IndexName) -> bool:
    try:
        allowed = _get_allowed_indices(collection_id)
    except Exception as e:
        logger.error("nl_to_openeo: cannot get allowed indices for '%s': %s", collection_id, e)
        return False
    return index in allowed


# ==========================================================
# 7) Index / band helpers (usano SOLO il KG)
# ==========================================================

def _index_bands_for_task(task: IndexTask, dsl: DSLRequest) -> List[str]:
    coll = dsl.collections[0]
    mapping = _get_band_mapping(coll)
    bands: List[str] = []

    if task.name == "NDVI":
        bands = [mapping.get("RED"), mapping.get("NIR")]
    elif task.name == "EVI":
        bands = [mapping.get("BLUE"), mapping.get("RED"), mapping.get("NIR")]
    elif task.name == "SAVI":
        bands = [mapping.get("RED"), mapping.get("NIR")]
    elif task.name in ("OSAVI", "MSAVI", "MSAVI2"):
        bands = [mapping.get("RED"), mapping.get("NIR")]
    elif task.name == "GNDVI":
        bands = [mapping.get("GREEN"), mapping.get("NIR")]
    elif task.name == "NDRE":
        bands = [mapping.get("RED_EDGE"), mapping.get("NIR")]
    elif task.name == "ARVI":
        bands = [mapping.get("BLUE"), mapping.get("RED"), mapping.get("NIR")]

    elif task.name == "NDWI":
        bands = [mapping.get("GREEN"), mapping.get("NIR")]
    elif task.name == "MNDWI":
        bands = [mapping.get("GREEN"), mapping.get("SWIR1")]
    elif task.name == "AWEI_SH":
        bands = [mapping.get("GREEN"), mapping.get("NIR"), mapping.get("SWIR1"), mapping.get("SWIR2")]
    elif task.name == "AWEI_NSH":
        bands = [mapping.get("BLUE"), mapping.get("GREEN"), mapping.get("NIR"), mapping.get("SWIR1"), mapping.get("SWIR2")]

    elif task.name == "NDSI":
        bands = [mapping.get("GREEN"), mapping.get("SWIR")]

    elif task.name == "NBR":
        bands = [mapping.get("NIR"), mapping.get("SWIR2")]
    elif task.name == "NBR2":
        bands = [mapping.get("SWIR1"), mapping.get("SWIR2")]
    elif task.name == "BAI":
        bands = [mapping.get("RED"), mapping.get("NIR")]

    elif task.name == "NDBI":
        bands = [mapping.get("SWIR1"), mapping.get("NIR")]
    elif task.name == "BSI":
        bands = [mapping.get("SWIR1"), mapping.get("RED"), mapping.get("NIR"), mapping.get("BLUE")]

    elif task.name in ("VV_VH_RATIO", "RVI"):
        bands = [mapping.get("VV"), mapping.get("VH")]

    elif task.name == "RGB":
        bands = [mapping.get("RED"), mapping.get("GREEN"), mapping.get("BLUE")]

    return [b for b in bands if b]


def _needed_bands_for_index_task(task: IndexTask, dsl: DSLRequest) -> List[str]:
    index_bands = _index_bands_for_task(task, dsl)
    bands: List[str] = list(index_bands)

    seen = set()
    uniq: List[str] = []
    for b in bands:
        if b and b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


# ----------------------------- Band-math helpers -----------------------------

class _SafeBandMathError(ValueError):
    pass


def _parse_band_math_expression(expr: str, allowed_vars: Set[str]) -> Tuple[ast.AST, Set[str]]:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as e:
        raise _SafeBandMathError(f"Invalid band-math expression '{expr}': {e}") from e

    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Load,
    )

    used_vars: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
            raise _SafeBandMathError(
                "Only pure arithmetic band-math expressions are allowed (no functions, attributes, or indexing)."
            )
        if not isinstance(node, allowed_node_types):
            raise _SafeBandMathError(
                f"Unsupported syntax in band-math expression: {node.__class__.__name__}"
            )
        if isinstance(node, ast.Name):
            var = node.id
            if var not in allowed_vars:
                raise _SafeBandMathError(
                    f"Unknown variable '{var}' in band-math expression. "
                    f"Allowed variables for this collection: {sorted(allowed_vars)}"
                )
            used_vars.add(var)

    if not used_vars:
        raise _SafeBandMathError(
            f"Band-math expression '{expr}' does not reference any band roles."
        )

    return tree, used_vars


def _eval_band_math_ast(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_band_math_ast(node.body, env)

    if isinstance(node, ast.Constant):
        val = node.value
        if not isinstance(val, (int, float)):
            raise _SafeBandMathError("Only numeric constants are allowed in band-math expressions.")
        return val

    if isinstance(node, ast.Name):
        try:
            return env[node.id]
        except KeyError:
            raise _SafeBandMathError(f"Variable '{node.id}' not found in environment.") from None

    if isinstance(node, ast.UnaryOp):
        operand = _eval_band_math_ast(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise _SafeBandMathError(f"Unsupported unary operator: {node.op}")

    if isinstance(node, ast.BinOp):
        left = _eval_band_math_ast(node.left, env)
        right = _eval_band_math_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left ** right
        else:
            raise _SafeBandMathError(f"Unsupported binary operator: {node.op}")

    raise _SafeBandMathError(
        f"Unsupported AST node in band-math expression: {node.__class__.__name__}"
    )


def _needed_bands_for_bandmath_task(task: BandMathTask, dsl: DSLRequest) -> List[str]:
    coll = dsl.collections[0]
    mapping = _get_band_mapping(coll)
    allowed_vars: Set[str] = set(mapping.keys())

    tree, used_vars = _parse_band_math_expression(task.expression, allowed_vars)

    bands: List[str] = []
    for role in used_vars:
        band_id = mapping.get(role) or mapping.get(role.upper())
        if not band_id:
            raise ValueError(
                f"Collection '{coll}' does not define a band for logical role '{role}' "
                f"required by band-math task '{task.name}'."
            )
        bands.append(band_id)

    seen: Set[str] = set()
    uniq: List[str] = []
    for b in bands:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def _all_index_bands(dsl: DSLRequest) -> List[str]:
    bands: List[str] = []
    for idx in dsl.indices:
        bands.extend(_needed_bands_for_index_task(idx, dsl))

    if dsl.band_math:
        for bm in dsl.band_math:
            bands.extend(_needed_bands_for_bandmath_task(bm, dsl))

    seen: Set[str] = set()
    uniq: List[str] = []
    for b in bands:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def _needed_bands_for_all_tasks(dsl: DSLRequest) -> List[str]:
    return _all_index_bands(dsl)


def _validate_indices_and_bands(dsl: DSLRequest) -> None:
    """
    FIX 2 + FIX 5:
      - valido se ci sono indices OR band_math
      - NON richiede supported_indices quando indices è vuoto e si usa solo band_math
    """
    if not dsl.collections:
        raise ValueError("DSL without 'collections' is not supported: at least one collection ID is required.")

    if len(dsl.collections) != 1:
        raise ValueError(
            f"Exactly one collection is supported per process graph, got {len(dsl.collections)}: {dsl.collections}. "
            "Split the request into separate graphs (one per collection)."
        )

    if (not dsl.indices) and (not dsl.band_math):
        raise ValueError(
            "DSL without tasks is not supported: provide at least one IndexTask in 'indices' or one BandMathTask in 'band_math'."
        )

    collection_id = dsl.collections[0]
    _ = _get_collection_profile(collection_id)

    required_band_count_map: Dict[IndexName, int] = {
        "NDVI": 2,
        "NDWI": 2,
        "NDSI": 2,
        "EVI": 3,
        "RGB": 3,
        "SAVI": 2,
        "OSAVI": 2,
        "MSAVI": 2,
        "MSAVI2": 2,
        "GNDVI": 2,
        "NDRE": 2,
        "ARVI": 3,
        "MNDWI": 2,
        "AWEI_SH": 4,
        "AWEI_NSH": 5,
        "NBR": 2,
        "NBR2": 2,
        "BAI": 2,
        "NDBI": 2,
        "BSI": 4,
        "VV_VH_RATIO": 2,
        "RVI": 2,
    }

    if dsl.indices:
        allowed_for_coll = _get_allowed_indices(collection_id)

        for task in dsl.indices:
            if task.name not in allowed_for_coll:
                raise ValueError(
                    f"Index '{task.name}' is not supported for collection '{collection_id}' according to the knowledge graph. "
                    f"Allowed indices for this collection: {allowed_for_coll}."
                )
            index_bands = _index_bands_for_task(task, dsl)
            expected = required_band_count_map.get(task.name, 0)
            if expected and len(index_bands) < expected:
                raise ValueError(
                    f"Collection '{collection_id}' does not define all bands required for index '{task.name}' "
                    f"according to the knowledge graph. Expected at least {expected} bands but got {len(index_bands)}."
                )

    if dsl.band_math:
        for bm in dsl.band_math:
            _ = _needed_bands_for_bandmath_task(bm, dsl)


# ==========================================================
# 8) AOI parsing utilities
# ==========================================================

def _strip_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _extract_first_json_block_loose(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON block '{...}' found.")
    chunk = text[start : end + 1]
    chunk = _strip_trailing_commas(chunk)
    return json.loads(chunk)


def parse_geojson_flexible(obj_or_str):
    if isinstance(obj_or_str, (dict, list)):
        return obj_or_str

    s = (obj_or_str or "").strip()
    if not s:
        raise ValueError("Empty AOI string")

    try:
        return json.loads(_strip_trailing_commas(s))
    except Exception:
        pass

    try:
        return _extract_first_json_block_loose(s)
    except Exception:
        pass

    try:
        pyobj = ast.literal_eval(s)
        return json.loads(json.dumps(pyobj))
    except Exception:
        pass

    try:
        arr = json.loads(re.sub(r"[^\[\],\d\.\-\s]", "", s))
        if isinstance(arr, list) and len(arr) == 4 and all(isinstance(x, (int, float)) for x in arr):
            w, s_, e, n = arr
            coords = [[[w, s_], [e, s_], [e, n], [w, n], [w, s_]]]
            return {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {"crs": "EPSG:4326"},
            }
    except Exception:
        pass

    raise ValueError("Cannot interpret AOI: not JSON, not Python dict, not a BBox.")


# ==========================================================
# 9) AOI → spatial_extent, temporal_extent
# ==========================================================

def _compute_bbox_from_geometry(geom: Dict[str, Any]) -> Dict[str, float]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    allowed_types = {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }

    if gtype not in allowed_types or coords is None:
        raise ValueError(f"Unsupported geometry type for bbox: {gtype!r}")

    xy = _collect_xy(coords)
    if not xy:
        raise ValueError("Empty geometry for bbox computation.")

    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]

    return {
        "west": min(xs),
        "east": max(xs),
        "south": min(ys),
        "north": max(ys),
    }


def _collect_xy(c):
    if isinstance(c, (float, int)):
        raise ValueError("Unexpected scalar in coordinates")
    if isinstance(c, list):
        if (
            len(c) >= 2
            and all(isinstance(v, (int, float)) for v in c[:2])
            and all(not isinstance(el, list) for el in c)
        ):
            return [(c[0], c[1])]
        else:
            res = []
            for el in c:
                res.extend(_collect_xy(el))
            return res
    raise ValueError("Invalid coordinate structure")


def _extract_crs_from_aoi(aoi_geojson: Dict[str, Any], default: str = "EPSG:4326") -> str:
    def _normalize_crs_value(crs_obj: Any) -> Optional[str]:
        if isinstance(crs_obj, str) and crs_obj.strip():
            return crs_obj.strip()

        if isinstance(crs_obj, dict):
            props = crs_obj.get("properties") or {}
            name = props.get("name") or crs_obj.get("name")
            if isinstance(name, str) and name.strip():
                name = name.strip()
                m = re.search(r"(EPSG)[:/]{1,2}(\d+)", name, re.IGNORECASE)
                if m:
                    return f"EPSG:{m.group(2)}"
                return name
        return None

    if not isinstance(aoi_geojson, dict):
        return default

    crs = _normalize_crs_value(aoi_geojson.get("crs"))
    if crs:
        return crs

    props = aoi_geojson.get("properties") if isinstance(aoi_geojson.get("properties"), dict) else {}
    crs = _normalize_crs_value(props.get("crs"))
    if crs:
        return crs

    if aoi_geojson.get("type") == "Feature":
        f_props = aoi_geojson.get("properties") or {}
        crs = _normalize_crs_value(f_props.get("crs"))
        if crs:
            return crs

    if aoi_geojson.get("type") == "FeatureCollection":
        features = aoi_geojson.get("features") or []
        for f in features:
            if not isinstance(f, dict):
                continue
            crs = _normalize_crs_value(f.get("crs"))
            if crs:
                return crs
            f_props = f.get("properties") or {}
            crs = _normalize_crs_value(f_props.get("crs"))
            if crs:
                return crs

    return default


def _compute_spatial_extent_from_aoi(aoi_geojson: Dict[str, Any]) -> Dict[str, Any]:
    crs = _extract_crs_from_aoi(aoi_geojson, default="EPSG:4326")

    if (
        isinstance(aoi_geojson, dict)
        and aoi_geojson.get("type") == "FeatureCollection"
        and aoi_geojson.get("features")
    ):
        features = aoi_geojson["features"]
        bboxes = []
        for f in features:
            geom = f.get("geometry")
            if not isinstance(geom, dict):
                continue
            bboxes.append(_compute_bbox_from_geometry(geom))
        if not bboxes:
            raise ValueError("Empty FeatureCollection or invalid geometries.")
        west = min(b["west"] for b in bboxes)
        east = max(b["east"] for b in bboxes)
        south = min(b["south"] for b in bboxes)
        north = max(b["north"] for b in bboxes)
        bbox = {"west": west, "east": east, "south": south, "north": north}
    else:
        if isinstance(aoi_geojson, dict) and "geometry" in aoi_geojson:
            geom = aoi_geojson["geometry"]
        else:
            geom = aoi_geojson
        if not isinstance(geom, dict):
            raise ValueError("AOI must be a GeoJSON Feature/FeatureCollection/geometry.")
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if not (
                isinstance(coords, list)
                and len(coords) >= 2
                and all(isinstance(v, (int, float)) for v in coords[:2])
            ):
                raise ValueError("Invalid Point geometry for AOI.")
            lon, lat = coords[0], coords[1]
            eps = 0.0001
            bbox = {
                "west": lon - eps,
                "east": lon + eps,
                "south": lat - eps,
                "north": lat + eps,
            }
        else:
            bbox = _compute_bbox_from_geometry(geom)

    bbox["crs"] = crs
    return bbox


def _compute_temporal_extent(dsl: DSLRequest) -> Union[List[str], List[List[str]]]:
    if not dsl.temporal:
        raise ValueError("DSL without 'temporal' is not supported.")

    ranges: List[Tuple[datetime.date, datetime.date]] = []

    for t in dsl.temporal:
        if not t.start or not t.end:
            raise ValueError("Temporal ranges must have both 'start' and 'end' dates.")
        try:
            s_dt = datetime.fromisoformat(t.start[:10]).date()
            e_dt = datetime.fromisoformat(t.end[:10]).date()
        except Exception as exc:
            raise ValueError(
                f"Invalid temporal range '{t.start}'–'{t.end}'. Dates must be ISO 'YYYY-MM-DD'."
            ) from exc
        if e_dt < s_dt:
            raise ValueError(
                f"Invalid temporal range '{t.start}'–'{t.end}': end date is before start date."
            )
        ranges.append((s_dt, e_dt))

    ranges.sort(key=lambda r: r[0])

    merged: List[List[datetime.date]] = []
    for s_dt, e_dt in ranges:
        if not merged:
            merged.append([s_dt, e_dt])
            continue

        last_s, last_e = merged[-1]
        if s_dt <= last_e:
            if e_dt > last_e:
                merged[-1][1] = e_dt
        else:
            merged.append([s_dt, e_dt])

    if len(merged) == 1:
        s_dt, e_dt = merged[0]
        return [s_dt.isoformat(), e_dt.isoformat()]
    else:
        return [[s_dt.isoformat(), e_dt.isoformat()] for (s_dt, e_dt) in merged]


# ==========================================================
# 9A) AOI shape helper per mask_polygon
# ==========================================================

def _is_aoi_exact_bbox_rectangle(aoi_geojson: Dict[str, Any], spatial_extent: Dict[str, Any]) -> bool:
    if isinstance(aoi_geojson, dict) and aoi_geojson.get("type") == "Feature":
        geom = aoi_geojson.get("geometry")
    elif isinstance(aoi_geojson, dict) and aoi_geojson.get("type") == "FeatureCollection":
        return False
    else:
        geom = aoi_geojson

    if not isinstance(geom, dict):
        return False

    if geom.get("type") != "Polygon":
        return False

    coords = geom.get("coordinates")
    if not (isinstance(coords, list) and coords):
        return False

    ring = coords[0]
    if not (isinstance(ring, list) and len(ring) >= 4):
        return False

    pts = ring[:]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    if len(pts) != 4:
        return False

    west = spatial_extent["west"]
    east = spatial_extent["east"]
    south = spatial_extent["south"]
    north = spatial_extent["north"]

    target = [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
    ]

    def _eq_pt(p1, p2, eps=1e-9):
        return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps

    matched = [False] * 4
    for p in pts:
        if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
            return False
        found = False
        for i, t in enumerate(target):
            if not matched[i] and _eq_pt((p[0], p[1]), t):
                matched[i] = True
                found = True
                break
        if not found:
            return False

    return all(matched)


# ==========================================================
# 10) Aggregations
# ==========================================================

def _apply_aggregation(
    cube: DataCube,
    agg: Optional[AggregationName],
    period: Optional[str],
    threshold: Optional[float],
    aoi_geojson: Dict[str, Any],
) -> DataCube:
    if agg is None:
        return cube

    if agg == "monthly_pct_area":
        if threshold is None:
            raise ValueError(
                "Aggregation 'monthly_pct_area' requires 'threshold' to be set."
            )

        thr = float(threshold)

        def _gt_threshold_px(x, context=None, thr=thr):
            return x > thr

        bool_cube = cube.apply(_gt_threshold_px)

        def temporal_mean_reducer(data, context=None):
            return data.mean()

        monthly_frac = bool_cube.aggregate_temporal_period(
            period=period or "month",
            reducer=temporal_mean_reducer,
        )

        def spatial_mean_reducer(data, context=None):
            return data.mean()

        cube = monthly_frac.aggregate_spatial(
            geometries=aoi_geojson,
            reducer=spatial_mean_reducer,
        )
        return cube

    if agg in {"mean", "median", "max", "min"}:
        def reducer(data, context=None, agg_name=agg):
            return getattr(data, agg_name)()

        if period:
            cube = cube.aggregate_temporal_period(
                period=period,
                reducer=reducer,
            )
        else:
            time_dim = get_time_dimension_name(cube)
            cube = cube.reduce_dimension(dimension=time_dim, reducer=reducer)
        return cube

    if agg == "trend_yearly":
        def mean_reducer(data, context=None):
            return data.mean()

        yearly = cube.aggregate_temporal_period(
            period="year",
            reducer=mean_reducer,
        )

        time_dim = get_time_dimension_name(yearly)
        cube = yearly.linear_trend(dimension=time_dim)
        return cube

    return cube


# ==========================================================
# 11) Index computations (predefined + bandmath)
# ==========================================================

def _compute_predefined_index_cube(
    base_cube: DataCube,
    task: IndexTask,
    mapping: Dict[str, str],
    aoi_geojson: Dict[str, Any],
) -> DataCube:
    cube = base_cube

    if task.name == "RGB":
        index_bands = [
            mapping.get("RED"),
            mapping.get("GREEN"),
            mapping.get("BLUE"),
        ]
        index_bands = [b for b in index_bands if b]
        if index_bands:
            cube = cube.filter_bands(bands=index_bands)

        cube = apply_expert_ops(cube, task.expert)
        return cube

    if task.name == "NDVI":
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _ndvi_reduce(data, context=None):
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            num = nir - red
            den = nir + red
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_ndvi_reduce)

    elif task.name == "EVI":
        blue_label = mapping["BLUE"]
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _evi_reduce(data, context=None):
            blue = data.array_element(label=blue_label)
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            num = nir - red
            den = nir + (red * 6) - (blue * 7.5) + 1
            return (num / den) * 2.5

        cube = cube.reduce_dimension(dimension="bands", reducer=_evi_reduce)

    elif task.name == "SAVI":
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]
        L = 0.5

        def _savi_reduce(data, context=None, L=L):
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            num = (nir - red) * (1 + L)
            den = nir + red + L
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_savi_reduce)

    elif task.name == "OSAVI":
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _osavi_reduce(data, context=None):
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            num = (nir - red) * 1.16
            den = nir + red + 0.16
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_osavi_reduce)

    elif task.name in ("MSAVI", "MSAVI2"):
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _msavi_reduce(data, context=None):
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            term = (2 * nir + 1) ** 2 - 8 * (nir - red)
            sqrt_term = term.sqrt()
            return (2 * nir + 1 - sqrt_term) / 2

        cube = cube.reduce_dimension(dimension="bands", reducer=_msavi_reduce)

    elif task.name == "GNDVI":
        green_label = mapping["GREEN"]
        nir_label = mapping["NIR"]

        def _gndvi_reduce(data, context=None):
            g = data.array_element(label=green_label)
            nir = data.array_element(label=nir_label)
            num = nir - g
            den = nir + g
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_gndvi_reduce)

    elif task.name == "NDRE":
        re_label = mapping["RED_EDGE"]
        nir_label = mapping["NIR"]

        def _ndre_reduce(data, context=None):
            re = data.array_element(label=re_label)
            nir = data.array_element(label=nir_label)
            num = nir - re
            den = nir + re
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_ndre_reduce)

    elif task.name == "ARVI":
        blue_label = mapping["BLUE"]
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _arvi_reduce(data, context=None):
            blue = data.array_element(label=blue_label)
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            rb = 2 * red - blue
            num = nir - rb
            den = nir + rb
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_arvi_reduce)

    elif task.name == "NDWI":
        g_label = mapping["GREEN"]
        nir_label = mapping["NIR"]

        def _ndwi_reduce(data, context=None):
            g = data.array_element(label=g_label)
            nir = data.array_element(label=nir_label)
            num = g - nir
            den = g + nir
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_ndwi_reduce)

    elif task.name == "MNDWI":
        g_label = mapping["GREEN"]
        swir1_label = mapping["SWIR1"]

        def _mndwi_reduce(data, context=None):
            g = data.array_element(label=g_label)
            sw1 = data.array_element(label=swir1_label)
            num = g - sw1
            den = g + sw1
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_mndwi_reduce)

    elif task.name == "AWEI_SH":
        g_label = mapping["GREEN"]
        nir_label = mapping["NIR"]
        swir1_label = mapping["SWIR1"]
        swir2_label = mapping["SWIR2"]

        def _awei_sh_reduce(data, context=None):
            g = data.array_element(label=g_label)
            nir = data.array_element(label=nir_label)
            sw1 = data.array_element(label=swir1_label)
            sw2 = data.array_element(label=swir2_label)
            return 4 * (g - sw1) - (0.25 * nir + 2.75 * sw2)

        cube = cube.reduce_dimension(dimension="bands", reducer=_awei_sh_reduce)

    elif task.name == "AWEI_NSH":
        b_label = mapping["BLUE"]
        g_label = mapping["GREEN"]
        nir_label = mapping["NIR"]
        swir1_label = mapping["SWIR1"]
        swir2_label = mapping["SWIR2"]

        def _awei_nsh_reduce(data, context=None):
            b = data.array_element(label=b_label)
            g = data.array_element(label=g_label)
            nir = data.array_element(label=nir_label)
            sw1 = data.array_element(label=swir1_label)
            sw2 = data.array_element(label=swir2_label)
            return b + 2.5 * g - 1.5 * (nir + sw1) - 0.25 * sw2

        cube = cube.reduce_dimension(dimension="bands", reducer=_awei_nsh_reduce)

    elif task.name == "NDSI":
        g_label = mapping["GREEN"]
        swir_label = mapping["SWIR"]

        def _ndsi_reduce(data, context=None):
            g = data.array_element(label=g_label)
            sw = data.array_element(label=swir_label)
            num = g - sw
            den = g + sw
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_ndsi_reduce)

    elif task.name == "NBR":
        nir_label = mapping["NIR"]
        swir2_label = mapping["SWIR2"]

        def _nbr_reduce(data, context=None):
            nir = data.array_element(label=nir_label)
            sw2 = data.array_element(label=swir2_label)
            num = nir - sw2
            den = nir + sw2
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_nbr_reduce)

    elif task.name == "NBR2":
        swir1_label = mapping["SWIR1"]
        swir2_label = mapping["SWIR2"]

        def _nbr2_reduce(data, context=None):
            sw1 = data.array_element(label=swir1_label)
            sw2 = data.array_element(label=swir2_label)
            num = sw1 - sw2
            den = sw1 + sw2
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_nbr2_reduce)

    elif task.name == "BAI":
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]

        def _bai_reduce(data, context=None):
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            return 1 / ((red - 0.1) ** 2 + (nir - 0.06) ** 2)

        cube = cube.reduce_dimension(dimension="bands", reducer=_bai_reduce)

    elif task.name == "NDBI":
        swir1_label = mapping["SWIR1"]
        nir_label = mapping["NIR"]

        def _ndbi_reduce(data, context=None):
            sw1 = data.array_element(label=swir1_label)
            nir = data.array_element(label=nir_label)
            num = sw1 - nir
            den = sw1 + nir
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_ndbi_reduce)

    elif task.name == "BSI":
        swir1_label = mapping["SWIR1"]
        red_label = mapping["RED"]
        nir_label = mapping["NIR"]
        blue_label = mapping["BLUE"]

        def _bsi_reduce(data, context=None):
            sw1 = data.array_element(label=swir1_label)
            red = data.array_element(label=red_label)
            nir = data.array_element(label=nir_label)
            blue = data.array_element(label=blue_label)
            num = (sw1 + red) - (nir + blue)
            den = (sw1 + red) + (nir + blue)
            return num / den

        cube = cube.reduce_dimension(dimension="bands", reducer=_bsi_reduce)

    elif task.name == "VV_VH_RATIO":
        vv_label = mapping["VV"]
        vh_label = mapping["VH"]

        def _vv_vh_reduce(data, context=None):
            vv = data.array_element(label=vv_label)
            vh = data.array_element(label=vh_label)
            return vv / vh

        cube = cube.reduce_dimension(dimension="bands", reducer=_vv_vh_reduce)

    elif task.name == "RVI":
        vv_label = mapping["VV"]
        vh_label = mapping["VH"]

        def _rvi_reduce(data, context=None):
            vv = data.array_element(label=vv_label)
            vh = data.array_element(label=vh_label)
            return 4 * vh / (vv + vh)

        cube = cube.reduce_dimension(dimension="bands", reducer=_rvi_reduce)

    else:
        raise NotImplementedError(f"Index not mapped (ProcessBuilder path): {task.name}")

    cube = apply_expert_ops(cube, task.expert)

    cube = _apply_aggregation(
        cube=cube,
        agg=task.agg,
        period=task.period,
        threshold=task.threshold,
        aoi_geojson=aoi_geojson,
    )
    return cube


def _compute_bandmath_cube(
    base_cube: DataCube,
    task: BandMathTask,
    mapping: Dict[str, str],
    aoi_geojson: Dict[str, Any],
) -> DataCube:
    cube = base_cube

    # FIX 6: identity band-math => filter_bands
    expr = (task.expression or "").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        role = expr
        band_id = mapping.get(role) or mapping.get(role.upper())
        if band_id:
            cube = cube.filter_bands(bands=[band_id])
            cube = apply_expert_ops(cube, task.expert)
            cube = _apply_aggregation(
                cube=cube,
                agg=task.agg,
                period=task.period,
                threshold=task.threshold,
                aoi_geojson=aoi_geojson,
            )
            return cube

    allowed_vars: Set[str] = set(mapping.keys())
    tree, used_vars = _parse_band_math_expression(task.expression, allowed_vars)

    def _bandmath_reduce(
        data,
        context=None,
        _tree=tree,
        _used_vars=used_vars,
        _mapping=mapping,
        _task_name=task.name,
    ):
        env: Dict[str, Any] = {}
        for role in _used_vars:
            band_id = _mapping.get(role) or _mapping.get(role.upper())
            if not band_id:
                raise _SafeBandMathError(
                    f"Logical role '{role}' required by band-math task '{_task_name}' "
                    f"has no band mapping in the KG."
                )
            env[role] = data.array_element(label=band_id)
        return _eval_band_math_ast(_tree, env)

    cube = cube.reduce_dimension(dimension="bands", reducer=_bandmath_reduce)

    cube = apply_expert_ops(cube, task.expert)

    cube = _apply_aggregation(
        cube=cube,
        agg=task.agg,
        period=task.period,
        threshold=task.threshold,
        aoi_geojson=aoi_geojson,
    )
    return cube


def _choose_output_format_for_task(
    default_output_format: Optional[str],
    is_rgb: bool,
    agg: Optional[AggregationName],
) -> str:
    if default_output_format:
        return default_output_format
    if is_rgb:
        return "PNG"
    if agg in {"monthly_pct_area", "trend_yearly"}:
        return "JSON"
    return "GTiff"


def _load_collection_with_optional_merge(
    oe_load_collection,
    collection_id: str,
    spatial_extent: Dict[str, Any],
    temporal_extent: Union[List[str], List[List[str]]],
    bands: Optional[List[str]],
    properties: Optional[Dict[str, Any]],
) -> DataCube:
    """
    FIX 4: se la collection è single-band-only, carica una banda alla volta e merge_cubes.
    """
    if not bands or len(bands) <= 1:
        return oe_load_collection(
            id=collection_id,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            bands=bands or None,
            properties=properties,
        )

    if _collection_single_band_only(collection_id):
        cube: Optional[DataCube] = None
        for b in bands:
            c = oe_load_collection(
                id=collection_id,
                spatial_extent=spatial_extent,
                temporal_extent=temporal_extent,
                bands=[b],
                properties=properties,
            )
            cube = c if cube is None else cube.merge_cubes(c)
        if cube is None:
            raise RuntimeError("Internal error: single-band-only merge produced no cube.")
        return cube

    return oe_load_collection(
        id=collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=bands,
        properties=properties,
    )


# ==========================================================
# 12) Autofill from user text (cloud.property, cloud.max, prefer_precomputed_indices, output_formats)
# ==========================================================

def _autofill_from_user_text(dsl: DSLRequest, instruction: str) -> DSLRequest:
    updates: Dict[str, Any] = {}

    # 1) cloud.property (override) from text, only if not already set
    inferred_prop = _extract_cloud_property_from_text(instruction)
    if inferred_prop:
        current_cloud = updates.get("cloud") if "cloud" in updates else dsl.cloud
        if (current_cloud is None) or (getattr(current_cloud, "property", None) is None):
            cloud_obj = current_cloud.model_copy(update={"property": inferred_prop}) if current_cloud else CloudMask(property=inferred_prop)
            updates["cloud"] = cloud_obj

    # 2) cloud.max from text, only if not already set
    current_cloud = updates.get("cloud") if "cloud" in updates else dsl.cloud
    if (current_cloud is None) or (current_cloud.max is None):
        inferred = _extract_cloud_max_from_text(instruction)
        if inferred is not None:
            cloud_obj = current_cloud.model_copy(update={"max": inferred}) if current_cloud else CloudMask(max=inferred)
            updates["cloud"] = cloud_obj

    if not dsl.prefer_precomputed_indices and _text_requests_precomputed_indices(instruction):
        updates["prefer_precomputed_indices"] = True

    if not dsl.output_formats:
        updates["output_formats"] = _derive_output_formats(dsl, instruction)

    if dsl.output_format is None:
        default_fmt = _extract_default_output_format_from_text(instruction)
        if default_fmt:
            updates["output_format"] = default_fmt

    return dsl.model_copy(update=updates) if updates else dsl


# ==========================================================
# 13) DSL → openEO ProcessBuilder process graph
# ==========================================================

def _dsl_to_process_graph_openeo_builder(
    dsl: DSLRequest,
    aoi_geojson: Dict[str, Any],
    endpoint_for_pg_build: str = "https://openeo.dataspace.copernicus.eu",
) -> Dict[str, Any]:
    if openeo is None:
        raise RuntimeError("openEO client is not installed, cannot use ProcessBuilder-based builder.")

    try:
        from openeo.processes import (
            load_collection as oe_load_collection,
            mask as oe_mask,
            array_create as oe_array_create,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import openeo.processes helpers: {e}")

    _validate_indices_and_bands(dsl)

    collection_id = dsl.collections[0]

    mapping = _get_band_mapping(collection_id)
    needed_bands: List[str] = _all_index_bands(dsl)

    class_mapping = _get_classification_band_mapping(collection_id)

    scl_band_id: Optional[str] = None
    if dsl.cloud and dsl.cloud.mask == "basic_s2_scl":
        scl_band_id = class_mapping.get("SCL")
        if not scl_band_id:
            logger.warning(
                "nl_to_openeo: 'basic_s2_scl' cloud mask requested for collection '%s' "
                "but no SCL band is defined in the KG (logical or classification roles). "
                "SCL mask will be ignored.",
                collection_id,
            )
            scl_band_id = None

    spatial_extent = _compute_spatial_extent_from_aoi(aoi_geojson)
    temporal_extent = _compute_temporal_extent(dsl)

    # ---- cloud metadata filter: allow explicit override via dsl.cloud.property, else KG ----
    cloud_prop: Optional[str] = None
    if dsl.cloud is not None:
        cloud_prop = getattr(dsl.cloud, "property", None) or None
        if isinstance(cloud_prop, str):
            cloud_prop = cloud_prop.strip() or None
    if not cloud_prop:
        cloud_prop = _get_cloud_cover_property(collection_id)

    properties: Optional[Dict[str, Any]] = None
    if dsl.cloud and dsl.cloud.max is not None:
        max_cloud = float(dsl.cloud.max)
        if not (0.0 <= max_cloud <= 1.0):
            logger.warning("nl_to_openeo: cloud.max (%s) outside [0,1], clipping.", max_cloud)
            max_cloud = max(0.0, min(1.0, max_cloud))

        if cloud_prop:
            properties = {
                cloud_prop: {
                    "process_graph": {
                        "cc": {
                            "process_id": "lte",
                            "arguments": {
                                "x": {"from_parameter": "value"},
                                "y": max_cloud * 100.0,
                            },
                            "result": True,
                        }
                    }
                }
            }
        else:
            raise RuntimeError(
                f"cloud.max was requested but neither DSL.cloud.property nor KG.cloud_cover_property "
                f"is available for collection '{collection_id}'. "
                f"Fix your KG (set cloud_cover_property, e.g. 'eo:cloud_cover') or specify it explicitly in the instruction."
            )

    # base cube
    base_cube: DataCube = _load_collection_with_optional_merge(
        oe_load_collection=oe_load_collection,
        collection_id=collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=needed_bands or None,
        properties=properties,
    )

    # load SCL separately (if requested)
    scl_cube: Optional[DataCube] = None
    if scl_band_id is not None:
        try:
            scl_cube = oe_load_collection(
                id=collection_id,
                spatial_extent=spatial_extent,
                temporal_extent=temporal_extent,
                bands=[scl_band_id],
                properties=properties,
            )
        except Exception:
            logger.warning(
                "nl_to_openeo: failed to load SCL band '%s' for cloud masking; "
                "continuing without SCL-based mask.",
                scl_band_id,
                exc_info=True,
            )
            scl_cube = None

    # resample target
    resample_kwargs: Optional[Dict[str, Any]] = None
    if dsl.crs or dsl.resolution is not None:
        resample_kwargs = {}

        if dsl.resolution is not None:
            resample_kwargs["resolution"] = dsl.resolution
        elif dsl.crs:
            resample_kwargs["resolution"] = 0

        if dsl.crs:
            crs = dsl.crs
            if isinstance(crs, str) and crs.upper().startswith("EPSG:"):
                try:
                    crs = int(crs.split(":")[1])
                except Exception:
                    pass
            resample_kwargs["projection"] = crs

    if resample_kwargs is not None:
        base_cube = base_cube.resample_spatial(method="bilinear", **resample_kwargs)

        if scl_cube is not None:
            try:
                scl_cube = scl_cube.resample_spatial(method="nearest", **resample_kwargs)
            except Exception:
                logger.warning(
                    "nl_to_openeo: resample_spatial(nearest) on SCL cube failed; continuing with native SCL grid.",
                    exc_info=True,
                )

    # apply SCL cloud mask
    if scl_cube is not None:
        def scl_to_cloud_mask(x, context=None, _scl_band_id=scl_band_id):
            scl = x.array_element(label=_scl_band_id)
            is_cloud = ((scl == 3) + (scl == 8) + (scl == 9) + (scl == 10)) > 0
            return is_cloud

        try:
            cloud_mask = scl_cube.apply(scl_to_cloud_mask)
            base_cube = oe_mask(data=base_cube, mask=cloud_mask)
        except Exception:
            logger.warning(
                "nl_to_openeo: SCL cloud mask application failed; continuing without SCL-based mask.",
                exc_info=True,
            )

    # AOI polygon: skip if exact bbox rectangle
    try:
        if _is_aoi_exact_bbox_rectangle(aoi_geojson, spatial_extent):
            logger.info("nl_to_openeo: skipping mask_polygon because AOI is exact bbox rectangle.")
        else:
            base_cube = base_cube.mask_polygon(mask=aoi_geojson)
    except Exception:
        logger.warning(
            "nl_to_openeo: mask_polygon with AOI failed; continuing with bbox-only spatial_extent.",
            exc_info=True,
        )

    # ---- normalize result graph helper ----
    def _normalize_results(obj: Dict[str, Any]) -> Dict[str, Any]:
        if "process_graph" in obj and isinstance(obj["process_graph"], dict):
            pg = obj["process_graph"]
        else:
            pg = obj

        if not isinstance(pg, dict):
            return obj

        save_nodes = []
        array_nodes = []

        for _node_id, node in pg.items():
            if not isinstance(node, dict):
                continue
            pid = node.get("process_id")
            if pid == "save_result":
                save_nodes.append(node)
            elif pid == "array_create":
                array_nodes.append(node)

        for node in pg.values():
            if isinstance(node, dict) and "result" in node:
                node.pop("result", None)

        if array_nodes:
            array_nodes[0]["result"] = True
        elif save_nodes:
            save_nodes[0]["result"] = True

        return {"process_graph": pg}

    # ==========================================================
    # FIX 7: auto-collapse pure band export
    # ==========================================================
    if (dsl.output_packing == "multi_asset") and (not dsl.indices) and dsl.band_math:
        wanted_band_ids: List[str] = []
        ok = True

        for bm in dsl.band_math:
            if bm.expert or bm.agg or bm.period or (bm.threshold is not None):
                ok = False
                break

            expr = (bm.expression or "").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
                ok = False
                break

            if (bm.name or "").strip().upper() != expr.upper():
                ok = False
                break

            band_id = mapping.get(expr) or mapping.get(expr.upper())
            if not band_id:
                ok = False
                break

            wanted_band_ids.append(band_id)

        if ok and wanted_band_ids:
            export_cube = base_cube.filter_bands(bands=wanted_band_ids)
            export_cube = _apply_global_temporal_aggregation(export_cube, dsl.temporal_aggregation)

            fmt = dsl.output_format or ("NetCDF" if len(wanted_band_ids) > 1 else "GTiff")
            result_builder = oe_save_result(data=export_cube, format=fmt, options={})

            pg_obj = json.loads(result_builder.to_json())
            return _normalize_results(pg_obj)

    # ==========================================================
    # Helper: apply common postprocessing to any cube (base or extra collections)
    # ==========================================================
    def _apply_common_postprocessing(c: DataCube) -> DataCube:
        cc = c

        if resample_kwargs is not None:
            cc = cc.resample_spatial(method="bilinear", **resample_kwargs)

        if scl_cube is not None:
            try:
                def scl_to_cloud_mask2(x, context=None, _scl_band_id=scl_band_id):
                    scl = x.array_element(label=_scl_band_id)
                    is_cloud = ((scl == 3) + (scl == 8) + (scl == 9) + (scl == 10)) > 0
                    return is_cloud
                cloud_mask2 = scl_cube.apply(scl_to_cloud_mask2)
                cc = oe_mask(data=cc, mask=cloud_mask2)
            except Exception:
                logger.warning("nl_to_openeo: SCL cloud mask failed on extra cube; skipping.", exc_info=True)

        try:
            if not _is_aoi_exact_bbox_rectangle(aoi_geojson, spatial_extent):
                cc = cc.mask_polygon(mask=aoi_geojson)
        except Exception:
            logger.warning("nl_to_openeo: mask_polygon failed on extra cube; skipping.", exc_info=True)

        return cc

    # ==========================================================
    # Build result cubes (with precomputed indices if enabled)
    # ==========================================================
    index_results: List[Tuple[str, DataCube, Optional[AggregationName], bool]] = []

    for task in dsl.indices:
        try:
            # Precomputed indices (skip RGB)
            if dsl.prefer_precomputed_indices and task.name != "RGB":
                src = _get_precomputed_source_for_index(task.name)
                if src is not None:
                    pre_coll = src["collection_id"]
                    pre_band = src["band"]

                    pre_cloud_prop = _get_cloud_cover_property(pre_coll)
                    pre_properties = None
                    if dsl.cloud and dsl.cloud.max is not None and pre_cloud_prop:
                        pre_properties = {
                            pre_cloud_prop: {
                                "process_graph": {
                                    "cc": {
                                        "process_id": "lte",
                                        "arguments": {
                                            "x": {"from_parameter": "value"},
                                            "y": float(dsl.cloud.max) * 100.0,
                                        },
                                        "result": True,
                                    }
                                }
                            }
                        }

                    pre_cube = _load_collection_with_optional_merge(
                        oe_load_collection=oe_load_collection,
                        collection_id=pre_coll,
                        spatial_extent=spatial_extent,
                        temporal_extent=temporal_extent,
                        bands=[pre_band],
                        properties=pre_properties,
                    )

                    pre_cube = _apply_common_postprocessing(pre_cube)
                    pre_cube = apply_expert_ops(pre_cube, task.expert)
                    pre_cube = _apply_aggregation(
                        cube=pre_cube,
                        agg=task.agg,
                        period=task.period,
                        threshold=task.threshold,
                        aoi_geojson=aoi_geojson,
                    )

                    index_results.append((task.name, pre_cube, task.agg, False))
                    continue

            # fallback: compute from base collection
            idx_cube = _compute_predefined_index_cube(base_cube, task, mapping, aoi_geojson)
            is_rgb = (task.name == "RGB")
            index_results.append((task.name, idx_cube, task.agg, is_rgb))

        except Exception as e:
            raise RuntimeError(f"Failed to compute index '{task.name}': {e}") from e

    if dsl.band_math:
        for bm in dsl.band_math:
            try:
                bm_cube = _compute_bandmath_cube(base_cube, bm, mapping, aoi_geojson)
                index_results.append((bm.name, bm_cube, bm.agg, False))
            except _SafeBandMathError as e:
                raise RuntimeError(f"Band-math task '{bm.name}' has invalid expression: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to compute band-math task '{bm.name}': {e}") from e

    if not index_results:
        raise RuntimeError(
            "No indices or band-math tasks produced any result cube. "
            "This should not happen if DSL has at least one task."
        )

    packing = dsl.output_packing or "multi_asset"

    # ==========================================================
    # SAVE: multi_band
    # ==========================================================
    if packing == "multi_band":
        # guard: no RGB
        for name, _cube, _agg, is_rgb in index_results:
            if is_rgb:
                raise RuntimeError(
                    "output_packing='multi_band' cannot include RGB outputs because RGB is inherently multi-band. "
                    "Use output_packing='multi_asset' or remove RGB."
                )

        # choose final format:
        # 1) explicit dsl.output_format
        # 2) if output_formats exists -> all must match (and not JSON/PNG)
        # 3) else default NetCDF
        if dsl.output_format:
            chosen_fmt = dsl.output_format
        elif dsl.output_formats:
            uniq = {dsl.output_formats.get(name) for (name, _, _, _) in index_results}
            uniq.discard(None)
            if len(uniq) != 1:
                raise RuntimeError(
                    "output_packing='multi_band' requires a single output format. "
                    f"Got per-task formats: {sorted(uniq)}"
                )
            chosen_fmt = list(uniq)[0]
        else:
            chosen_fmt = "NetCDF"

        if str(chosen_fmt).upper() in {"JSON", "PNG"}:
            raise RuntimeError(
                f"output_packing='multi_band' is not compatible with format '{chosen_fmt}'. "
                "Use 'GTiff' or 'NetCDF', or switch to output_packing='multi_asset'."
            )

        multi_cube: Optional[DataCube] = None

        for name, cube, agg, is_rgb in index_results:
            cube = _apply_global_temporal_aggregation(cube, dsl.temporal_aggregation)
            labeled = cube.add_dimension(name="bands", label=name, type="bands")
            multi_cube = labeled if multi_cube is None else multi_cube.merge_cubes(labeled)

        result_builder = oe_save_result(data=multi_cube, format=chosen_fmt, options={})

    # ==========================================================
    # SAVE: multi_asset
    # ==========================================================
    else:
        save_nodes: List[DataCube] = []

        for name, cube, agg, is_rgb in index_results:
            if dsl.output_formats and dsl.output_formats.get(name):
                fmt = dsl.output_formats[name]
            else:
                fmt = _choose_output_format_for_task(
                    default_output_format=dsl.output_format,
                    is_rgb=is_rgb,
                    agg=agg,
                )

            cube_to_save = _apply_global_temporal_aggregation(cube, dsl.temporal_aggregation)
            save_node = oe_save_result(data=cube_to_save, format=fmt, options={})
            save_nodes.append(save_node)

        if len(save_nodes) == 1:
            result_builder = save_nodes[0]
        else:
            from openeo.processes import array_create as oe_array_create
            result_builder = oe_array_create(data=save_nodes)

    try:
        pg_json = result_builder.to_json()
        pg_obj = json.loads(pg_json)
    except Exception as e:
        raise RuntimeError(f"Failed to export ProcessBuilder to JSON: {e}")

    if isinstance(pg_obj, dict):
        # normalize results
        if "process_graph" in pg_obj and isinstance(pg_obj["process_graph"], dict):
            pg = pg_obj["process_graph"]
        else:
            pg = pg_obj

        save_nodes = []
        array_nodes = []

        for _node_id, node in pg.items():
            if not isinstance(node, dict):
                continue
            pid = node.get("process_id")
            if pid == "save_result":
                save_nodes.append(node)
            elif pid == "array_create":
                array_nodes.append(node)

        for node in pg.values():
            if isinstance(node, dict) and "result" in node:
                node.pop("result", None)

        if array_nodes:
            array_nodes[0]["result"] = True
        elif save_nodes:
            save_nodes[0]["result"] = True

        return {"process_graph": pg}
    else:
        raise RuntimeError("Unexpected type from result_builder.to_json().")


def dsl_to_process_graph(dsl: DSLRequest, aoi_geojson: Dict[str, Any]) -> Dict[str, Any]:
    return _dsl_to_process_graph_openeo_builder(dsl, aoi_geojson)


# ==========================================================
# 14) LangChain tooling (LLM → DSL / DIRECT PG)
# ==========================================================

def _extract_first_json_block(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found in model output.")
    chunk = _strip_trailing_commas(text[start : end + 1])
    return json.loads(chunk)


def _normalize_dsl_payload(raw: Dict[str, Any], default_collection: str) -> Dict[str, Any]:
    """
    Normalizza wrappers e applica SOLO default deterministici.

    FIX 1:
      - accetta 'temporal_extent' come sinonimo
      - accetta formato openEO: ["YYYY-MM-DD","YYYY-MM-DD"]
      - accetta formato openEO multi-range: [["YYYY-MM-DD","YYYY-MM-DD"], ...]
      - normalizza sempre in: temporal = [{"start":..,"end":..}, ...]

    FIX 2:
      - non richiede più 'indices' obbligatorio
      - richiede almeno uno tra 'indices' e 'band_math'

    FIX 8:
      - normalizza temporal_aggregation / aggregate_temporal_period (global)
        in: temporal_aggregation = {"period": "dekad|month|year", "reducer":"mean|median|max|min"}
    """
    cand = dict(raw)

    for k in ("value", "data", "dsl"):
        if isinstance(cand.get(k), dict):
            cand = cand[k]
    if "type" in raw and isinstance(raw.get("value"), dict):
        cand = raw["value"]

    # collections
    if "collections" not in cand or cand.get("collections") is None:
        cand["collections"] = [default_collection]

    collections_field = cand.get("collections")

    if not isinstance(collections_field, list):
        raise ValueError(
            f"'collections' must be a list of collection IDs, got {type(collections_field).__name__}."
        )
    if not collections_field:
        raise ValueError(
            "'collections' must contain exactly one collection ID, got an empty list."
        )

    collection = str(collections_field[0])
    cand["collections"] = [collection]

    # temporal (FIX 1)
    if "temporal" not in cand or cand.get("temporal") is None:
        temporal_synonyms = [
            "temporal_extent",
            "time_ranges",
            "time_range",
            "time",
            "time_extent",
            "date_range",
            "date_ranges",
            "data_range",
            "data_ranges",
            "time_tanges",
        ]
        for alt in temporal_synonyms:
            if alt in cand and cand.get(alt) is not None:
                cand["temporal"] = cand[alt]
                break

    if "temporal" not in cand or cand.get("temporal") is None:
        raise ValueError(
            "Missing 'temporal' in DSLRequest. The time range must be specified explicitly in the LLM output."
        )

    tmp_temporal = cand["temporal"]

    raw_list = None

    if isinstance(tmp_temporal, dict):
        raw_list = [tmp_temporal]

    elif isinstance(tmp_temporal, list):
        if len(tmp_temporal) == 2 and all(isinstance(x, str) for x in tmp_temporal):
            raw_list = [{"start": tmp_temporal[0], "end": tmp_temporal[1]}]

        elif all(
            isinstance(x, list) and len(x) == 2 and all(isinstance(y, str) for y in x)
            for x in tmp_temporal
        ):
            raw_list = [{"start": x[0], "end": x[1]} for x in tmp_temporal]

        elif all(isinstance(x, dict) for x in tmp_temporal):
            raw_list = tmp_temporal

        else:
            raise ValueError(f"'temporal' list format not recognized: {tmp_temporal!r}")

    else:
        raise ValueError(
            f"'temporal' must be a dict, a list of dicts, or openEO style [start,end]; got {type(tmp_temporal).__name__}."
        )

    normalized_temporal: List[Dict[str, Any]] = []
    for t in raw_list:
        if not isinstance(t, dict):
            raise ValueError(f"Temporal element must be an object, got {type(t).__name__}: {t!r}")
        t2 = dict(t)

        if "start" not in t2:
            if "start_date" in t2:
                t2["start"] = t2["start_date"]
            elif "from" in t2:
                t2["start"] = t2["from"]

        if "end" not in t2:
            if "end_date" in t2:
                t2["end"] = t2["end_date"]
            elif "to" in t2:
                t2["end"] = t2["to"]

        if "start" not in t2 or "end" not in t2:
            raise ValueError(f"Temporal range must have 'start' and 'end' fields, got {t!r}.")

        normalized_temporal.append({"start": str(t2["start"]), "end": str(t2["end"])})

    if not normalized_temporal:
        raise ValueError("Empty or invalid 'temporal' list after normalization.")

    cand["temporal"] = normalized_temporal

    # FIX 8: global temporal aggregation normalization
    if "temporal_aggregation" not in cand or cand.get("temporal_aggregation") is None:
        for alt in ("aggregate_temporal_period", "global_temporal_aggregation", "temporal_agg", "temporal_reduce"):
            if isinstance(cand.get(alt), dict):
                cand["temporal_aggregation"] = cand[alt]
                break

    ta = cand.get("temporal_aggregation")
    if isinstance(ta, dict):
        ta = dict(ta)
        period = ta.get("period")
        reducer = ta.get("reducer", "mean")

        if not isinstance(period, str) or not period.strip():
            raise ValueError("temporal_aggregation.period must be one of: dekad, month, year.")
        p = period.strip().lower()
        period_map = {
            "dekad": "dekad", "dekadal": "dekad", "decade": "dekad", "dekade": "dekad", "10days": "dekad", "10-days": "dekad",
            "month": "month", "monthly": "month", "mensile": "month", "mese": "month", "mesi": "month",
            "year": "year", "yearly": "year", "annual": "year", "annuale": "year", "anno": "year", "anni": "year",
        }
        if p not in period_map:
            raise ValueError(f"Unsupported temporal_aggregation.period: {period!r}")
        ta["period"] = period_map[p]

        if not isinstance(reducer, str) or not reducer.strip():
            raise ValueError("temporal_aggregation.reducer must be one of: mean, median, max, min.")
        r = reducer.strip().lower()
        reducer_map = {
            "avg": "mean",
            "average": "mean",
            "mean": "mean",
            "median": "median",
            "max": "max",
            "maximum": "max",
            "min": "min",
            "minimum": "min",
        }
        if r not in reducer_map:
            raise ValueError(f"Unsupported temporal_aggregation.reducer: {reducer!r}")
        ta["reducer"] = reducer_map[r]

        cand["temporal_aggregation"] = ta

    # FIX 2: require tasks
    if (not cand.get("indices")) and (not cand.get("band_math")):
        raise ValueError(
            "Missing tasks in DSLRequest. Provide at least one predefined index in 'indices' "
            "or at least one custom expression in 'band_math'."
        )

    # cloud.max normalized (cloud.property is allowed as-is)
    cloud = cand.get("cloud")
    if isinstance(cloud, dict) and cloud.get("max") is not None:
        try:
            max_val = float(cloud["max"])
            if not (0.0 <= max_val <= 1.0):
                logger.warning("nl_to_openeo: cloud.max outside [0,1]; clipping to range.")
                max_val = max(0.0, min(1.0, max_val))
            cloud["max"] = max_val
        except Exception:
            logger.warning(
                "nl_to_openeo: invalid cloud.max value %r, removing cloud section.",
                cloud.get("max"),
            )
            cand.pop("cloud", None)
            cloud = None

    cloud = cand.get("cloud")
    if isinstance(cloud, dict) and cloud.get("mask") == "basic_s2_scl":
        try:
            class_map = _get_classification_band_mapping(collection)
        except Exception as e:
            logger.warning(
                "nl_to_openeo: cannot get classification band mapping while normalizing cloud mask for '%s': %s",
                collection,
                e,
            )
            class_map = {}

        if not (isinstance(class_map, dict) and class_map.get("SCL")):
            logger.warning(
                "nl_to_openeo: 'basic_s2_scl' cloud mask requested for collection '%s' "
                "but no SCL band in KG (classification_roles/logical_roles). Removing cloud mask.",
                collection,
            )
            cand.pop("cloud", None)

    # indices normalization (with FIX 8 default-mean STOP if temporal_aggregation exists)
    normalized_indices = []
    allowed_names = {
        "NDVI", "NDWI", "NDSI", "EVI", "RGB",
        "SAVI", "OSAVI", "MSAVI", "MSAVI2", "GNDVI", "NDRE", "ARVI",
        "MNDWI", "AWEI_SH", "AWEI_NSH",
        "NBR", "NBR2", "BAI",
        "NDBI", "BSI",
        "VV_VH_RATIO", "RVI",
    }

    if isinstance(cand.get("indices"), list):
        for idx in cand.get("indices", []):
            if not isinstance(idx, dict):
                continue

            name = idx.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each IndexTask must have a non-empty 'name' field.")

            name_upper = name.upper()
            if name_upper in {"TRUE_COLOR", "TRUE_COLOUR", "TRUE COLOUR", "TRUE-COLOR"}:
                name_upper = "RGB"

            if name_upper not in allowed_names:
                raise ValueError(
                    f"Unsupported index name '{name}'. "
                    f"Allowed indices are: {sorted(allowed_names)}."
                )

            idx["name"] = name_upper
            name = name_upper

            allowed_for_coll = _get_allowed_indices(collection)
            if name not in allowed_for_coll:
                raise ValueError(
                    f"Index '{name}' is not supported for collection '{collection}' "
                    f"according to the knowledge graph. Allowed indices: {allowed_for_coll}."
                )

            agg = idx.get("agg")
            if isinstance(agg, str):
                agg_l = agg.lower()
                agg_map = {
                    "average": "mean",
                    "avg": "mean",
                    "mean": "mean",
                    "median": "median",
                    "max": "max",
                    "maximum": "max",
                    "min": "min",
                    "minimum": "min",
                }
                if agg_l in agg_map:
                    idx["agg"] = agg_map[agg_l]

            period = idx.get("period")
            if isinstance(period, str):
                p_l = period.lower()
                if p_l in {"month", "monthly", "mensile", "mese", "mesi"}:
                    idx["period"] = "month"
                elif p_l in {"year", "yearly", "annual", "annuale", "anno", "anni"}:
                    idx["period"] = "year"
                elif p_l in {"dekad", "dekadal", "decade"}:
                    idx["period"] = "dekad"

            continuous_mean_default = {
                "NDVI", "NDWI", "NDSI", "EVI",
                "SAVI", "OSAVI", "MSAVI", "MSAVI2",
                "GNDVI", "NDRE", "ARVI",
                "MNDWI", "AWEI_SH", "AWEI_NSH",
                "NBR", "NBR2", "BAI",
                "NDBI", "BSI",
                "VV_VH_RATIO",
                "RVI",
            }

            if name == "RGB":
                idx.pop("agg", None)
                idx.pop("threshold", None)
                idx.pop("period", None)

            elif name == "NDSI":
                agg_val = idx.get("agg")
                if agg_val == "monthly_pct_area":
                    if idx.get("threshold") is None:
                        idx["threshold"] = 0.4
                    if not idx.get("period"):
                        idx["period"] = "month"
                else:
                    # FIX 8: default mean only if NO global temporal_aggregation
                    if cand.get("temporal_aggregation") is None:
                        if agg_val is None:
                            idx["agg"] = "mean"

            elif name in continuous_mean_default:
                # FIX 8: default mean only if NO global temporal_aggregation
                if cand.get("temporal_aggregation") is None:
                    if idx.get("agg") is None:
                        idx["agg"] = "mean"

            normalized_indices.append(idx)

    if normalized_indices:
        cand["indices"] = normalized_indices
    else:
        cand.pop("indices", None)

    # band_math normalization
    band_math_raw = cand.get("band_math")
    if isinstance(band_math_raw, list):
        normalized_band_math = []
        for bm in band_math_raw:
            if not isinstance(bm, dict):
                continue
            name = bm.get("name")
            expr = bm.get("expression")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each BandMathTask must have a non-empty 'name' field.")
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError("Each BandMathTask must have a non-empty 'expression' field.")

            agg = bm.get("agg")
            if isinstance(agg, str):
                agg_l = agg.lower()
                agg_map = {
                    "average": "mean",
                    "avg": "mean",
                    "mean": "mean",
                    "median": "median",
                    "max": "max",
                    "maximum": "max",
                    "min": "min",
                    "minimum": "min",
                }
                if agg_l in agg_map:
                    bm["agg"] = agg_map[agg_l]

            period = bm.get("period")
            if isinstance(period, str):
                p_l = period.lower()
                if p_l in {"month", "monthly", "mensile", "mese", "mesi"}:
                    bm["period"] = "month"
                elif p_l in {"year", "yearly", "annual", "annuale", "anno", "anni"}:
                    bm["period"] = "year"
                elif p_l in {"dekad", "dekadal", "decade"}:
                    bm["period"] = "dekad"

            normalized_band_math.append(bm)

        if normalized_band_math:
            cand["band_math"] = normalized_band_math
        else:
            cand.pop("band_math", None)

    # output_packing normalization
    packing_raw = cand.get("output_packing")
    if isinstance(packing_raw, str):
        p = packing_raw.strip().lower()
        if p in {"multi_band", "multi-band", "stack_bands", "stacked", "bands"}:
            cand["output_packing"] = "multi_band"
        else:
            cand["output_packing"] = "multi_asset"
    else:
        cand["output_packing"] = "multi_asset"

    if (not cand.get("indices")) and (not cand.get("band_math")):
        raise ValueError(
            "After normalization, no valid tasks remained in 'indices' or 'band_math'. "
            "Provide at least one task."
        )

    return cand


# ==========================================================
# 14A) DIRECT PG routing + result normalization
# ==========================================================

def _instruction_looks_like_direct_pg(instruction: str) -> bool:
    s = (instruction or "").lower()
    # routing semplice (non parsing bbox/geojson)
    return (
        "build a process graph" in s
        or "process_graph" in s
        or ("load collection" in s and "save_result" in s)
    )


def _normalize_pg_result_flag(pg_obj: dict) -> dict:
    """
    Garantisce che ci sia un solo nodo result:true.
    Se nessuno è marcato, prova a marcare un save_result.
    """
    if not isinstance(pg_obj, dict):
        return pg_obj

    pg = pg_obj.get("process_graph") if isinstance(pg_obj.get("process_graph"), dict) else pg_obj
    if not isinstance(pg, dict):
        return pg_obj

    for node in pg.values():
        if isinstance(node, dict):
            node.pop("result", None)

    save_ids = [k for k, v in pg.items() if isinstance(v, dict) and v.get("process_id") == "save_result"]
    if save_ids:
        pg[save_ids[0]]["result"] = True
    else:
        first_id = next(iter(pg.keys()), None)
        if first_id:
            pg[first_id]["result"] = True

    return {"process_graph": pg}


# ==========================================================
# 15) Tools
# ==========================================================

@tool
def build_process_graph_from_instruction(
    instruction: str,
    aoi_feature_collection_json: str,
    default_collection: str = "SENTINEL2_L2A",
    return_dsl: bool = False,
) -> str:
    """
    - AOI override via LLM:
        Se nel testo c'è una AOI esplicita (bbox/geojson), l'LLM la estrae in un campo top-level "aoi"
        e questa sovrascrive la AOI passata come default (aoi_feature_collection_json).
        Se non c'è AOI esplicita, "aoi" viene omesso e si usa la default AOI.
    - DIRECT mode:
        Se l'istruzione sembra una specifica procedurale di process graph (es. "BUILD A PROCESS GRAPH..."),
        bypassa il DSL e fa generare all'LLM direttamente il process graph JSON (supporta multi-collection/merge).
    """
    try:
        _load_kg()
    except Exception as e:
        return json.dumps(
            {"error": "Knowledge graph loading failed", "details": str(e)},
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------
    # MODE 1: DIRECT PG
    # -----------------------
    if _instruction_looks_like_direct_pg(instruction):
        model = ChatOpenAI(
            api_key=_read_api_key(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.5-flash",
            temperature=0,
        )

        system = SystemMessage(
            content=(
                "You convert the user's instruction into an openEO process graph JSON.\n"
                "Return ONLY JSON.\n"
                "Valid outputs:\n"
                "  - {\"process_graph\": { ... }}\n"
                "  - or directly { ... } where keys are node ids and values are nodes (we will wrap it).\n"
                "\n"
                "Rules:\n"
                "- If the instruction specifies collection ids, bands, bbox, temporal extent, properties filters, follow it exactly (copy numbers verbatim).\n"
                "- If the instruction does NOT specify a spatial filter, you MAY use the provided DEFAULT AOI below.\n"
                "- Ensure exactly ONE node has \"result\": true (prefer save_result).\n"
            )
        )

        user = HumanMessage(
            content=(
                f"DEFAULT_COLLECTION:\n{default_collection}\n\n"
                f"DEFAULT_AOI_GEOJSON:\n{aoi_feature_collection_json}\n\n"
                f"INSTRUCTION:\n{instruction}\n"
            )
        )

        raw_text = model.invoke([system, user]).content or ""
        try:
            raw_obj = _extract_first_json_block(raw_text)
        except Exception:
            return json.dumps(
                {"error": "Direct PG construction failed", "details": "LLM did not return valid JSON."},
                ensure_ascii=False,
                indent=2,
            )

        if isinstance(raw_obj, dict) and "process_graph" in raw_obj:
            pg_obj = raw_obj
        elif isinstance(raw_obj, dict):
            pg_obj = {"process_graph": raw_obj}
        else:
            return json.dumps(
                {"error": "Direct PG construction failed", "details": "Unexpected JSON type from LLM."},
                ensure_ascii=False,
                indent=2,
            )

        pg_obj = _normalize_pg_result_flag(pg_obj)
        return json.dumps(pg_obj, ensure_ascii=False, indent=2)

    # -----------------------
    # MODE 2: DSL -> KG -> PG
    # -----------------------
    model = ChatOpenAI(
        api_key=_read_api_key(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.0-flash",
        temperature=0,
    )

    system = SystemMessage(
        content=(
            "You convert natural language Earth observation requests into JSON.\n"
            "\n"
            "Return ONLY JSON.\n"
            "\n"
            "Output format:\n"
            "- You may return either:\n"
            "  (A) a plain DSLRequest object, OR\n"
            "  (B) an object {\"dsl\": <DSLRequest>, \"aoi\": <AOI>}.\n"
            "\n"
            "AOI extraction (IMPORTANT):\n"
            "- If the instruction contains an explicit spatial extent (e.g. filter_bbox extent with west/east/south/north, "
            "  or an explicit bbox array, or an embedded GeoJSON), extract it into top-level field \"aoi\".\n"
            "- \"aoi\" MUST be either:\n"
            "    * a bbox array: [west, south, east, north]  (copy numbers verbatim), OR\n"
            "    * a GeoJSON Feature/FeatureCollection/geometry.\n"
            "- If you do NOT see an explicit AOI in the instruction, OMIT \"aoi\" entirely.\n"
            "- Do NOT guess the AOI.\n"
            "\n"
            "VERY IMPORTANT: Band mapping and index availability are handled by a separate knowledge graph in Python. "
            "You MUST NOT invent bands.\n"
            "\n"
            "The JSON MUST contain at least one task, either:\n"
            "  - a top-level 'indices' array with one or more predefined IndexTask objects, OR\n"
            "  - a top-level 'band_math' array with one or more BandMathTask objects.\n"
            "\n"
            "Supported predefined index names are exactly: "
            "\"NDVI\", \"NDWI\", \"NDSI\", \"EVI\", \"RGB\", "
            "\"SAVI\", \"OSAVI\", \"MSAVI\", \"MSAVI2\", \"GNDVI\", \"NDRE\", \"ARVI\", "
            "\"MNDWI\", \"AWEI_SH\", \"AWEI_NSH\", "
            "\"NBR\", \"NBR2\", \"BAI\", "
            "\"NDBI\", \"BSI\", "
            "\"VV_VH_RATIO\", \"RVI\". "
            "Do not use any other predefined index name.\n"
            "\n"
            "For band_math:\n"
            "Each element is an object with:\n"
            "  - name: string label for the derived band/index\n"
            "  - expression: arithmetic expression using logical roles like "
            "    'RED', 'GREEN', 'BLUE', 'NIR', 'SWIR1', 'SWIR2', 'VV', 'VH', etc.\n"
            "    Only +, -, *, /, ** and parentheses are allowed.\n"
            "  - optional 'agg', 'period', 'threshold'\n"
            "\n"
            "Global temporal aggregation (FIX 8):\n"
            "- If the user explicitly asks for 'aggregate_temporal_period' on the FINAL merged cube (e.g. dekad mean),\n"
            "  set a top-level field:\n"
            "    \"temporal_aggregation\": { \"period\": \"dekad\"|\"month\"|\"year\", \"reducer\": \"mean\"|\"median\"|\"max\"|\"min\" }\n"
            "\n"
            "Temporal rules:\n"
            "- If the user gives explicit start and end dates, copy them exactly.\n"
            "- All dates must be ISO YYYY-MM-DD.\n"
            "\n"
            "Cloud / SCL masking:\n"
            "- If the user asks for SCL cloud masking, set \"cloud\": { \"mask\": \"basic_s2_scl\" }.\n"
            "- If the user asks for max cloud cover like 20%, set \"cloud\": { \"max\": 0.2 }.\n"
            "- If the user explicitly mentions a cloud-cover metadata property (e.g. eo:cloud_cover), you MAY set:\n"
            "    \"cloud\": { \"property\": \"eo:cloud_cover\" }\n"
            "\n"
            "Collections:\n"
            "- The DSL field 'collections' MUST contain exactly one collection ID string if provided.\n"
            "- If user does not mention any collection, you MAY omit 'collections' (Python will use default).\n"
            "\n"
            "Output packing:\n"
            "- Default is \"multi_asset\".\n"
            "- If user wants one raster with multiple bands (one band per task), set \"multi_band\".\n"
            "- Do NOT use \"multi_band\" together with JSON outputs.\n"
            f"- If you completely omit 'collections', Python will use ['{default_collection}'].\n"
        )
    )

    user = HumanMessage(
        content=f"Instruction:\n{instruction}\nReturn ONLY JSON."
    )

    raw_text = model.invoke([system, user]).content or ""

    try:
        raw_obj = _extract_first_json_block(raw_text)
    except Exception:
        return json.dumps(
            {"error": "DSL construction failed", "details": "LLM did not return valid JSON."},
            ensure_ascii=False,
            indent=2,
        )

    if not isinstance(raw_obj, dict) or not raw_obj:
        return json.dumps(
            {"error": "DSL construction failed", "details": "LLM returned empty or non-object JSON."},
            ensure_ascii=False,
            indent=2,
        )

    # Wrapper support: {"dsl": {...}, "aoi": ...}
    aoi_from_llm = None
    if isinstance(raw_obj.get("aoi"), (dict, list, str)):
        aoi_from_llm = raw_obj["aoi"]

    dsl_payload = raw_obj.get("dsl") if isinstance(raw_obj.get("dsl"), dict) else raw_obj

    try:
        normalized = _normalize_dsl_payload(dsl_payload, default_collection)
    except ValueError as e:
        return json.dumps(
            {"error": "DSL normalization failed", "details": str(e)},
            ensure_ascii=False,
            indent=2,
        )

    if not normalized.get("crs"):
        m = re.search(r"EPSG:\d{4,5}", instruction)
        if m:
            normalized["crs"] = m.group(0)

    try:
        dsl = DSLRequest.model_validate(normalized)
    except ValidationError as e:
        return json.dumps(
            {"error": "DSL validation failed", "details": json.loads(e.json())},
            ensure_ascii=False,
            indent=2,
        )

    # hook autofill (cloud.property, cloud.max, prefer_precomputed_indices, output_formats)
    dsl = _autofill_from_user_text(dsl, instruction)

    # AOI selection: LLM AOI overrides default AOI
    aoi_input = aoi_from_llm if aoi_from_llm is not None else aoi_feature_collection_json

    try:
        fc = parse_geojson_flexible(aoi_input)
    except Exception as e:
        return json.dumps(
            {
                "error": "AOI parsing failed",
                "details": str(e),
                "aoi_source": "llm" if aoi_from_llm is not None else "default",
            },
            ensure_ascii=False,
            indent=2,
        )

    try:
        graph = dsl_to_process_graph(dsl, fc)
    except Exception as e:
        return json.dumps(
            {"error": "Process graph construction failed", "details": str(e)},
            ensure_ascii=False,
            indent=2,
        )

    if return_dsl:
        payload: Dict[str, Any] = {"dsl": dsl.model_dump(mode="json"), "process_graph": graph}
    else:
        payload = graph

    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def submit_process_graph(
    process_graph_json: str,
    endpoint: str = "https://openeo.dataspace.copernicus.eu",
) -> str:
    """
    Submit an openEO process graph to a backend and start a job.
    Returns a JSON with job_id and initial status. Requires openeo client.
    """
    if openeo is None:
        return json.dumps({"error": "openeo client not installed. pip install openeo"}, indent=2)
    try:
        conn = openeo.connect(endpoint).authenticate_oidc()
        pg = json.loads(process_graph_json)
        job = conn.create_job(pg)
        job.start()
        info = job.describe()
        return json.dumps({"job_id": job.job_id, "status": info.get("status")}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def demo_run_with_your_widgets_textarea(ta_value: str, user_request: str):
    """
    Helper per Jupyter/widget:
      - prende AOI dal textarea
      - prende richiesta NL dall'utente
      - usa tool LLM+KG per costruire il process graph
    """
    TOOLS: List[StructuredTool] = [build_process_graph_from_instruction, submit_process_graph]

    model = ChatOpenAI(
        api_key=_read_api_key(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        temperature=0,
    )
    model_with_tools = model.bind_tools(TOOLS)

    sys_msg = SystemMessage(
        content=(
            "You help the user build openEO process graphs from natural language. "
            "When appropriate, call the build_process_graph_from_instruction tool with the AOI FeatureCollection. "
            "Return concise answers with a short code block."
        )
    )
    human = HumanMessage(
        content=(
            f"AOI FeatureCollection:\n{ta_value}\n\n"
            f"REQUEST:\n{user_request}\n"
            "Build the openEO process graph."
        )
    )

    msgs = [sys_msg, human]
    ai_1: AIMessage = model_with_tools.invoke(msgs)

    tool_messages: List[ToolMessage] = []
    if getattr(ai_1, "tool_calls", None):
        for tc in ai_1.tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            call_id = tc.get("id")
            if name == "build_process_graph_from_instruction":
                args["aoi_feature_collection_json"] = ta_value
            result = [t for t in TOOLS if t.name == name][0].invoke(args)
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call_id, name=name)
            )

    msgs_final = msgs + [ai_1] + tool_messages
    ai_final: AIMessage = model.invoke(msgs_final)
    return ai_final.content

