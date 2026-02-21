# `collections_kg.py` — openEO Collection Knowledge Graph Builder

Builds a **JSON knowledge graph** (`collections_kg.json`) from an openEO backend (e.g. CDSE), by inspecting collection metadata and deriving EO-friendly semantics.

## What it does

For each collection, it:

* fetches metadata via `list_collections()` + `describe_collection()`
* extracts band info (`name`, `common_name`, wavelengths, etc.)
* assigns **logical roles** (`RED`, `GREEN`, `BLUE`, `NIR`, `SWIR1`, `SWIR2`, `RED_EDGE`, `VV`, `VH`, `SCL`, ...)
* infers **sensor type** (`optical`, `sar`, `other`)
* computes **supported indices** (e.g. `NDVI`, `NBR`, `NDWI`, `RVI`) using a local `INDEX_DEFS` catalog
* exports everything to `collections_kg.json`

## Why it’s useful

This script creates a reusable **grounding layer** for EO automation pipelines:

* avoids hardcoded band mappings in downstream code
* enables deterministic validation of collection/index compatibility
* improves robustness for NL/LLM → openEO process graph workflows
* keeps EO assumptions explicit (roles, wavelengths, sensor type)

## Data sources

* **From openEO backend metadata**: bands, dimensions, extents, CRS, collection properties
* **From local script catalog (`INDEX_DEFS`)**: index definitions, required roles, sensor type, wavelength constraints

> `INDEX_DEFS` is local (not downloaded from openEO).

## Output

`collections_kg.json` includes:

* backend endpoint
* `indices_catalog`
* per-collection profiles with:

  * `bands`
  * `logical_roles`
  * `sensor_type`
  * `supported_indices`
  * extents / CRS / metadata

## Requirements

* Python 3.10+
* `openeo` Python client

```bash id="uhwtnh"
pip install openeo
python collections_kg.py
```

## Typical use

Used as a preprocessing step for systems like:

**NL request → DSL → KG grounding (`collections_kg.json`) → deterministic openEO process graph**

---
