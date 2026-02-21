
# README — openEO GeoAgent Prototype for Task Automation (Grounded NL → Process Graph)

**Natural Language → Intent DSL → Knowledge-Graph Grounding → Deterministic Compilation to openEO Process Graphs**
This repository contains a **GeoAgent prototype** that translates high-level Earth Observation (EO) requests into **openEO process graphs**. The emphasis is on **practical automation with guardrails**: the LLM captures user intent, while a **Knowledge Graph (KG)** and a **strict compiler** enforce collection-specific correctness (band mapping, supported indices, cloud metadata rules) before any process graph is produced.

> **Positioning note:** This is intentionally presented as a **prototype**—a solid, working foundation that demonstrates an effective architecture for LLM-assisted openEO workflows, with clear constraints and safety-by-construction design choices. This prototype was developed by the author during a scientific traineeship at the JRC in Ispra, in personal free time. It is an independent personal work and does not necessarily reflect the views, positions, or official policies of the author’s team/group, the JRC, the European Commission, or any affiliated institution.

---

## 1. What this project does

The GeoAgent takes:

* **A natural-language request** (e.g., “Compute NDVI monthly mean with <20% cloud cover and export as GeoTIFF”)
* **An AOI** (GeoJSON Feature/FeatureCollection or bbox)
* Optionally a default **collection ID** (e.g., `SENTINEL2_L2A`)

…and returns:

* A valid **openEO `process_graph` JSON**, generated via openEO **ProcessBuilder** primitives
* Optionally, the intermediate **DSL** representation (useful for inspection and debugging)

You can then submit the process graph to an openEO backend (e.g., Copernicus Data Space) as a job.

---

## 2. Why it’s a “GeoAgent” (and why it’s a prototype)

This is a **GeoAgent** because it performs a complete “task-to-execution” transformation:

* interprets a user goal (LLM),
* grounds that goal in EO domain constraints (KG),
* compiles into an executable artifact (openEO process graph),
* optionally submits it to a backend (job).

It is presented as a **prototype** because it is designed to be:

* an **architecture reference** you can evolve,
* a **practical demonstrator** of how to combine LLM intent extraction + deterministic EO compilation,
* a system with **explicit boundaries** that prioritize predictable outcomes in an EO context.

This is not “LLM free-form graph generation”: the project is built to *avoid* that.

---

## 3. Design philosophy: LLM for intent, KG + compiler for correctness

EO workflows have strict, collection-specific semantics:

* band identifiers differ across collections,
* some collections support only a subset of indices,
* cloud metadata uses different STAC properties,
* classification bands (SCL) may or may not exist,
* some backends have constraints like “load one band at a time”.

To handle this reliably, the pipeline is intentionally split:

### 3.1 LLM: Intent extraction (not EO truth)

The LLM produces a structured **DSLRequest** that captures:

* “what to compute” (indices / band-math products)
* “how to post-process” (aggregation, temporal grouping)
* “quality constraints” (cloud threshold, masking strategy)
* “packaging preferences” (multi-asset vs multi-band)
* “formats” (GTiff/NetCDF/PNG/JSON)

**It does not decide**:

* which physical bands correspond to `RED`, `NIR`, `SWIR1`, etc.
* whether an index is supported by the chosen collection
* which cloud-cover STAC property is valid

### 3.2 Knowledge Graph: EO grounding

A local **collections knowledge graph** (`collections_kg.json`) provides authoritative mapping:

* `logical_roles` → physical band ids
* `classification_roles` (e.g., `SCL`)
* `supported_indices`
* `cloud_cover_property`
* collection quirks (including “single-band-only” hints)
* optional `indices_catalog` for precomputed products

### 3.3 Compiler: deterministic openEO graph construction

Once the DSL is validated and grounded, the compiler constructs the process graph using ProcessBuilder:

* `load_collection`, `filter_bands`
* `reduce_dimension`, `aggregate_temporal_period`, `aggregate_spatial`
* `mask`, `mask_polygon`, `resample_spatial`
* `merge_cubes`, `add_dimension`
* `save_result`, `array_create`

The output is not “hand-written JSON”: it’s produced through the openEO builder primitives.

---

## 4. High-level architecture

### Mode A — Agent mode (default): **NL → DSL → KG → Process Graph**

1. LLM returns JSON (DSLRequest or `{dsl, aoi}`)
2. DSL normalization (robust temporal parsing, synonyms, defaults)
3. DSL validation (Pydantic v2)
4. Autofill from raw text (formats, cloud settings, precomputed hints)
5. KG grounding and constraints checking
6. Process graph compilation (ProcessBuilder)

### Mode B — DIRECT PG mode (expert escape hatch)

If the instruction looks like a procedural specification (“BUILD A PROCESS GRAPH…”, includes `process_graph`, etc.), the system can:

* instruct the LLM to emit the process graph JSON directly
* normalize the `result: true` flag
* return `{ "process_graph": ... }`

This is intended for advanced workflows (including multi-collection merges) that go beyond the DSL’s “one collection per graph” rule.

---

## 5. Core features

### 5.1 Multi-product process graphs

A single request can produce:

* multiple predefined indices (NDVI, NBR, NDWI, etc.)
* multiple custom derived products via band-math
* optional per-product aggregations

### 5.2 Predefined indices supported

The DSL supports these predefined products:

* Base: `NDVI`, `NDWI`, `NDSI`, `EVI`, `RGB`
* Vegetation/soil: `SAVI`, `OSAVI`, `MSAVI`, `MSAVI2`, `GNDVI`, `NDRE`, `ARVI`
* Water: `MNDWI`, `AWEI_SH`, `AWEI_NSH`
* Fire: `NBR`, `NBR2`, `BAI`
* Urban/soil: `NDBI`, `BSI`
* SAR: `VV_VH_RATIO`, `RVI`

> Availability is collection-dependent and enforced by the KG (`supported_indices`).

### 5.3 Generic band math (safe arithmetic subset)

Custom band-math tasks accept expressions like:

* `(NIR - SWIR1) / (NIR + SWIR1)`
* `(RED + GREEN + BLUE) / 3`

Enforced rules:

* only arithmetic: `+ - * / **`, parentheses, numeric constants
* no function calls / attributes / indexing
* variables must be logical roles known to the KG (e.g., `RED`, `NIR`, `SWIR1`, `VV`, `VH`)

This enables flexibility without opening up arbitrary code paths.

### 5.4 Cloud controls (metadata + optional SCL mask)

**Cloud cover metadata filter**

* If `cloud.max` is set (fraction `[0..1]`), the compiler injects a `properties` filter at `load_collection`
* The metadata property is resolved in this order:

  1. explicit override from user text → `dsl.cloud.property`
  2. fallback to `KG.cloud_cover_property`

**SCL masking (`basic_s2_scl`)**

* Loads SCL as a separate cube when available
* Resamples SCL using `nearest` when spatial resampling is requested
* Masks clouds based on specific SCL classes (3, 8, 9, 10)

### 5.5 AOI: flexible parsing + CRS preservation

AOI input supports:

* GeoJSON Feature / FeatureCollection / geometry
* bbox arrays `[west, south, east, north]`
* embedded JSON blocks inside text
* python-dict-like structures (via `ast.literal_eval`)

Spatial handling:

* bbox is used for `spatial_extent`
* CRS is extracted from AOI properties (no hardcoded EPSG:4326)
* `mask_polygon` is applied unless the AOI is exactly the bbox rectangle (performance optimization)

### 5.6 Temporal: normalization + multi-range merge

The system accepts temporal input in multiple shapes:

* DSL `{start, end}`
* openEO array form `["start","end"]`
* multi-range arrays `[[...],[...]]`

Overlapping ranges are merged and normalized for `load_collection`.

### 5.7 Aggregations and analytics

Per task aggregations supported:

* `mean`, `median`, `max`, `min`
* `monthly_pct_area` (fraction of AOI where index > threshold per period)
* `trend_yearly` (linear trend over yearly means)

Additionally, a **global temporal aggregation** can be applied on final outputs:

* `temporal_aggregation = { period: dekad|month|year, reducer: mean|median|max|min }`
* applied only when a time dimension still exists (guarded)

### 5.8 Output packaging

Two output strategies:

**`multi_asset` (default)**

* one `save_result` per task
* multiple outputs are grouped with `array_create`

**`multi_band`**

* stacks multiple single-band outputs as bands into one cube
* rejects RGB in this mode
* rejects JSON/PNG formats for this mode

### 5.9 Format selection (smart defaults + text overrides)

* detects format intent from user text (“NDVI in NetCDF, RGB in PNG”)
* supports global default format and per-task overrides
* chooses sensible defaults:

  * RGB → PNG
  * analytics like trend/monthly area → JSON
  * raster indices → GTiff

---

## 6. Knowledge Graph (`collections_kg.json`)

### 6.1 What it contains

The KG is a JSON document with:

* `collections[]`: per-collection profiles
* `indices_catalog`: optional lookup table for precomputed index sources

Per collection profile the agent uses:

* `collection_id`
* `bands`: list of band objects with `band_id`
* `logical_roles`: mapping like `{ "RED": "B04", "NIR": "B08", ... }`
* `classification_roles`: mapping like `{ "SCL": "SCL" }`
* `supported_indices`: list of allowed indices
* `cloud_cover_property`: e.g. `"eo:cloud_cover"`
* `description`: optional hints (e.g., “only supports loading one band at a time”)
* `sensor_type`: optional (`sar` triggers role fallbacks for RGB-like requests)

### 6.2 Why the KG is critical

The KG ensures that:

* the same logical request is compiled consistently across collections,
* unsupported products are rejected early with clear errors,
* cloud masking and property filters use valid collection metadata,
* the agent stays grounded even when the LLM output is incomplete.

---

## 7. Public interfaces (LangChain tools)

### 7.1 `build_process_graph_from_instruction` (main)

```python
build_process_graph_from_instruction(
    instruction: str,
    aoi_feature_collection_json: str,
    default_collection: str = "SENTINEL2_L2A",
    return_dsl: bool = False,
) -> str
```

Returns:

* process graph JSON (stringified)
* optionally also the DSL for inspection (`return_dsl=True`)

Key behaviors:

* AOI override: if LLM returns top-level `"aoi"`, it overrides the default AOI
* routing: chooses Agent mode vs DIRECT PG mode based on instruction pattern
* autofill: extracts formats, cloud parameters, and precomputed hints from raw text

### 7.2 `submit_process_graph` (execution helper)

```python
submit_process_graph(
    process_graph_json: str,
    endpoint: str = "https://openeo.dataspace.copernicus.eu",
) -> str
```

* connects to backend and authenticates via OIDC
* creates and starts an openEO job
* returns `{job_id, status}`

### 7.3 `demo_run_with_your_widgets_textarea`

Notebook convenience wrapper for UI workflows.

---

## 8. Setup

### 8.1 Requirements

* Python 3.10+ (recommended)
* Packages:

  * `openeo`
  * `langchain-core`
  * `langchain-openai`
  * `pydantic>=2`

Plus local modules:

* `eo_expert_ops.py` (expert pipeline primitives)
* `collections_kg.json` (generated by `collections_kg.py`)

### 8.2 API key (Gemini via OpenAI-compatible endpoint)

Key lookup order:

1. `GEMINI_API_KEY` env var
2. `./GEMINI` file
3. `./.keys/GEMINI` file

### 8.3 Optional `.env` file (recommended for local development)

You can store tokens and local configuration in a `.env` file, for example:

```bash
GEMINI_API_KEY=your_gemini_key_here
OPENEO_COLLECTIONS_KG_PATH=collections_kg.json
```

> Note: this script does **not** load `.env` automatically. Make sure environment variables are exported by your shell, notebook environment, or a launcher script (e.g., via `python-dotenv`).

---

## 9. Typical usage

### 9.1 Build a process graph (Agent mode)

* Provide:

  * `instruction` (natural language)
  * `aoi_feature_collection_json` (default AOI)

The tool returns `{"process_graph": {...}}`.

### 9.2 AOI override via instruction

If the user includes a bbox or GeoJSON explicitly, the LLM extracts it to `"aoi"` and the agent uses it instead of the default AOI.

### 9.3 Direct PG mode for expert control

Write a procedural instruction that explicitly references process graph structure. The agent switches to DIRECT PG mode and returns graph JSON directly.

---

## 10. Constraints and intentional guardrails

### 10.1 One collection per graph in Agent mode

In DSL-based agent mode:

* exactly one `collections[0]` is allowed
* simplifies grounding and makes compilation deterministic

### 10.2 Band math is intentionally restricted

The AST-based evaluator:

* allows arithmetic only
* prevents function calls and complex Python constructs
* requires all variables to map to KG roles

### 10.3 `multi_band` is designed for single-band outputs

* RGB is rejected in `multi_band`
* formats incompatible with multi-band raster outputs are rejected

---

## 11. Extending the prototype (roadmap-friendly)

This prototype is structured to evolve cleanly:

1. **KG enrichment**

   * expand collection coverage
   * add scaling/offset and nodata semantics
   * improve role taxonomy

2. **More ExpertOps**

   * best-pixel mosaics
   * change detection templates
   * phenology metrics
   * QA masking strategies

3. **Multi-collection planning**

   * add a planning step that decomposes a request into subgraphs
   * merge products deterministically post-hoc
   * keep the same “LLM intent / KG truth / compiler determinism” split

4. **Graph validation against backend capabilities**

   * optional linting step
   * backend process availability checks
   * clearer error messages with actionable remedies

---

## 12. Summary

This repository demonstrates a clear pattern for **LLM-assisted EO automation**:

* **LLM**: captures intent and structure
* **Knowledge Graph**: provides EO truth and constraints
* **Compiler**: deterministically generates correct openEO process graphs
* **Output**: job-ready process graphs, with flexible multi-product packaging and robust AOI/temporal handling

As a **GeoAgent prototype**, it is already useful for real workflows and provides a strong foundation for expanding toward broader EO task planning and execution.

---
