# README — openEO GeoAgent Prototype (Short Guide)

## Start here

Before using the code, please check the documentation in the **`documentation/`** folder for the full project description, design choices, and detailed module notes (including the Knowledge Graph builder).

---

## Repository structure

```text
repo/
├─ README.md                         # Short project guide (GitHub-friendly)
├─ .gitignore
│
├─ src/
│  └─ openeo_geoagent/
│     ├─ __init__.py                 # Can be empty
│     ├─ openeo_llm.py               # Main GeoAgent module (NL → DSL → KG → openEO PG)
│     ├─ eo_expert_ops.py            # Optional advanced EO operations (expert helpers)
│     └─ collections_kg.py           # KG builder script (builds collections_kg.json)
│
├─ examples/
│  └─ wrapper_notebook.ipynb         # Recommended interactive wrapper notebook
│  
├─ documentation/
│  ├─ README_OPENEO_LLM.md           # Main project documentation (detailed)
│  └─ README_COLLECTIONS_KG.md       # collections_kg module documentation
│
│
└─ .env.example                      # Example env variables (no secrets)

```

> Use the **wrapper notebook in `examples/`** as the recommended entry point for interactive runs and demos.

---

## Overview

This repository contains a **GeoAgent prototype** for Earth Observation task automation with openEO.

It translates:

* a **natural-language EO request**
* an **AOI** (GeoJSON / bbox)

into a deterministic **openEO `process_graph`** using a guarded pipeline:

**NL → DSL → Knowledge Graph grounding → Process Graph compilation**

The design goal is to combine **LLM flexibility** (intent extraction) with **EO correctness** (collection-specific band/index rules).


> **Important (prototype / research note):** This repository is a **work in progress** and is primarily intended to test and compare different approaches for grounded EO task automation (e.g., Knowledge Graph grounding, LangChain-based orchestration, and deterministic compilation strategies).  
> Generated process graphs may be incomplete, suboptimal, or incorrect in some cases and should be reviewed/validated before operational use.
---



## Core components

### 1) `openeo_llm.py` (main GeoAgent module)

Main module that:

* parses user intent into a DSL (via LLM)
* validates and normalizes the DSL
* grounds the request using `collections_kg.json`
* compiles a deterministic openEO process graph
* optionally submits jobs to an openEO backend

---

### 2) `collections_kg.py` (Knowledge Graph builder)

Utility script that builds `collections_kg.json` from the metadata exposed by an openEO backend (e.g. CDSE).

It automatically derives:

* band metadata
* logical roles (`RED`, `NIR`, `SWIR1`, `VV`, `VH`, `SCL`, ...)
* sensor type (`optical`, `sar`, `other`)
* supported indices per collection (e.g. `NDVI`, `NBR`, `NDWI`, `RVI`)

This file is the **grounding layer** used by `openeo_llm.py`.

---

### 3) `eo_expert_ops.py` (optional advanced EO helpers)

Contains advanced EO operations (e.g. temporal mosaicking, filtering, morphology, UDF hooks) used by the main pipeline when requested.

---

## Why `collections_kg.json` matters

The Knowledge Graph avoids fragile “guess the band names” logic and enables deterministic validation of requests such as:

* Which band is `RED` in this collection?
* Does this collection support `NDSI`?
* Does it have `SCL` for cloud masking?
* Is it optical or SAR?

This is especially important in **LLM-assisted workflows**, where the model should express **intent**, not invent collection-specific EO details.

---

## Quick setup

### Requirements

* Python 3.10+
* `openeo`
* `langchain-core`
* `langchain-openai`
* `pydantic>=2`

Install (minimal):

```bash
pip install openeo langchain-core langchain-openai "pydantic>=2"
```

---

## API key / environment variables

`openeo_llm.py` looks for the Gemini key in this order:

1. `GEMINI_API_KEY` environment variable
2. `./GEMINI` file
3. `./.keys/GEMINI` file

Optional environment variable:

* `OPENEO_COLLECTIONS_KG_PATH` → custom path to `collections_kg.json`

Example `.env` (optional, for local development):

```bash
GEMINI_API_KEY=your_gemini_key_here
OPENEO_COLLECTIONS_KG_PATH=collections_kg.json
```

> Note: `.env` is **not loaded automatically** unless you use a launcher or `python-dotenv`.

---

## How to use the repository (recommended workflow)

### Step 1 — Build the collection Knowledge Graph

Generate `collections_kg.json` from openEO backend metadata:

```bash
python collections_kg.py
```

---

### Step 2 — Use the wrapper notebook in `examples/`

Use the **wrapper notebook in the `examples/` folder** to:

* provide an AOI
* write a natural-language request
* inspect the intermediate DSL (optional)
* generate the openEO process graph
* optionally submit/run jobs and download results

This is the recommended entry point for interactive usage and demonstrations in this repository.

---

### Step 3 — (Optional) Submit the process graph to openEO

If needed, use the helper functions provided in `openeo_llm.py` (or the notebook wrapper) to submit the generated process graph to an openEO backend and retrieve results.

---

## Typical execution order

1. **Read the docs in `documentation/`**
2. **Generate or refresh** `collections_kg.json`
3. **Run the wrapper notebook in `examples/`**
4. **Build process graphs** from natural-language EO requests
5. **Submit jobs** and **download outputs** (optional)

---

## Why this architecture is useful

* **LLM for intent only** (not EO truth)
* **Knowledge Graph for collection-specific grounding**
* **Deterministic compiler for process graph generation**
* **Reusable KG artifact** (`collections_kg.json`) across notebooks and tools

This makes the system significantly more robust than free-form LLM generation of openEO graphs.

---

## Notes

* `collections_kg.py` uses **openEO backend metadata** (machine-readable collection metadata)
* `INDEX_DEFS` is a **local catalog in the script** and is used to compute index compatibility
* Role assignment is heuristic-based and can be refined for unusual collections
