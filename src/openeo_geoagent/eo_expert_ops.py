# eo_expert_ops.py
# ==========================================================
# Helper per operazioni "da esperto" su DataCube openEO:
#   - specifiche DSL (TemporalMosaicSpec, SpatialFilterSpec, MorphologySpec,
#     TemporalFilterSpec, UdfSpec, ExpertOps)
#   - funzioni helper per applicarle a un DataCube:
#       * get_time_dimension_name
#       * apply_expert_ops
#
# Questo file NON dipende dal DSL principale:
# il DSL principale importa ExpertOps + apply_expert_ops da qui.
# ==========================================================

from __future__ import annotations

from typing import Optional, List, Literal

from pydantic import BaseModel, Field


from typing import Optional, List, Literal, TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    
    from openeo.rest.datacube import DataCube as _OpenEODataCube
    DataCube: TypeAlias = _OpenEODataCube
else:
    # A runtime, se openeo non è installato o non serve il tipo reale,
    # usiamo un alias ad Any così non esplode niente.
    DataCube: TypeAlias = Any


# ==========================================================
# 1) Modelli DSL per operazioni esperte
# ==========================================================

class TemporalMosaicSpec(BaseModel):
    """
    Mosaicking temporale avanzato.

    Esempi NL:
      - "mosaico mensile prendendo il massimo"
      - "ultimo valore disponibile per ogni mese"
      - "media su tutto il periodo"

    Il mapping nel DSL tipico:
      {
        "method": "max" | "min" | "mean" | "median" | "most_recent",
        "period": "month" | "year" | null
      }
    """
    method: Literal["most_recent", "mean", "max", "min", "median"] = "most_recent"
    period: Optional[str] = Field(
        None,
        description="e.g. 'month', 'year'; None = tutto l'intervallo così com'è",
    )


class SpatialFilterSpec(BaseModel):
    """
    Filtri spaziali (kernel o neighborhood).

    type = "kernel"
      - kernel: matrice numerica
      - factor: opzionale (se None → 1/somma(kernel))

    type = "neighborhood"
      - size: lato finestra quadrata (3 → 3x3)
      - reducer: 'mean', 'median', 'max', 'min'
    """
    type: Literal["kernel", "neighborhood"] = "kernel"

    # per type == "kernel"
    kernel: Optional[List[List[float]]] = Field(
        None,
        description="Matrice del kernel, es [[1,2,1],[2,4,2],[1,2,1]]",
    )
    factor: Optional[float] = Field(
        None,
        description="Fattore di normalizzazione; se None e kernel presente → 1/somma(kernel)",
    )

    # per type == "neighborhood"
    size: Optional[int] = Field(
        None,
        description="Lato della finestra quadrata (es. 3 → 3x3, 5 → 5x5)",
    )
    reducer: Optional[Literal["mean", "median", "max", "min"]] = None


class MorphologySpec(BaseModel):
    """
    Operazioni morfologiche su maschere (tipicamente mask binarie).

    operation:
      - 'dilate' → dilatazione
      - 'erode'  → erosione
      - 'open'   → apertura (erode poi dilate, approssimata)
      - 'close'  → chiusura (dilate poi erode, approssimata)

    threshold:
      - opzionale, per derivare la mask dall'indice continuo (es NDVI > 0.3)
    """
    operation: Literal["dilate", "erode", "open", "close"] = "dilate"
    iterations: int = 1
    threshold: Optional[float] = None


class TemporalFilterSpec(BaseModel):
    """
    Filtri temporali / rimozione outlier.

    type:
      - 'rolling_mean'      → media mobile (approssimata, vedi commento in apply_expert_ops)
      - 'rolling_median'    → mediana mobile
      - 'remove_outliers'   → rimozione outlier via z-score

    ATTENZIONE: l'implementazione concreta di rolling/outlier dipende
    dai processi disponibili nel backend (apply_dimension + funzioni).
    """
    type: Literal["rolling_mean", "rolling_median", "remove_outliers"] = "rolling_mean"
    window: int = 3
    zscore: Optional[float] = Field(
        None,
        description="Soglia in sigma per type == 'remove_outliers' (default 3.0)",
    )


class UdfSpec(BaseModel):
    """
    UDF generiche da inserire nella pipeline.

    language:
      - 'python', 'r', 'javascript'

    source:
      - codice sorgente UDF

    runtime:
      - nome runtime lato backend (es. 'Python 3.11') se necessario

    mode:
      - 'apply'             → run_udf su tutto il cube
      - 'apply_dimension'   → run_udf lungo una dimensione (tipicamente temporale)
      - 'apply_neighborhood'→ run_udf su vicinato (spaziale/temporale)
    """
    language: Literal["python", "r", "javascript"]
    source: str = Field(
        ...,
        description="Codice sorgente della UDF",
    )
    runtime: Optional[str] = Field(
        None,
        description="Nome runtime lato backend, es. 'Python 3.11'",
    )
    mode: Literal["apply", "apply_dimension", "apply_neighborhood"] = "apply"


class ExpertOps(BaseModel):
    """
    Blocco 'esperto' opzionale per ogni indice / band-math.

    Il DSL principale può includere, per ciascun IndexTask/BandMathTask,
    un campo:

      \"expert\": {
        \"temporal_mosaic\": {...},
        \"spatial_filter\": {...},
        \"morphology\": {...},
        \"temporal_filter\": {...},
        \"udf\": {...}
      }

    Tutti i campi sono opzionali e, se presenti, vengono applicati
    in questo ordine:

      1) temporal_mosaic
      2) spatial_filter
      3) morphology
      4) temporal_filter
      5) udf
    """
    temporal_mosaic: Optional[TemporalMosaicSpec] = None
    spatial_filter: Optional[SpatialFilterSpec] = None
    morphology: Optional[MorphologySpec] = None
    temporal_filter: Optional[TemporalFilterSpec] = None
    udf: Optional[UdfSpec] = None


# ==========================================================
# 2) Helper su DataCube
# ==========================================================

def get_time_dimension_name(cube: DataCube) -> str:
    """
    Prova a individuare il nome della dimensione temporale del cube,
    senza assumere che sia sempre 't'.

    Strategia:
      1) guarda in cube.metadata['dimensions'] se esiste
      2) se c'è 't', usa 't'
      3) se c'è 'time', usa 'time'
      4) altrimenti, prima dimensione con type=='temporal'
      5) fallback finale: 't'
    """
    dims_meta = {}
    try:
        meta = getattr(cube, "metadata", {}) or {}
        dims_meta = meta.get("dimensions", {}) or {}
    except Exception:
        dims_meta = {}

    if "t" in dims_meta:
        return "t"
    if "time" in dims_meta:
        return "time"

    for name, d in dims_meta.items():
        if isinstance(d, dict) and d.get("type") == "temporal":
            return name

    return "t"


def apply_expert_ops(
    cube: DataCube,
    expert: Optional[ExpertOps],
) -> DataCube:
    """
    Applica in cascata le operazioni 'esperte' se presenti nel task:

      1) temporal_mosaic
      2) spatial_filter
      3) morphology
      4) temporal_filter
      5) udf

    Se expert è None o tutti i campi sono None, il cube viene restituito invariato.
    """
    if expert is None:
        return cube

    # ----------------- 1) mosaico temporale custom -----------------
    if expert.temporal_mosaic is not None:
        spec = expert.temporal_mosaic
        time_dim = get_time_dimension_name(cube)

        def _mk_reducer(agg_name: str):
            def _reducer(data, context=None, _a=agg_name):
                # data: array lungo time_dim per ogni pixel
                return getattr(data, _a)()
            return _reducer

        if spec.method == "most_recent":
            # euristica: 'most_recent' come max rispetto al tempo
            reducer = _mk_reducer("max")
        else:
            reducer = _mk_reducer(spec.method)

        if spec.period:
            cube = cube.aggregate_temporal_period(
                period=spec.period,
                reducer=reducer,
            )
        else:
            cube = cube.aggregate_temporal(reducer=reducer)

    # ----------------- 2) filtri spaziali --------------------------
    if expert.spatial_filter is not None:
        sf = expert.spatial_filter

        if sf.type == "kernel":
            if not sf.kernel:
                raise ValueError("SpatialFilterSpec.type='kernel' but 'kernel' is empty.")
            if sf.factor is None:
                s = sum(sum(row) for row in sf.kernel)
                factor = 1.0 / s if s != 0 else 1.0
            else:
                factor = sf.factor
            cube = cube.apply_kernel(kernel=sf.kernel, factor=factor)

        elif sf.type == "neighborhood":
            if sf.size is None or sf.size <= 0:
                raise ValueError("SpatialFilterSpec.type='neighborhood' requires a positive 'size'.")
            if not sf.reducer:
                raise ValueError("SpatialFilterSpec.type='neighborhood' requires 'reducer'.")

            size = sf.size

            def _neigh_reducer(data, context=None, _r=sf.reducer):
                return getattr(data, _r)()

            cube = cube.apply_neighborhood(
                size=[size, size],
                reducer=_neigh_reducer,
            )

    # ----------------- 3) morfologia su maschere -------------------
    if expert.morphology is not None:
        m = expert.morphology

        def _to_mask(data, context=None, _thr=m.threshold):
            x = data
            if _thr is not None:
                x = x > _thr
            return x

        mask_cube = cube.apply(_to_mask)

        def _morph_reducer(data, context=None, _op=m.operation):
            if _op == "dilate":
                # max su vicinato → dilatazione
                return data.max()
            elif _op == "erode":
                # min → erosione
                return data.min()
            elif _op == "open":
                # erosione poi dilatazione (approssimata: min().max())
                return data.min().max()
            elif _op == "close":
                # dilatazione poi erosione (approssimata: max().min())
                return data.max().min()
            else:
                return data

        iterations = max(1, m.iterations)
        for _ in range(iterations):
            mask_cube = mask_cube.apply_neighborhood(
                size=[3, 3],
                reducer=_morph_reducer,
            )

        # riapplichiamo la mask morfologica all'indice continuo
        cube = cube.mask(mask_cube)

    # ----------------- 4) filtri temporali -------------------------
    if expert.temporal_filter is not None:
        tf = expert.temporal_filter
        time_dim = get_time_dimension_name(cube)
        win = max(1, tf.window)

        # ⚠️ NOTA IMPORTANTE:
        # Le implementazioni sotto assumono la disponibilità di processi
        # tipo "apply_dimension" con supporto a funzioni come mean()/median().
        # Adatta al tuo backend se servono process_graph più espliciti.

        if tf.type in {"rolling_mean", "rolling_median"}:
            # "rolling" qui è schematizzato: spesso lo si implementa meglio via UDF.
            reducer_name = "mean" if tf.type == "rolling_mean" else "median"

            def _rolling(data, context=None, _w=win, _r=reducer_name):
                # pseudo-codice: sostituisci con implementazione reale
                # (es. UDF o process_graph più dettagliato)
                return getattr(data, _r)()
            cube = cube.apply_dimension(dimension=time_dim, process=_rolling)

        elif tf.type == "remove_outliers":
            zthr = tf.zscore or 3.0

            def _remove_outliers(data, context=None, _z=zthr):
                mean = data.mean()
                std = data.std()
                z = (data - mean) / std
                # pseudo-codice: il "where" qui è concettuale
                return data.where(z.abs() <= _z, mean)

            cube = cube.apply_dimension(dimension=time_dim, process=_remove_outliers)

    # ----------------- 5) UDF generica -----------------------------
    if expert.udf is not None:
        u = expert.udf
        mode = u.mode
        runtime = u.runtime or u.language

        if mode == "apply":
            cube = cube.run_udf(
                udf=u.source,
                runtime=runtime,
            )
        elif mode == "apply_dimension":
            time_dim = get_time_dimension_name(cube)
            cube = cube.run_udf(
                udf=u.source,
                runtime=runtime,
                context={"dimension": time_dim},
            )
        elif mode == "apply_neighborhood":
            cube = cube.run_udf(
                udf=u.source,
                runtime=runtime,
                context={"mode": "neighborhood"},
            )

    return cube
