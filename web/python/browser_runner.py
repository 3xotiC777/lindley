"""Thin browser adapter for the unchanged Lindley selection engine.

The deployment workflow places the production ``config``, ``ingest``,
``selector_engine`` and ``exporter`` modules next to this file.  This adapter
only maps browser-provided paths/options to the same calls used by the desktop
application.
"""

from pathlib import Path

import config
import exporter
import ingest
import selector_engine


OUTPUT_NAME = "SELECCION_MUESTRA_LINDLEY_RESULTADO.xlsx"
SUPERVISION_NAME = "ARCHIVO_DEF_SUP_LINDLEY_RESULTADO.xlsx"
VALID_SCOPES = {"OFF"}


def run_browser_selection(
    *,
    preselection_path,
    previous_path,
    lima_path,
    output_dir,
    total_target,
    increase_scope,
):
    """Run the production selection flow with files stored in Pyodide's FS."""
    total_target = int(total_target)
    if total_target <= 0:
        raise ValueError("La muestra total debe ser un número entero mayor que cero.")

    increase_scope = str(increase_scope).strip().upper()
    if increase_scope not in VALID_SCOPES:
        raise ValueError("El aumento debe aplicarse a OFF porque las cuotas ON son fijas.")

    preselection_path = Path(preselection_path)
    previous_path = Path(previous_path)
    lima_path = Path(lima_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, path in (
        ("preselección", preselection_path),
        ("selección anterior", previous_path),
        ("cuotas Lima", lima_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de {label}.")

    config.WORK_PRESELECCION_PATH = preselection_path
    config.WORK_PREV_SELECCION_PATH = previous_path
    config.LIMA_QUOTAS_PATH = lima_path
    config.OUTPUT_SELECCION_PATH = output_dir / OUTPUT_NAME
    config.OUTPUT_DEF_SUP_PATH = output_dir / SUPERVISION_NAME

    eligible, non_eligible, historical = ingest.prepare_merged_datasets()
    final, controls = selector_engine.run_selection_process(
        eligible,
        historical,
        total_target=total_target,
        increase_scope=increase_scope,
    )
    exporter.export_selection_workbook(
        final,
        non_eligible,
        controls,
        config.OUTPUT_SELECCION_PATH,
    )
    exporter.export_def_sup_workbook(final, config.OUTPUT_DEF_SUP_PATH)

    selected = final[config.SELECTION_COL].eq("T")
    selected_on = selected & final["_CANAL_SELECCION"].eq("ON")
    actual_total = int(selected.sum())
    actual_on = int(selected_on.sum())
    actual_off = actual_total - actual_on
    control_states = controls.get("Estado")
    controls_review = (
        int(control_states.astype(str).str.upper().eq("REVISAR").sum())
        if control_states is not None
        else 0
    )

    if not config.OUTPUT_SELECCION_PATH.is_file() or not config.OUTPUT_DEF_SUP_PATH.is_file():
        raise RuntimeError("El cálculo terminó, pero no fue posible crear los dos archivos de salida.")

    return {
        "requestedTotal": total_target,
        "actualTotal": actual_total,
        "actualOn": actual_on,
        "actualOff": actual_off,
        "increaseScope": increase_scope,
        "controlsReview": controls_review,
        "selectionFile": config.OUTPUT_SELECCION_PATH.name,
        "supervisionFile": config.OUTPUT_DEF_SUP_PATH.name,
    }
