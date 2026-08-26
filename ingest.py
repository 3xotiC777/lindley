import re
import unicodedata
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pyxlsb

import config


def _key(value) -> str:
    """Normalize labels so selection headers work despite accent encoding."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _find_column(columns, expected: str) -> str:
    expected_key = _key(expected)
    for column in columns:
        if _key(column) == expected_key:
            return column
    raise KeyError(f"No se encontró la columna requerida: {expected}")


def _find_selection_column(columns) -> str:
    for column in columns:
        if _key(column).startswith("SELECC"):
            return column
    raise KeyError("No se encontró la columna de selección en el archivo del mes anterior.")


def _clean_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.upper()
    return df


def load_preselection(file_path):
    """Load the current BD sheet and preserve the source columns."""
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    try:
        sheet_name = next(
            (
                sheet for sheet in wb.sheetnames
                if sheet.strip().upper() == "BD" or "SELECCION_MUESTRA" in sheet.upper()
            ),
            None,
        )
        if sheet_name is None:
            raise ValueError("La preselección no contiene una hoja BD o SELECCION_MUESTRA.")
        ws = wb[sheet_name]
        data = list(ws.iter_rows(values_only=True))
        extra_data = None
        if sheet_name.strip().upper() != "BD" and "NO ELEGIBLES" in wb.sheetnames:
            extra_data = list(wb["NO ELEGIBLES"].iter_rows(values_only=True))
    finally:
        wb.close()

    header = [str(cell).strip() if cell is not None else "" for cell in data[0]]
    df = pd.DataFrame(data[1:], columns=header)
    if extra_data:
        extra_header = [str(cell).strip() if cell is not None else "" for cell in extra_data[0]]
        df = pd.concat([df, pd.DataFrame(extra_data[1:], columns=extra_header)], ignore_index=True)
    # A previous selection workbook can be reused as the preselection source.
    # Remove its calculated fields before merging the current historical data.
    df = df.drop(
        columns=[
            "SEL_MES_ANT", "VISITAS_ANT", "RUTA_MES_ANT", "Esparta_Flag",
            "PV_Flag", "CR_Flag", config.SELECTION_COL, "_CANAL_SELECCION",
            "SEL_OFF", "SEL_ON", "SEL_TOTAL", "VISITAS",
        ],
        errors="ignore",
    )
    df["CODIGO"] = pd.to_numeric(df["CODIGO"], errors="coerce").fillna(0).astype(np.int64)
    return _clean_categories(
        df,
        [
            "Ciudad", "NOMBRE", "Subcanal2", "ESTRATEGICA", "Esparta",
            "Programa de Valor", "Coraje rojo", "ESTADO", "RUT.COM", "LOC.COM",
            "CANAL", "ESPECIALIZADA",
        ],
    )


def load_previous_selection(file_path):
    """Load both merge fields and the historical T quotas by city/segment/subchannel."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    rows = []

    if suffix == ".xlsb":
        with pyxlsb.open_workbook(file_path) as wb:
            sheet_name = next(
                (sheet for sheet in wb.sheets if "SELECCION_MUESTRA" in sheet.upper() or sheet.upper() == "BD"),
                None,
            )
            if sheet_name is None:
                raise ValueError("El archivo del mes anterior no contiene una hoja BD o SELECCION_MUESTRA.")
            with wb.get_sheet(sheet_name) as sheet:
                for row in sheet.rows():
                    rows.append([cell.v for cell in row])
    elif suffix == ".xlsx":
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        try:
            sheet_name = next(
                (sheet for sheet in wb.sheetnames if "SELECCION_MUESTRA" in sheet.upper() or sheet.upper() == "BD"),
                None,
            )
            if sheet_name is None:
                raise ValueError("El archivo del mes anterior no contiene una hoja BD o SELECCION_MUESTRA.")
            rows = [list(row) for row in wb[sheet_name].iter_rows(values_only=True)]
        finally:
            wb.close()
    else:
        raise ValueError("La selección del mes anterior debe ser un archivo .xlsx o .xlsb.")

    if not rows:
        raise ValueError("La hoja de selección del mes anterior está vacía.")

    header = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    previous = pd.DataFrame(rows[1:], columns=header)
    codigo_col = _find_column(previous.columns, "CODIGO")
    selection_col = _find_selection_column(previous.columns)
    visitas_col = _find_column(previous.columns, "VISITAS")
    ruta_col = _find_column(previous.columns, "RUT.COM")

    previous["CODIGO"] = pd.to_numeric(previous[codigo_col], errors="coerce").fillna(0).astype(np.int64)
    historical = previous.copy()
    historical = _clean_categories(historical, ["NOMBRE", "ESTRATEGICA", "Subcanal2", "LOC.COM"])
    historical["ES_TITULAR_HISTORICO"] = historical[selection_col].astype(str).str.strip().str.upper().eq("T")
    historical = historical.drop_duplicates(subset=["CODIGO"])

    merged_fields = historical[["CODIGO", selection_col, visitas_col, ruta_col]].copy()
    merged_fields.columns = ["CODIGO", "SEL_MES_ANT", "VISITAS_ANT", "RUTA_MES_ANT"]
    return merged_fields, historical


def prepare_merged_datasets():
    """Merge sources, split eligibility, and return the historical quota source."""
    print("Loading August preselection dataset...")
    current = load_preselection(config.WORK_PRESELECCION_PATH)
    print("Loading July previous selection dataset...")
    previous_for_merge, historical = load_previous_selection(config.WORK_PREV_SELECCION_PATH)

    merged = pd.merge(current, previous_for_merge, on="CODIGO", how="left")
    merged["SEL_MES_ANT"] = merged["SEL_MES_ANT"].fillna("NO")
    merged["VISITAS_ANT"] = pd.to_numeric(merged["VISITAS_ANT"], errors="coerce").fillna(0)
    merged["RUTA_MES_ANT"] = merged["RUTA_MES_ANT"].fillna(merged["RUT.COM"])

    for source, flag in [("Esparta", "Esparta_Flag"), ("Programa de Valor", "PV_Flag"), ("Coraje rojo", "CR_Flag")]:
        merged[flag] = merged[source].astype(str).str.strip().str.upper().isin({"1", "1.0", "SI", "TRUE"}).astype(int)
    merged["Ventas"] = pd.to_numeric(merged["Ventas"], errors="coerce").fillna(0)

    is_elegible = merged["ESTADO"].eq("ELEGIBLE")
    elegible = merged[is_elegible].copy()
    no_elegible = merged[~is_elegible].copy()
    print(f"Total Universe: {len(merged)}")
    print(f"ELEGIBLE (BD Universe): {len(elegible)}")
    print(f"NO ELEGIBLE Universe: {len(no_elegible)}")
    return elegible, no_elegible, historical
