from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MONTH_NAME = "AGOSTO_2026"
SELECTION_COL = "SELECCION"

# Prefer local source copies, but fall back to the files kept beside the app.
_LOCAL_PRESELECTION = PROJECT_DIR / "Universo_Agosto_2026_Preseleccion_V1.xlsx"
_FALLBACK_PRESELECTION = PROJECT_DIR.parent / "SELECCION_MUESTRA_LINDLEY_AGOSTO_V1.xlsx"
WORK_PRESELECCION_PATH = (
    _LOCAL_PRESELECTION if _LOCAL_PRESELECTION.exists() else _FALLBACK_PRESELECTION
)
_LOCAL_PREVIOUS = PROJECT_DIR / "SELECCION_MUESTRA_LINDLEY_JULIO_V1.xlsb"
_FALLBACK_PREVIOUS = PROJECT_DIR.parent / "SELECCION_MUESTRA_LINDLEY_JULIO_V1.xlsb"
WORK_PREV_SELECCION_PATH = _LOCAL_PREVIOUS if _LOCAL_PREVIOUS.exists() else _FALLBACK_PREVIOUS
LIMA_QUOTAS_PATH = PROJECT_DIR.parent.parent / "cuotas lima.xlsx"

# No se sobrescribe el archivo que el usuario puede tener abierto en Excel.
OUTPUT_SELECCION_PATH = PROJECT_DIR / "SELECCION_MUESTRA_LINDLEY_AGOSTO_V7_ESTRATEGIA_1PP.xlsx"
OUTPUT_DEF_SUP_PATH = PROJECT_DIR / "ARCHIVO_DEF_SUP_AGOSTO_V7_ESTRATEGIA_1PP.xlsx"

MIN_PDVS_PER_ROUTE = 15
ON_SPECIALIZED_TARGET = 0.90
ON_SPECIALIZED_ROUTE_MIN = 30
ON_SPECIALIZED_ROUTE_ACTIVE_MIN = 10
ON_SPECIALIZED_ROUTE_MAX = 30
ON_NON_SPECIALIZED_ROUTE_MIN = 4

# Cuotas ON operativas confirmadas por Planeación. Son valores exactos: el
# aumento de la muestra mensual se asigna a OFF y nunca recalcula estas cuotas.
ON_CITY_QUOTAS = {
    "AREQUIPA": 65,
    "CHICLAYO": 50,
    "ICA": 50,
    "IQUITOS": 50,
    "LIMA": 635,
    "TRUJILLO": 160,
    "HUARAZ": 25,
    "PIURA": 50,
    "HUANCAYO": 50,
    "HUACHO": 25,
    "AYACUCHO": 25,
    "TARAPOTO": 25,
    "CUSCO": 50,
    "TACNA": 25,
}
ON_FIXED_TOTAL = sum(ON_CITY_QUOTAS.values())

# The August input has the Esparta field empty.  These are the Esparta cities
# documented in the original selection procedure; a populated Esparta flag is
# still honoured as well.
ESPARTA_CITIES = {"AREQUIPA", "CHICLAYO", "LIMA", "PIURA", "TRUJILLO"}


def load_lima_quotas() -> dict[str, int]:
    """Read the current quota file instead of duplicating its values in code."""
    quotas = pd.read_excel(LIMA_QUOTAS_PATH)
    quotas.columns = [str(c).strip().upper() for c in quotas.columns]
    if not {"LOC", "CUOTA"}.issubset(quotas.columns):
        raise ValueError("El archivo de cuotas Lima debe contener las columnas LOC y cuota.")

    quotas["LOC"] = quotas["LOC"].astype(str).str.strip().str.upper()
    quotas["CUOTA"] = pd.to_numeric(quotas["CUOTA"], errors="coerce").fillna(0).astype(int)
    return dict(zip(quotas["LOC"], quotas["CUOTA"]))
