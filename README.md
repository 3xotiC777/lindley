# Selección Lindley

Automatización de la selección mensual para los canales OFF y ON. El repositorio
contiene el motor Python usado para generar el ejecutable de escritorio y una
versión web para GitHub Pages.

## Selector web

La página permite cargar:

1. la preselección actual en `.xlsx`;
2. la selección del mes anterior en `.xlsx` o `.xlsb`;
3. las cuotas de Lima en `.xlsx`.

También permite indicar la muestra total y elegir si el aumento se asigna a
`OFF`, `ON` o a ambos canales. La ciudad se identifica mediante la columna
`NOMBRE`. Las 14 cuotas ON forman una base de 1.285 titulares: permanecen
exactas al elegir `OFF` y, al elegir `ON` o ambos, el incremento se distribuye
por encima de esa base. Al terminar, entrega:

- `SELECCION_MUESTRA_LINDLEY_RESULTADO.xlsx`;
- `ARCHIVO_DEF_SUP_LINDLEY_RESULTADO.xlsx`.

Los archivos de entrada no se envían a un backend. El sitio carga Python con
Pyodide, ejecuta el cálculo dentro de un Web Worker y conserva los datos solo en
la memoria local de la pestaña. La primera ejecución requiere conexión a
internet para descargar el runtime y las librerías de Excel.

## Un solo motor de reglas

La página no mantiene una reimplementación de las reglas en JavaScript. El
workflow de publicación copia `config.py`, `ingest.py`, `selector_engine.py` y
`exporter.py` directamente al artefacto de GitHub Pages. El archivo
`web/python/browser_runner.py` es únicamente un adaptador de rutas y opciones
para invocar el mismo flujo usado por la aplicación de escritorio.

## Publicación

El workflow `.github/workflows/pages.yml` publica el contenido de `web/` y una
copia exacta del motor Python con cada cambio en `main`. El repositorio debe usar
**GitHub Actions** como fuente de GitHub Pages.

## Privacidad del repositorio

Los archivos `.xlsx`, `.xlsb`, perfiles del navegador, temporales y resultados
están excluidos mediante `.gitignore`. No deben agregarse bases de clientes ni
credenciales al repositorio público.
