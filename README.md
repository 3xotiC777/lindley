# Selección Lindley

Automatización de la selección mensual para los canales OFF y ON. El repositorio
contiene el motor Python usado para generar el ejecutable de escritorio y una
versión web para GitHub Pages.

## Selector web

La página permite cargar:

1. la preselección actual en `.xlsx`;
2. la selección del mes anterior en `.xlsx` o `.xlsb`;
3. las cuotas de Lima en `.xlsx`;
4. opcionalmente, las cuotas por CDA en `.xlsx` (hoja `General`).

También permite indicar la muestra total y elegir si el aumento se asigna a
`OFF`, `ON` o a ambos canales. La ciudad se identifica mediante la columna
`NOMBRE`. Las 14 cuotas ON forman una base de 1.285 titulares: permanecen
exactas al elegir `OFF` y, al elegir `ON` o ambos, el incremento se distribuye
por encima de esa base.

Si se adjuntan cuotas por CDA, las columnas `LOCACION/CIUDAD`, `On Premise` y
`Total` determinan toda la muestra: ON = `On Premise`, OFF = `Total - On Premise`.
La página cruza el CDA con `DES LOC_COM` del universo y toma el total de la
suma de las filas, sin usar el tamaño ni el canal de aumento manuales. Una
cuota que supera la capacidad del universo o de las rutas se completa hasta
el máximo seleccionable de ese mismo CDA. La página muestra el aviso de
capacidad junto a las descargas y `CONTROL CUOTAS` conserva la cuota original,
el resultado y el faltante para revisión comercial; no se redistribuye a otro
CDA. La descarga no equivale a una aprobación comercial. En este modo, las
rutas OFF con Titán o Fénix tienen prioridad. Si no bastan para cumplir la
cuota CDA y el mix, se completan titulares y suplentes en rutas OFF sin esos
puntos. Los controles identifican cuántos titulares usaron ese apoyo por CDA;
las demás reglas no cambian. Sin archivo de cuotas CDA, la restricción
original de rutas OFF permanece vigente.

Con cuotas CDA, `NOMBRE` vacío (`nan`, `0` o celda vacía) excluye los puntos
ordinarios de la selección. Los Titán/Fénix continúan siendo titulares
obligatorios aun sin `NOMBRE` y cuentan **dentro**, no además, de la cuota de su
CDA. `CONTROL CUOTAS` muestra por CDA/canal los titulares con `NOMBRE`, los
obligatorios sin `NOMBRE` y los puntos ordinarios excluidos. Una fila adicional
`Titán/Fénix sin NOMBRE por CDA` marca estos casos para revisión, incluido
Chulucanas. Si al excluir los puntos ordinarios sin `NOMBRE` un CDA ya no tiene
capacidad suficiente, su faltante queda señalado allí; no se traslada a otro
CDA ni se inventa una ciudad.

Con cuotas CDA, una variación del mix histórico superior a +/- 1 punto
porcentual no impide la descarga: queda marcada como `REVISAR` en `CONTROL
CUOTAS`. Tampoco la impide una diferencia en el balance exacto entre rutas ON
si cada ruta sigue entre 10 y 30 titulares especializados y conserva suplentes.
Las cuotas CDA exactas, salvo un faltante documentado por capacidad, los
puntos obligatorios y los demás controles críticos siguen siendo requisitos
para generar los archivos. Un CDA inexistente en el universo o una cuota
inferior a sus Titán/Fénix obligatorios todavía impide el cálculo.

Al terminar, entrega:

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
