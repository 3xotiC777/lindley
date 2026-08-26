"""Desktop launcher for the Lindley selection process."""

from pathlib import Path
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import config
import exporter
import ingest
import selector_engine


OUTPUT_NAME = "SELECCION_MUESTRA_LINDLEY_AGOSTO_CORREGIDA.xlsx"
SUPERVISION_NAME = "ARCHIVO_DEF_SUP_AGOSTO_CORREGIDO.xlsx"
INCREASE_SCOPE_OPTIONS = {
    "Solo OFF": "OFF",
    "Solo ON": "ON",
    "Ambos canales": "AMBOS",
}


class SelectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Selección Lindley")
        self.resizable(False, False)
        self.paths = {
            "preselection": tk.StringVar(),
            "previous": tk.StringVar(),
            "lima": tk.StringVar(),
            "output": tk.StringVar(),
        }
        self.sample_size = tk.StringVar()
        self.increase_scope = tk.StringVar(value="Solo OFF")
        self.requested_sample_size = 0
        self.increase_scope_code = "OFF"
        self.status = tk.StringVar(value="Selecciona los archivos y la carpeta de salida.")
        self._build()

    def _build(self):
        frame = ttk.Frame(self, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        fields = [
            ("Preselección actual (.xlsx)", "preselection", self._choose_preselection),
            ("Selección mes anterior (.xlsx o .xlsb)", "previous", self._choose_previous),
            ("Cuotas Lima (.xlsx)", "lima", self._choose_lima),
            ("Carpeta de salida", "output", self._choose_output),
        ]
        for row, (label, key, command) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row * 2, column=0, columnspan=2, sticky="w", pady=(0, 3))
            entry = ttk.Entry(frame, width=72, textvariable=self.paths[key], state="readonly")
            entry.grid(row=row * 2 + 1, column=0, sticky="ew", padx=(0, 8))
            ttk.Button(frame, text="Buscar…", command=command).grid(row=row * 2 + 1, column=1, sticky="e")

        ttk.Label(frame, text="Muestra total de titulares (ejemplo: 17200)").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(12, 3)
        )
        ttk.Entry(frame, width=24, textvariable=self.sample_size).grid(row=9, column=0, sticky="w")

        ttk.Label(frame, text="Canal que recibe el aumento respecto al mes anterior").grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(12, 3)
        )
        ttk.Combobox(
            frame,
            width=22,
            textvariable=self.increase_scope,
            values=list(INCREASE_SCOPE_OPTIONS),
            state="readonly",
        ).grid(row=11, column=0, sticky="w")
        ttk.Label(
            frame,
            text="Cada ciudad conserva su cuota; sin universo queda en 0.",
        ).grid(row=11, column=1, sticky="e")

        self.run_button = ttk.Button(frame, text="Generar selección", command=self._start)
        self.run_button.grid(row=12, column=0, columnspan=2, pady=(16, 8))
        ttk.Separator(frame).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(frame, textvariable=self.status, wraplength=560, justify="left").grid(row=14, column=0, columnspan=2, sticky="w")

    def _choose_preselection(self):
        self._choose_file("preselection", "Preselección actual", [("Excel", "*.xlsx")])

    def _choose_previous(self):
        self._choose_file(
            "previous",
            "Selección del mes anterior",
            [
                ("Archivos de Excel", "*.xlsx *.xlsb"),
                ("Excel normal", "*.xlsx"),
                ("Excel binario", "*.xlsb"),
            ],
        )

    def _choose_lima(self):
        self._choose_file("lima", "Cuotas de Lima", [("Excel", "*.xlsx")])

    def _choose_file(self, key, title, filetypes):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            self.paths[key].set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(title="Carpeta donde se guardarán los resultados")
        if path:
            self.paths["output"].set(path)

    def _start(self):
        missing = [label for label, key in [
            ("la preselección", "preselection"), ("la selección anterior", "previous"),
            ("las cuotas de Lima", "lima"), ("la carpeta de salida", "output"),
        ] if not self.paths[key].get()]
        if missing:
            messagebox.showwarning("Faltan datos", "Selecciona " + ", ".join(missing) + ".")
            return
        normalized_size = self.sample_size.get().strip().replace(" ", "").replace(".", "").replace(",", "")
        if not normalized_size.isdigit() or int(normalized_size) <= 0:
            messagebox.showwarning(
                "Muestra inválida",
                "Escribe la cantidad total de titulares como un número entero positivo. Ejemplo: 17200.",
            )
            return
        self.requested_sample_size = int(normalized_size)
        self.increase_scope_code = INCREASE_SCOPE_OPTIONS[self.increase_scope.get()]
        self.run_button.state(["disabled"])
        self.status.set(
            f"Procesando una muestra de {self.requested_sample_size:,} titulares; "
            f"aumento aplicado a {self.increase_scope.get()}. Esto puede tardar unos minutos…"
        )
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            output_dir = Path(self.paths["output"].get())
            config.WORK_PRESELECCION_PATH = Path(self.paths["preselection"].get())
            config.WORK_PREV_SELECCION_PATH = Path(self.paths["previous"].get())
            config.LIMA_QUOTAS_PATH = Path(self.paths["lima"].get())
            config.OUTPUT_SELECCION_PATH = output_dir / OUTPUT_NAME
            config.OUTPUT_DEF_SUP_PATH = output_dir / SUPERVISION_NAME

            eligible, non_eligible, historical = ingest.prepare_merged_datasets()
            final, controls = selector_engine.run_selection_process(
                eligible, historical, total_target=self.requested_sample_size,
                increase_scope=self.increase_scope_code,
            )
            exporter.export_selection_workbook(final, non_eligible, controls, config.OUTPUT_SELECCION_PATH)
            exporter.export_def_sup_workbook(final, config.OUTPUT_DEF_SUP_PATH)
            actual_sample_size = int(final[config.SELECTION_COL].eq("T").sum())
            actual_on = int((final[config.SELECTION_COL].eq("T") & final["_CANAL_SELECCION"].eq("ON")).sum())
            actual_off = actual_sample_size - actual_on
            self.after(0, lambda: self._completed(actual_sample_size, actual_on, actual_off))
        except Exception as error:
            details = "".join(traceback.format_exception(error))
            self.after(0, lambda: self._failed(str(error), details))

    def _completed(self, actual_sample_size, actual_on, actual_off):
        self.run_button.state(["!disabled"])
        text = (
            "Proceso terminado.\n\n"
            f"Titulares solicitados: {self.requested_sample_size:,}\n"
            f"Titulares seleccionados: {actual_sample_size:,}\n\n"
            f"Canal del aumento: {self.increase_scope.get()}\n"
            f"Titulares ON: {actual_on:,}\n"
            f"Titulares OFF: {actual_off:,}\n\n"
            f"Selección: {config.OUTPUT_SELECCION_PATH}\n"
            f"Supervisión: {config.OUTPUT_DEF_SUP_PATH}"
        )
        self.status.set(text)
        messagebox.showinfo("Selección Lindley", text)

    def _failed(self, error, details):
        self.run_button.state(["!disabled"])
        self.status.set("El proceso no pudo terminar. Revisa los archivos seleccionados.")
        messagebox.showerror("Error en la selección", f"{error}\n\nDetalle técnico:\n{details}")


if __name__ == "__main__":
    SelectorApp().mainloop()
