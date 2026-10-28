const labels = {
  preselection: "Preselección actual",
  previous: "Selección del mes anterior",
  lima: "Cuotas Lima",
  cda: "Cuotas CDA",
};

function readWithFileReader(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("No se pudo leer el archivo."));
    reader.onabort = () => reject(new Error("Se canceló la lectura del archivo."));
    reader.readAsArrayBuffer(file);
  });
}

export async function readSelectedFile(file, key) {
  const label = labels[key] || "Archivo";
  try {
    if (!file) throw new Error("El archivo ya no está seleccionado.");
    let buffer;
    try {
      buffer = await file.arrayBuffer();
    } catch {
      // Some browsers can recover a transient Blob read failure this way.
      // Both methods use the local file; neither sends its contents anywhere.
      buffer = await readWithFileReader(file);
    }
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== file.size || !buffer.byteLength) {
      throw new Error("La lectura del archivo quedó vacía o incompleta.");
    }
    return buffer;
  } catch (cause) {
    const error = new Error(
      `No se pudo leer ${label}: “${file?.name || "sin archivo"}”. `
      + "Copia o descarga el Excel a una carpeta local como Descargas, comprueba que abre correctamente "
      + "y vuelve al formulario para seleccionarlo otra vez. Si está en OneDrive o SharePoint, "
      + "espera a que termine de descargarse o sincronizarse."
    );
    error.fileInputKey = key;
    error.cause = cause;
    throw error;
  }
}
