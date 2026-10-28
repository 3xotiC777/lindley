import assert from "node:assert/strict";
import { test } from "node:test";
import { readSelectedFile } from "../web/file-reader.mjs";

test("reads a local file without changing its contents", async () => {
  const file = new File([new Uint8Array([80, 75, 3, 4])], "universo.xlsx");
  assert.deepEqual(new Uint8Array(await readSelectedFile(file, "preselection")), new Uint8Array([80, 75, 3, 4]));
});

test("recovers a Blob read failure with FileReader", async () => {
  const original = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsArrayBuffer() {
      this.result = new Uint8Array([80, 75]).buffer;
      this.onload();
    }
  };
  try {
    const file = {name: "previo.xlsx", size: 2, arrayBuffer: async () => {throw new DOMException("Permission lost", "NotReadableError");}};
    assert.equal((await readSelectedFile(file, "previous")).byteLength, 2);
  } finally {
    globalThis.FileReader = original;
  }
});

test("identifies the exact unreadable file and explains how to reselect it", async () => {
  const original = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsArrayBuffer() {
      this.error = new DOMException("Permission lost", "NotReadableError");
      this.onerror();
    }
  };
  try {
    for (const key of ["preselection", "previous", "lima", "cda"]) {
      await assert.rejects(
        readSelectedFile({name: "bloqueado.xlsx", size: 2, arrayBuffer: async () => {throw new Error("failed");}}, key),
        (error) => error.fileInputKey === key && error.message.includes("bloqueado.xlsx")
          && error.message.includes("Descargas") && error.cause.name === "NotReadableError",
      );
    }
  } finally {
    globalThis.FileReader = original;
  }
});

test("rejects a missing, empty or incomplete file before starting the engine", async () => {
  for (const file of [undefined, new File([], "vacio.xlsx"), {name:"parcial.xlsx",size:5,arrayBuffer:async () => new ArrayBuffer(2)}]) {
    await assert.rejects(readSelectedFile(file, "lima"), (error) => error.fileInputKey === "lima");
  }
});
