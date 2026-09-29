#!/usr/bin/env bash
# Descarga y verificación del dataset PeMS04 (307 sensores, 16 992 pasos, 3 variables)
# Tarea de la Semana 1 asignada a Mayta.
set -e

mkdir -p data/raw

echo "Descargando pems04.npz..."
curl -sL -o data/raw/pems04.npz \
  "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz"

echo ""
echo "Archivo descargado:"
ls -lh data/raw/pems04.npz

echo ""
echo "Hash SHA-256 calculado:"
sha256sum data/raw/pems04.npz

echo ""
echo "Hash de referencia verificado el 15 sep. 2026:"
echo "95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a"
echo ""
echo "Si ambos hashes coinciden, el archivo esta integro."
echo "Guarda esta salida (o el archivo hash.txt) como evidencia de reproducibilidad."

sha256sum data/raw/pems04.npz > data/raw/pems04.sha256.txt
