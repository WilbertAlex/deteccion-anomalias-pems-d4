@echo off
REM Descarga y verificacion del dataset PeMS04 (307 sensores, 16992 pasos, 3 variables)
REM Version para cmd.exe - equivalente a download_pems04.sh

if not exist data\raw mkdir data\raw

echo Descargando pems04.npz...
curl -sL -o data\raw\pems04.npz "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz"

echo.
echo Archivo descargado:
dir data\raw\pems04.npz

echo.
echo Hash SHA-256 calculado:
certutil -hashfile data\raw\pems04.npz SHA256 > data\raw\pems04.sha256.txt
type data\raw\pems04.sha256.txt

echo.
echo Hash de referencia verificado el 15 sep. 2026:
echo 95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a
echo.
echo Si ambos hashes coinciden (mayusculas/minusculas no importan), el archivo esta integro.
