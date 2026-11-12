# G7 - SI-DMSeq

Modelo publicado usado como condicion G7 del PPI Grupo 4. Corre en PyTorch, en un entorno separado del de G1-G6.

## Origen del codigo
- Repositorio de los autores: https://github.com/minanouri/SI-DMSeq
- Commit usado: 9e148c4 ("Update README.md")
- El repositorio no tiene archivo LICENSE; por eso su codigo NO se copia aqui. Se clona en ese commit y no se modifica.

## Entorno
- Python 3.11.9 y requirements-g7-lock.txt (torch 2.2.2+cpu, pandas 1.5.3, numpy 1.23.3). Solo CPU (Decision 19).

## Reconstruir G7 desde cero (Windows)
    git clone https://github.com/minanouri/SI-DMSeq.git D:\ppi\SI-DMSeq
    cd D:\ppi\SI-DMSeq
    git checkout 9e148c4
    py -3.11 -m venv .venv-g7
    .\.venv-g7\Scripts\Activate.ps1
    pip install -r D:\ppi\deteccion-anomalias-pems-d4\requirements-g7-lock.txt
    Copy-Item D:\ppi\deteccion-anomalias-pems-d4\g7_*.py, D:\ppi\deteccion-anomalias-pems-d4\smoke_g7.py D:\ppi\SI-DMSeq\
    python smoke_g7.py
    python g7_preparar.py --proyecto D:\ppi\deteccion-anomalias-pems-d4
    python g7_entrenar.py --esquema A
    python g7_entrenar.py --esquema B
    python g7_final.py --proyecto D:\ppi\deteccion-anomalias-pems-d4 --esquema A
    python g7_final.py --proyecto D:\ppi\deteccion-anomalias-pems-d4 --esquema B

## Scripts propios (raiz de este repo)
- smoke_g7.py: verifica que FastTrafficDataset (unfold) da lo mismo que TrafficDataset de los autores.
- g7_inspeccionar.py: revisa codigo y datos.
- g7_preparar.py: comprobaciones V1-V6 y genera g7_datos.npz desde data/processed.
- g7_entrenar.py: validacion en los 3 pliegues -> resultados_g7/ (copiado a results/g7/).
- g7_final.py: entrenamiento final en pasos 0-13593 y scores del bloque de umbral 10194-13593 -> resultados_g7_final/ (copiado a results/g7/final/).

## Adaptaciones a PeMS-D4 (registrar en la bitacora)
- 873 canales: flujo, ocupacion y velocidad de los 291 sensores; sin variables temporales (el modelo trata el 0 como faltante).
- Configuracion por defecto de los autores, sin busqueda: ventana 24, embedding 64, 2 capas, dropout 0.2, Adam lr 0.01, MSE suma, lote 64, 20 epocas, rho 0.25.
- Semillas fijas del codigo original (0 datos, 1 modelo, 4 entrenamiento): 1 corrida por esquema.
- Normalizacion z-score con media y desviacion de la serie ORIGINAL del bloque de entrenamiento (Decision 14).
- Score: error de reconstruccion del ultimo paso de la ventana, suma por sensor de sus 3 canales.
- pandas 1.5.3 por compatibilidad con el codigo de los autores.
