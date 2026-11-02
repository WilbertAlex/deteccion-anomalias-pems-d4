# Presupuesto de computo (calculado de results/eda/cronometro_fit_*.json)

Supuestos: 12 configuraciones, hasta 10 epocas, 3 semillas en el reentrenamiento final, 2 esquemas. Factor de los 3 pliegues = 2.00; bloque 80 % = 1.333 x pliegue 3.

| Arquitectura | ms/paso | min/epoca busqueda | min/epoca final (80 %) | busqueda (h) | reent. final (h) | ablacion (h) | pesos de clase (h) |
|---|---|---|---|---|---|---|---|
| autoencoder | 2.55 | 0.49 | 2.62 | 3.9 | 2.6 | 0.0 | 0.9 |
| cnn1d | 4.43 | 0.86 | 4.56 | 6.8 | 4.6 | 0.8 | 1.5 |
| cnn_lstm_gru | 10.68 | 2.06 | 10.98 | 16.5 | 11.0 | 1.8 | 3.7 |
| gru | 12.85 | 2.48 | 13.22 | 19.8 | 13.2 | 2.2 | 4.4 |
| transformer | 19.55 | 3.77 | 20.11 | 30.2 | 20.1 | 3.4 | 6.7 |
| lstm | 9.95 | 1.92 | 10.24 | 15.4 | 10.2 | 1.7 | 3.4 |

Sin medir: ninguna.
Total si las faltantes cuestan el promedio de las medidas: **185 h = 7.7 dias continuos**.
Total si cuestan el doble: **185 h = 7.7 dias**. Disponible por maquina hasta el 29 nov: 648 h.

No incluye validacion por epoca, puntuacion final ni G7. La parada temprana solo reduce estas cifras.
