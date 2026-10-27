"""
src/modelos.py - Arquitecturas G1-G6 (TensorFlow/Keras). Entrada (12, 8) float32.

Estructura segun el Cuadro 6 del PPI (configuracion inicial). Ningun hiperparametro tiene valor por
defecto: todos se pasan en `cfg` (configs/config_modelos.json: "inicial" = Cuadro 6; "espacio" = rangos de la
busqueda, Decision 17). Si falta una clave, falla con KeyError.

Arquitectura   claves requeridas en cfg
  lstm, gru    unidades (lista, p. ej. [64, 32]), dropout, lr                    Adam
  cnn1d        filtros (lista, p. ej. [32, 64, 128]), kernel, weight_decay, lr   AdamW, GlobalMaxPooling1D
  transformer  bloques, cabezas, d_model, ff, dropout, lr, weight_decay          AdamW
  cnn_lstm_gru filtros (int), kernel, unidades_lstm, unidades_gru, dropout, lr   Adam
  autoencoder  capas (lista, p. ej. [32, 16, 8]), l2, lr                         Adam; decodificador simetrico

Interpretaciones NO fijadas por el Cuadro 6 (declaradas en la Decision 17):
  - `dropout` de LSTM/GRU/CNN-LSTM-GRU = argumento `dropout` de la capa recurrente (sobre la entrada);
  - convoluciones con padding causal; Transformer con posicion aprendida y GlobalAveragePooling1D;
  - weight_decay de AdamW (G3, G5) y l2 del autoencoder son hiperparametros de la busqueda.
Modelos supervisados (G1, G2, G3, G5, G6): P(anomalia) con sigmoide, entropia cruzada binaria. El
autoencoder (G4) reconstruye la ventana (96 valores aplanados); su score es el error cuadratico medio de
reconstruccion y se entrena SOLO con ventanas de etiqueta 0.
"""
ARQUITECTURAS = ("lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru")
CLAVES = {
    "lstm": ("unidades", "dropout", "lr"),
    "gru": ("unidades", "dropout", "lr"),
    "cnn1d": ("filtros", "kernel", "weight_decay", "lr"),
    "transformer": ("bloques", "cabezas", "d_model", "ff", "dropout", "lr", "weight_decay"),
    "cnn_lstm_gru": ("filtros", "kernel", "unidades_lstm", "unidades_gru", "dropout", "lr"),
    "autoencoder": ("capas", "l2", "lr"),
}


def _pedir(arq, cfg):
    faltan = [k for k in CLAVES[arq] if k not in cfg or cfg[k] is None]
    if faltan:
        raise KeyError(f"{arq}: faltan hiperparametros {faltan}")
    return cfg


def _PosicionAprendida(ventana, d):
    import tensorflow as tf

    class PosicionAprendida(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.pos = self.add_weight(name="pos", shape=(ventana, d), initializer="random_normal", trainable=True)

        def call(self, x):
            return x + self.pos

    return PosicionAprendida()


def _compilar(m, arq, cfg):
    import tensorflow as tf
    if arq in ("cnn1d", "transformer"):
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = tf.keras.optimizers.Adam(cfg["lr"])
    if arq == "autoencoder":
        m.compile(optimizer=opt, loss="mse")
    else:
        m.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return m


def construir(arq, cfg, ventana=12, n_features=8):
    import tensorflow as tf
    L = tf.keras.layers
    assert arq in ARQUITECTURAS, arq
    cfg = _pedir(arq, cfg)
    inp = tf.keras.Input(shape=(ventana, n_features), dtype="float32")
    if arq in ("lstm", "gru"):
        capa = L.LSTM if arq == "lstm" else L.GRU
        x, u = inp, list(cfg["unidades"])
        for i, n in enumerate(u):
            x = capa(n, return_sequences=(i < len(u) - 1), dropout=cfg["dropout"])(x)
    elif arq == "cnn1d":
        x = inp
        for f in cfg["filtros"]:
            x = L.Conv1D(f, cfg["kernel"], padding="causal", activation="relu")(x)
        x = L.GlobalMaxPooling1D()(x)
    elif arq == "transformer":
        d = cfg["d_model"]
        assert d % cfg["cabezas"] == 0, "d_model debe ser multiplo de cabezas"
        x = _PosicionAprendida(ventana, d)(L.Dense(d)(inp))
        for _ in range(cfg["bloques"]):
            a = L.MultiHeadAttention(cfg["cabezas"], d // cfg["cabezas"], dropout=cfg["dropout"])(x, x)
            x = L.LayerNormalization()(L.Add()([x, a]))
            f = L.Dense(d)(L.Dense(cfg["ff"], activation="relu")(x))
            x = L.LayerNormalization()(L.Add()([x, L.Dropout(cfg["dropout"])(f)]))
        x = L.GlobalAveragePooling1D()(x)
    elif arq == "cnn_lstm_gru":
        x = L.Conv1D(cfg["filtros"], cfg["kernel"], padding="causal", activation="relu")(inp)
        x = L.LSTM(cfg["unidades_lstm"], return_sequences=True, dropout=cfg["dropout"])(x)
        x = L.GRU(cfg["unidades_gru"], dropout=cfg["dropout"])(x)
    else:  # autoencoder: Flatten(96) -> capas -> simetrico -> 96 -> Reshape(12, 8)
        reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] else None
        x = L.Flatten()(inp)
        for n in cfg["capas"]:
            x = L.Dense(n, activation="relu", kernel_regularizer=reg)(x)
        for n in list(cfg["capas"])[-2::-1]:
            x = L.Dense(n, activation="relu", kernel_regularizer=reg)(x)
        x = L.Dense(ventana * n_features)(x)
        out = L.Reshape((ventana, n_features))(x)
        return _compilar(tf.keras.Model(inp, out), arq, cfg)
    out = L.Dense(1, activation="sigmoid")(x)
    return _compilar(tf.keras.Model(inp, out), arq, cfg)


def score_ae(modelo, x):
    """Score de G4: error cuadratico medio de reconstruccion por ventana, shape (B,)."""
    import tensorflow as tf
    r = modelo(x, training=False)
    return tf.reduce_mean(tf.square(x - r), axis=[1, 2])
