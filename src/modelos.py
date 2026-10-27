"""
src/modelos.py - Arquitecturas G1-G6 (TensorFlow/Keras). Entrada (12, 8) float32.

NINGUN hiperparametro tiene valor por defecto: todos se pasan en un diccionario `cfg` y salen del
Cuadro 6 del PPI (via configs/config_modelos.json). Si falta una clave, falla con KeyError.
Las configuraciones diminutas de semana7_humo_modelos.py son SOLO pruebas de forma, no son del Cuadro 6.

Arquitectura   claves requeridas en cfg
  lstm, gru    unidades, capas, dropout, lr
  cnn1d        filtros, kernel, capas, dropout, lr
  transformer  d_model, cabezas, ff, capas, dropout, lr
  cnn_lstm_gru filtros, kernel, unidades_lstm, unidades_gru, dropout, lr
  autoencoder  unidades_ocultas, dim_latente, lr

Los modelos supervisados (G1, G2, G3, G5, G6) emiten P(anomalia) con sigmoide y se entrenan con entropia
cruzada binaria. El autoencoder (G4) reconstruye la ventana; su score es el error cuadratico medio de
reconstruccion de la ventana y se entrena SOLO con ventanas de etiqueta 0 (ver `entrenar_ae` en
semana7_busqueda.py). Si el Cuadro 6 define otro score para G4, cambia unicamente `score_ae`.
"""
ARQUITECTURAS = ("lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru")
CLAVES = {
    "lstm": ("unidades", "capas", "dropout", "lr"),
    "gru": ("unidades", "capas", "dropout", "lr"),
    "cnn1d": ("filtros", "kernel", "capas", "dropout", "lr"),
    "transformer": ("d_model", "cabezas", "ff", "capas", "dropout", "lr"),
    "cnn_lstm_gru": ("filtros", "kernel", "unidades_lstm", "unidades_gru", "dropout", "lr"),
    "autoencoder": ("unidades_ocultas", "dim_latente", "lr"),
}


def _PosicionAprendida(ventana, d):
    """Capa que suma un embedding de posicion aprendido (ventana, d) a la entrada (B, ventana, d)."""
    import tensorflow as tf

    class PosicionAprendida(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.pos = self.add_weight(name="pos", shape=(ventana, d), initializer="random_normal", trainable=True)

        def call(self, x):
            return x + self.pos

    return PosicionAprendida()


def _pedir(arq, cfg):
    faltan = [k for k in CLAVES[arq] if k not in cfg or cfg[k] is None]
    if faltan:
        raise KeyError(f"{arq}: faltan hiperparametros {faltan} (deben venir del Cuadro 6)")
    return cfg


def _compilar(m, cfg, supervisado=True):
    import tensorflow as tf
    if supervisado:
        m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    else:
        m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss="mse")
    return m


def construir(arq, cfg, ventana=12, n_features=8):
    import tensorflow as tf
    L = tf.keras.layers
    assert arq in ARQUITECTURAS, arq
    cfg = _pedir(arq, cfg)
    inp = tf.keras.Input(shape=(ventana, n_features), dtype="float32")
    if arq in ("lstm", "gru"):
        capa = L.LSTM if arq == "lstm" else L.GRU
        x = inp
        for i in range(cfg["capas"]):
            x = capa(cfg["unidades"], return_sequences=(i < cfg["capas"] - 1), dropout=cfg["dropout"])(x)
    elif arq == "cnn1d":
        x = inp
        for _ in range(cfg["capas"]):
            x = L.Conv1D(cfg["filtros"], cfg["kernel"], padding="causal", activation="relu")(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(cfg["dropout"])(x)
    elif arq == "transformer":
        d = cfg["d_model"]
        assert d % cfg["cabezas"] == 0, "d_model debe ser multiplo de cabezas"
        x = L.Dense(d)(inp)
        x = _PosicionAprendida(ventana, d)(x)
        for _ in range(cfg["capas"]):
            a = L.MultiHeadAttention(cfg["cabezas"], d // cfg["cabezas"], dropout=cfg["dropout"])(x, x)
            x = L.LayerNormalization()(L.Add()([x, a]))
            f = L.Dense(d)(L.Dense(cfg["ff"], activation="relu")(x))
            x = L.LayerNormalization()(L.Add()([x, L.Dropout(cfg["dropout"])(f)]))
        x = L.GlobalAveragePooling1D()(x)
    elif arq == "cnn_lstm_gru":
        x = L.Conv1D(cfg["filtros"], cfg["kernel"], padding="causal", activation="relu")(inp)
        x = L.LSTM(cfg["unidades_lstm"], return_sequences=True, dropout=cfg["dropout"])(x)
        x = L.GRU(cfg["unidades_gru"], dropout=cfg["dropout"])(x)
    else:  # autoencoder
        x = L.Flatten()(inp)
        x = L.Dense(cfg["unidades_ocultas"], activation="relu")(x)
        z = L.Dense(cfg["dim_latente"], activation="relu")(x)
        x = L.Dense(cfg["unidades_ocultas"], activation="relu")(z)
        x = L.Dense(ventana * n_features)(x)
        out = L.Reshape((ventana, n_features))(x)
        return _compilar(tf.keras.Model(inp, out), cfg, supervisado=False)
    out = L.Dense(1, activation="sigmoid")(x)
    return _compilar(tf.keras.Model(inp, out), cfg)


def score_ae(modelo, x):
    """Score de G4: error cuadratico medio de reconstruccion por ventana, shape (B,)."""
    import tensorflow as tf
    r = modelo(x, training=False)
    return tf.reduce_mean(tf.square(x - r), axis=[1, 2])
