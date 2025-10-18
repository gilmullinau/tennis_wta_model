// gru.js (MLP model implementation)
const tf = window.tf;

export class ModelMLP {
  constructor(inputDim, config = {}) {
    this.inputDim = inputDim;
    this.model = null;
    this.modelKey = "wta-mlp-v1";
    this.metaKey = `${this.modelKey}-meta`;
    this.config = {
      hiddenUnits: config.hiddenUnits && config.hiddenUnits.length ? config.hiddenUnits : [128, 64],
      dropout: typeof config.dropout === "number" ? config.dropout : 0.3,
      learningRate: typeof config.learningRate === "number" ? config.learningRate : null,
    };
  }

  build() {
    if (this.model) this.dispose();
    const model = tf.sequential();
    const hiddenUnits = Array.from(this.config.hiddenUnits).filter((u) => Number.isFinite(u) && u > 0);
    if (hiddenUnits.length === 0) hiddenUnits.push(64);
    hiddenUnits.forEach((units, idx) => {
      const layerConfig = { units, activation: "relu" };
      if (idx === 0) layerConfig.inputShape = [this.inputDim];
      model.add(tf.layers.dense(layerConfig));
      if (idx === 0 && this.config.dropout && this.config.dropout > 0) {
        model.add(tf.layers.dropout({ rate: Math.max(0, Math.min(0.9, this.config.dropout)) }));
      }
    });
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    const optimizer = this.config.learningRate
      ? tf.train.adam(this.config.learningRate)
      : tf.train.adam();
    model.compile({
      optimizer,
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
    this.model = model;
  }

  async train(X_train, y_train, { epochs = 6, batchSize = 256, validationSplit = 0.2, onEpochEnd = null } = {}) {
    if (!this.model) throw new Error("Model not built. Call build() first.");
    const cb = {
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs);
        await tf.nextFrame();
      }
    };
    return await this.model.fit(X_train, y_train, {
      epochs, batchSize, validationSplit, callbacks: cb, shuffle: true,
    });
  }

  async evaluate(X_test, y_test) {
    if (!this.model) throw new Error("Model not built or loaded.");
    const evalOut = await this.model.evaluate(X_test, y_test, { batchSize: 256 });
    const lossTensor = evalOut[0];
    const accTensor = evalOut[1];
    const loss = (await lossTensor.data())[0];
    const acc = (await accTensor.data())[0];
    lossTensor.dispose();
    accTensor.dispose();
    return { loss, acc };
  }

  predictProba(X) {
    if (!this.model) throw new Error("Model not built or loaded.");
    return this.model.predict(X);
  }

  async confusionMatrix(X_test, y_test) {
    const probs = this.model.predict(X_test);
    const probsData = await probs.data();
    probs.dispose();
    const yTrue = Array.from((await y_test.data()));
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const pred = probsData[i] >= 0.5 ? 1 : 0;
      if (pred === 1 && yTrue[i] === 1) tp++;
      else if (pred === 0 && yTrue[i] === 0) tn++;
      else if (pred === 1 && yTrue[i] === 0) fp++;
      else fn++;
    }
    return { tp, tn, fp, fn };
  }

  async save() {
    if (!this.model) throw new Error("Model not built.");
    await this.model.save(`localstorage://${this.modelKey}`);
    localStorage.setItem(this.metaKey, JSON.stringify({
      inputDim: this.inputDim,
      config: this.config,
    }));
  }

  async load() {
    this.model = await tf.loadLayersModel(`localstorage://${this.modelKey}`);
    try {
      const meta = JSON.parse(localStorage.getItem(this.metaKey) || "null");
      if (meta && Number.isFinite(meta.inputDim)) this.inputDim = meta.inputDim;
      if (meta && meta.config) {
        this.config = {
          hiddenUnits: Array.isArray(meta.config.hiddenUnits) ? meta.config.hiddenUnits : this.config.hiddenUnits,
          dropout: typeof meta.config.dropout === "number" ? meta.config.dropout : this.config.dropout,
          learningRate: typeof meta.config.learningRate === "number" ? meta.config.learningRate : this.config.learningRate,
        };
      }
    } catch (err) {
      console.warn("Failed to restore model metadata", err);
    }
    const inferred = this.model?.inputs?.[0]?.shape?.[1];
    if (Number.isFinite(inferred)) this.inputDim = inferred;
    return this.model;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (tf?.engine) {
      tf.engine().disposeVariables();
    }
  }
}
