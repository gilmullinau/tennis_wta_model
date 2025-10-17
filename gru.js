// gru.js (MLP model implementation)
const tf = window.tf;

export class ModelMLP {
  constructor(inputDim) {
    this.inputDim = inputDim;
    this.model = null;
    this.modelKey = "wta-mlp-lite-v2";
  }

  build() {
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        units: 1,
        activation: "sigmoid",
        inputShape: [this.inputDim],
        kernelInitializer: "glorotUniform",
      })
    );
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
    this.model = model;
  }

  async train(
    X_train,
    y_train,
    {
      epochs = 5,
      batchSize = 1024,
      validationSplit = 0.15,
      onEpochEnd = null,
      patience = 1,
    } = {}
  ) {
    if (!this.model) throw new Error("Model not built. Call build() first.");
    const earlyStop = tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience,
      minDelta: 1e-4,
      restoreBestWeights: true,
    });
    const historyCallback = new tf.CustomCallback({
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs);
        await tf.nextFrame();
      },
    });
    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit,
      callbacks: [historyCallback, earlyStop],
      shuffle: true,
    });
  }

  async evaluate(X_test, y_test) {
    if (!this.model) throw new Error("Model not built or loaded.");
    const evalOut = await this.model.evaluate(X_test, y_test, { batchSize: 1024 });
    const loss = (await evalOut[0].data())[0];
    const acc = (await evalOut[1].data())[0];
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
  }

  async load() {
    this.model = await tf.loadLayersModel(`localstorage://${this.modelKey}`);
    return this.model;
  }
}
