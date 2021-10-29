class Network {

  Sequential model;
  Sequential.Options opt;

  ArrayList<Float> epochTrainLoss, epochValLoss;
  ArrayList<Integer> totEpochs;

  DataManager trainData, testData;
  boolean training, testing, predicting, loaded;

  Network(DataManager trainData, DataManager testData) {

    this.trainData = trainData;
    this.testData = testData;
    
    epochTrainLoss = new ArrayList<Float>();
    epochValLoss = new ArrayList<Float>();
    totEpochs = new ArrayList<Integer>();

    model = new Sequential();
    training = false;
    testing = false;
    predicting = false;
    loaded = false;
  }

  void prepare() {

    loaded = false;

    model = new Sequential();
    model.add(new Conv2D(new int[] {h, w, 1}, 4, new int[] {3, 3}, Sequential.ACTIVATION.RELU));
    //model.add(new Dense(w * h, 32, Sequential.ACTIVATION.RELU));
    model.add(new Dense(64, Sequential.ACTIVATION.RELU));
    model.add(new Dense(trainData.dataSetsCount, Sequential.ACTIVATION.SOFTMAX));
    model.optimizer(Optimizer.LOSS.CROSSENTROPY);

    opt = model.new Options(model) {

      @Override public void tweak() {

        lr = 0.001;
        shuffle = true;
        epochs = nEpochs;
        //valSplit = 0.2;
      }
      
      @Override public void onEpochEnd(int epoch, float tLoss, float vLoss, String log) {
        
        epochTrainLoss.add(tLoss);
        epochValLoss.add(vLoss);
        totEpochs.add(epochCount++);
      }
    };
    
    resetTesting();
    eraseData();
  }

  void loadTrained() {

    loaded = true;
    resetTesting();
    
    model = Sequential.loadModel(dataPath("SavedModels/digits60k_SOFTMAX-CROSSENTROPY_3.txt"));
  }

  void nextTrainData(int nData) {

    trainData.prepareSets(nData);
  }

  void train() {

    training = true;
    trainData.hasToUpdateData = true;

    Vector[] trainSet = trainData.getSampleSet();
    Vector[] labelSet = trainData.getLabelSet();

    model.fit(trainSet, labelSet, opt);
    training = false;
  }

  int testOnce(Vector sample, Vector label) {

    if (predict(sample).indexMax() == label.indexMax())
      return 1;

    return 0;
  }
  
  void eraseData() {
   
    epochTrainLoss.clear();
    epochValLoss.clear();
    totEpochs.clear();
    epochCount = 1;
  }

  Vector predict(Vector input) {

    return model.feedForward(input);
  }

  boolean isPrepared() {

    return model != null && !model.isEmpty();
  }

  boolean isTraining() {

    return training;
  }

  boolean isTesting() {

    return testing;
  }

  boolean isPredicting() {

    return predicting;
  }

  boolean isBusy() {

    return isTraining() || isTesting() || isPredicting();
  }

  boolean isLoaded() {

    return loaded;
  }

  void validateModel() {

    if (!isPrepared())
      throw new RuntimeException("Model is not prepared");

    if (model.isEmpty())
      throw new RuntimeException("Model is empty");
  }
}
