import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import tensors.Float.*; 
import deepLearning.utilities.*; 
import grafica.*; 
import checkBox.*; 
import java.util.Random; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class mnist extends PApplet {







PlotWindow graphs;

int w = 28;
int h = 28;

int epochCount = 1;

int nEpochs = 10;
int trainDataAmount = 100;

int globalTestWinTotal = 0;

CheckBox prepModelToggle, nextTrainDataToggle, trainToggle, testToggle, predictBoardToggle, loadTrainedToggle, resetTestingToggle;
CheckBox[] boxes;
DataManager mnistTrain, mnistTest;

Network myModel;
GPlot lossPlot;

PGraphics board;

public void settings() {
  
  DataSet[] mnistTrainData = organizeMnistData(loadBytes("train-images.idx3-ubyte"), loadBytes("train-labels.idx1-ubyte"), 16, 8);
  DataSet[] mnistTestData = organizeMnistData(loadBytes("test-images.idx3-ubyte"), loadBytes("test-labels.idx1-ubyte"), 16, 8);

  mnistTrain = new DataManager(mnistTrainData);
  mnistTest = new DataManager(mnistTestData);

  myModel = new Network(mnistTrain, mnistTest);

  runGraphWindow();
  size(1008, 800);
}

public void setup() {

  initCheckBoxes();

  board = createGraphics(width, height);
}

public void draw() {

  background(0);
  image(board, 0, 0);

  if (myModel.isTesting())
    testModel();
  else if (myModel.isPredicting())
    showPrediction();
    
  showDataCount();
  showCheckBoxes();
}
class DataManager {

  DataSet[] dataSets;

  int totalData;
  int dataSetsCount;

  Vector[] sampleSet;
  Vector[] labelSet;
  
  boolean dataReady;
  boolean hasToUpdateData;
  
  Random rand = new Random();

  DataManager(DataSet[] dataSet) {
    this.dataSets = dataSet;
    totalData = 0;
    dataSetsCount = dataSet.length;

    for (DataSet ds : dataSet)
      totalData += ds.totalData;
      
    dataReady = false;
    hasToUpdateData = true;
  }

  public void prepareSets(int size) {
    
    dataReady = true;
    hasToUpdateData = false;

    sampleSet = new Vector[dataSetsCount * size];
    labelSet = new Vector[dataSetsCount * size];

    for (int i = 0; i < dataSetsCount; i++) {
      
      Vector[] smallSSet = dataSets[i].getNextData(size);
      Vector[] smallLSet = new Vector[size];
      
      int label = dataSets[i].label;
      for (int j = 0; j < size; j++) {
        
        smallLSet[j] = new Vector(dataSetsCount);
        smallLSet[j].set(label, 1.0f);
        
        int index = i * size + j;

        sampleSet[index] = smallSSet[j];
        labelSet[index] = smallLSet[j];
      }
    }
  }
  
  public Vector[] getSampleSet() {
   
    return sampleSet;
  }
  
  public Vector[] getLabelSet() {
   
    return labelSet;
  }
  
  public Vector[] getRandomSample() {
    
    DataSet randomSet = dataSets[rand.nextInt(dataSetsCount)];
    
    Vector label = new Vector(dataSetsCount);
    label.set(randomSet.label, 1);
    Vector sample = randomSet.getInputVector(rand.nextInt(dataSetsCount));
    
    return new Vector[] {sample, label};
  }
  
  public PImage getRandomImage() {
   
    DataSet randomSet = dataSets[rand.nextInt(dataSetsCount)];
    
    return randomSet.getImage(rand.nextInt(randomSet.totalData));
  }
  
  public DataSet getRandomDataSet() {
   
    return dataSets[rand.nextInt(dataSetsCount)];
  }
  
  public boolean hasDataReady() {
   
    return dataReady;
  }
  
  public int getUsedData() {
   
    int total = 0;
    for (DataSet ds : dataSets)
      total += ds.usedData;
      
    return total;
  }
  
  public void resetUsedData() {
   
    for (DataSet ds : dataSets)
      ds.resetUsedData();
  }
}
class DataSet {

  byte[] dataBytes;
  int label, usedData;

  int totalData;
  String name;

  DataSet(byte[] rawBytes, int label, int excess, String name) {

    dataBytes = getRelevantBytes(rawBytes, excess);
    this.label = label;
    this.name = name;

    usedData = 0;
    
    int size = w * h;
    totalData = dataBytes.length / size;
  }

  public byte[] getRelevantBytes(byte[] rawBytes, int excess) {
    byte[] result = new byte[rawBytes.length - excess];

    for (int i = excess; i < rawBytes.length; i++)
      result[i - excess] = rawBytes[i];

    return result;
  }
  
  public Vector[] getNextData(int size) {
   
     Vector[] sampleSet = new Vector[size];
     for (int i = 0; i < size; i++) {
       
       int j = i + usedData;
       sampleSet[i] = getInputVector(j);
     }
     
     usedData += size;
     return sampleSet;
  }
  
  public PImage getNextImage() {
   
    return getImage(usedData++);
  }
  
  public Vector getInputVector(int index) {
   
    return getInputVector(getImage(index));
  }
  
  public Vector getInputVector(PImage img) {
    
    Vector result = new Vector(w * h);
    img.loadPixels();
    
    for (int i = 0; i < w * h; i++) {
     
      float normalized = map(img.pixels[i], color(0), color(255), 0, 1);
      result.set(i, normalized);
    }
    
    img.updatePixels();
    return result;
  }

  public PImage getImage(int index) {

    PImage result = createImage(w, h, RGB);
    result.loadPixels();

    int start = index * w * h;
    for (int i = 0; i < result.width * result.height; i++) {
      int j = start + i;
      result.pixels[i] = color(dataBytes[j] & 0xff);
    }

    result.updatePixels();
    return result;
  }
  
  public void resetUsedData() {
   
    usedData = 0;
  }
}
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

  public void prepare() {

    loaded = false;

    model = new Sequential();
    model.add(new Conv2D(new int[] {h, w, 1}, 4, new int[] {3, 3}, Sequential.ACTIVATION.RELU));
    //model.add(new Dense(w * h, 32, Sequential.ACTIVATION.RELU));
    model.add(new Dense(64, Sequential.ACTIVATION.RELU));
    model.add(new Dense(trainData.dataSetsCount, Sequential.ACTIVATION.SOFTMAX));
    model.optimizer(Optimizer.LOSS.CROSSENTROPY);

    opt = model.new Options(model) {

      @Override public void tweak() {

        lr = 0.001f;
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

  public void loadTrained() {

    loaded = true;
    resetTesting();
    
    model = Sequential.loadModel(dataPath("SavedModels/digits60k_SOFTMAX-CROSSENTROPY_3.txt"));
  }

  public void nextTrainData(int nData) {

    trainData.prepareSets(nData);
  }

  public void train() {

    training = true;
    trainData.hasToUpdateData = true;

    Vector[] trainSet = trainData.getSampleSet();
    Vector[] labelSet = trainData.getLabelSet();

    model.fit(trainSet, labelSet, opt);
    training = false;
  }

  public int testOnce(Vector sample, Vector label) {

    if (predict(sample).indexMax() == label.indexMax())
      return 1;

    return 0;
  }
  
  public void eraseData() {
   
    epochTrainLoss.clear();
    epochValLoss.clear();
    totEpochs.clear();
    epochCount = 1;
  }

  public Vector predict(Vector input) {

    return model.feedForward(input);
  }

  public boolean isPrepared() {

    return model != null && !model.isEmpty();
  }

  public boolean isTraining() {

    return training;
  }

  public boolean isTesting() {

    return testing;
  }

  public boolean isPredicting() {

    return predicting;
  }

  public boolean isBusy() {

    return isTraining() || isTesting() || isPredicting();
  }

  public boolean isLoaded() {

    return loaded;
  }

  public void validateModel() {

    if (!isPrepared())
      throw new RuntimeException("Model is not prepared");

    if (model.isEmpty())
      throw new RuntimeException("Model is empty");
  }
}
class PlotWindow extends PApplet {
 
  GPointsArray trainPoints;
  GPointsArray valPoints;

  public void settings() {
    size(640, 360);
  }

  public void setup() {

    trainPoints = new GPointsArray();
    valPoints = new GPointsArray();

    lossPlot = new GPlot(this, 0, 0, width, height);
    lossPlot.addLayer("Train Loss", trainPoints);
    lossPlot.addLayer("Val Loss", valPoints);

    lossPlot.getXAxis().setAxisLabelText("Epochs");
    lossPlot.getYAxis().setAxisLabelText("Model Loss");

    lossPlot.getLayer("Train Loss").setLineColor(color(255, 0, 0));
    lossPlot.getLayer("Train Loss").setPointColor(color(255, 0, 0));
    lossPlot.getLayer("Train Loss").setPointSize(1);

    lossPlot.getLayer("Val Loss").setLineColor(color(0, 0, 255));
    lossPlot.getLayer("Val Loss").setPointColor(color(0, 0, 255));
    lossPlot.getLayer("Val Loss").setPointSize(1);
  }

  int threshold = 0;
  public void draw() {

    if (threshold > myModel.totEpochs.size()) {

      trainPoints = new GPointsArray();
      valPoints = new GPointsArray();
      threshold = 0;
    } else
      for (int i = threshold; i < myModel.totEpochs.size(); i++) {

        int epoch = myModel.totEpochs.get(i);

        float tLoss = myModel.epochTrainLoss.get(i);
        float vLoss = myModel.epochValLoss.get(i);

        trainPoints.add(epoch, tLoss);
        valPoints.add(epoch, vLoss);
      }

    lossPlot.setPoints(trainPoints, "Train Loss");
    lossPlot.setPoints(valPoints, "Val Loss");

    threshold = myModel.totEpochs.size();    
    lossPlot.defaultDraw();
    lossPlot.drawLines();
  }
}
public void mouseClicked() {

  for (CheckBox cb : boxes)
    if (cb.overlaps(mouseX, mouseY) && !cb.cannotClick())
      cb.action();
}

public void mouseDragged() {

  if (!overlapsAnyCheckBox(mouseX, mouseY) && !myModel.isTesting())
    drawInBoard();
}

public void keyPressed() {

  if (keyCode == BACKSPACE) {
    board.beginDraw();
    board.clear();
    board.endDraw();
  } else if (keyCode == ENTER)
      showImageOnBoard(getRandomImage());
}
public void runGraphWindow() {

  graphs = new PlotWindow();
  String[] args = {graphs.getClass().getSimpleName()};
  runSketch(args, graphs);
}

public void trainModel() {

  myModel.train();
}

public void showCheckBoxes() {

  for (CheckBox cb : boxes)
    cb.show();
}

public void drawInBoard() {

  board.beginDraw();

  board.stroke(255, random(130));
  board.strokeWeight(60);

  board.line(pmouseX, pmouseY, mouseX, mouseY);

  board.endDraw();
}

public PImage getRandomImage() {

  return myModel.testData.getRandomImage();
}

public void showImageOnBoard(PImage img) {

  if (img != null) {
    board.beginDraw();
    img.resize(width, height);
    board.image(img, 0, 0);
    board.endDraw();
  }
}

public void showPrediction() {
  
  board.beginDraw();
  PImage img = board.get();
  board.endDraw();
  
  img.resize(w, h);
  Vector input = myModel.testData.dataSets[0].getInputVector(img);
  Vector guess = myModel.predict(input);
  
  int ansIndex = guess.indexMax();
  
  String guesses = "";
  for (int i = 0; i < myModel.testData.dataSetsCount; i++)
    guesses += myModel.testData.dataSets[i].name + ": " + round(guess.get(i) * 100) + "%    ";
    
  push();
  
  fill(255);
  textAlign(CORNER);
  textSize(20);
  
  text("Guess: " + guesses, 0, 30);
  text("Answer: " + myModel.testData.dataSets[ansIndex].name + ": " + round(guess.get(ansIndex) * 100) + "%", 0, 50);

  pop();  
}

public void showDataCount() {

  push();

  fill(255);
  textAlign(CORNER);

  text("Train data: " + myModel.trainData.getUsedData() + "/" + myModel.trainData.totalData, width - 250, height - 80);
  text("Test data: " + myModel.testData.getUsedData() + "/" + myModel.testData.totalData, width - 250, height - 60);

  pop();
}

public void testModel() {

  DataSet ds = myModel.testData.getRandomDataSet();

  PImage img = ds.getNextImage();

  Vector sample = ds.getInputVector(img);
  Vector label = new Vector(myModel.testData.dataSetsCount);

  label.set(ds.label, 1);

  globalTestWinTotal += myModel.testOnce(sample, label);

  img.resize(width, height);
  board.beginDraw();
  board.image(img, 0, 0);
  board.endDraw();

  push();

  textAlign(CORNER);
  fill(255);
  text("Test win rate: " + round(globalTestWinTotal * 100 * 100.0f / myModel.testData.getUsedData()) / 100.0f + "%", 450, height - 60);

  pop();
}

public void resetTesting() {

  myModel.testData.resetUsedData();
  globalTestWinTotal = 0;
}

public boolean overlapsAnyCheckBox(float x, float y) {

  for (CheckBox cb : boxes)
    if (cb.overlaps(x, y))
      return true;

  return false;
}

public void initCheckBoxes() {

  int w = 150;
  int h = 50;

  int x = 0;
  int y = height - h;

  prepModelToggle = new CheckBox(this, x, y, w, h, "Prepare model") {

    @Override public void action() {

      myModel.prepare();
    }

    @Override public boolean hasToClick() {

      return !myModel.isPrepared();
    }

    @Override public boolean isDone() {

      return myModel.isPrepared();
    }

    @Override public boolean cannotClick() {

      return myModel.isBusy();
    }
  };

  nextTrainDataToggle = new CheckBox(this, x + w, y, w, h, "Next training data") {

    @Override public void action() {

      myModel.nextTrainData(trainDataAmount);
    }

    @Override public boolean cannotClick() {

      return !myModel.trainData.hasToUpdateData;
    }

    @Override public boolean canClick() {

      return myModel.trainData.hasToUpdateData;
    }

    @Override public boolean hasToClick() {

      return !myModel.trainData.hasDataReady();
    }

    @Override public void tweak() {

      textSize = 15;
    }
  };

  trainToggle = new CheckBox(this, x + 2 * w, y, w, h, "Train model") {

    @Override public void action() {
      thread("trainModel");
    }

    @Override public boolean cannotClick() {

      return myModel.isBusy() || !myModel.trainData.hasDataReady() || myModel.isLoaded();
    }

    @Override public boolean canClick() {

      return myModel.trainData.hasDataReady();
    }

    @Override public boolean isDone() {

      return myModel.trainData.hasToUpdateData;
    }
  };

  testToggle = new CheckBox(this, x + 3 * w, y, w, h, "Test model") {

    @Override public void action() {

      myModel.testing = !myModel.testing;
    }

    @Override public boolean cannotClick() {

      return myModel.isTraining() || myModel.isPredicting() || !myModel.isPrepared();
    }

    @Override public boolean isDone() {

      return myModel.isTesting();
    }
  };

  predictBoardToggle = new CheckBox(this, x + 4 * w, y, w, h, "Predict board") {

    @Override public void action() {

      myModel.predicting = !myModel.predicting;
    }

    @Override public boolean cannotClick() {

      return myModel.isTraining() || myModel.isTesting() || !myModel.isPrepared();
    }

    @Override public boolean isDone() {

      return myModel.isPredicting();
    }
  };

  loadTrainedToggle = new CheckBox(this, x, y - h, w, h, "Load trained") {

    @Override public void action() {

      myModel.loadTrained();
    }

    @Override public boolean cannotClick() {

      return myModel.isBusy();
    }

    @Override public boolean isDone() {

      return myModel.isLoaded();
    }
  };

  resetTestingToggle = new CheckBox(this, x + 5 * w, y, w, h, "Reset test") {

    @Override public void action() {

      resetTesting();
    }

    @Override public boolean isDone() {

      return myModel.testData.getUsedData() == 0;
    }

    @Override public boolean cannotClick() {

      return myModel.isBusy();
    }
  };

  boxes = new CheckBox[]{prepModelToggle, nextTrainDataToggle, trainToggle, testToggle, predictBoardToggle, loadTrainedToggle, resetTestingToggle};
  for (CheckBox cb : boxes)
    cb.tweak();
}

public DataSet[] organizeMnistData(byte[] rawImages, byte[] rawLabels, int excessIm, int excessLab) {

  byte[] images = getRelevantBytes(rawImages, excessIm);
  byte[] labels = getRelevantBytes(rawLabels, excessLab);

  ArrayList<ArrayList<Byte>> organized = new ArrayList<ArrayList<Byte>>(10);

  for (int i = 0; i < 10; i++)
    organized.add(new ArrayList<Byte>());

  for (int i = 0; i < labels.length; i++)
    for (int j = 0; j < 10; j++)
      if (PApplet.parseInt(labels[i]) == j)
        for (int k = i * w * h; k < (i + 1) * w * h; k++)
          organized.get(j).add(images[k]);

  DataSet[] result = new DataSet[10];

  for (int i = 0; i < 10; i++) {

    ArrayList<Byte> digitsList = organized.get(i);
    byte[] digits = new byte[digitsList.size()];

    int j = 0;
    for (Byte b : digitsList)
      digits[j++] = b;

    result[i] = new DataSet(digits, i, 0, String.valueOf(i));
  }

  return result;
}

public byte[] getRelevantBytes(byte[] rawBytes, int excess) {
  byte[] result = new byte[rawBytes.length - excess];

  for (int i = excess; i < rawBytes.length; i++)
    result[i - excess] = rawBytes[i];

  return result;
}
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "mnist" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
