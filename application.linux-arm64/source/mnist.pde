import tensors.Float.*;
import deepLearning.utilities.*;
import grafica.*;
import checkBox.*;
import java.util.Random;

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

void settings() {
  
  DataSet[] mnistTrainData = organizeMnistData(loadBytes("train-images.idx3-ubyte"), loadBytes("train-labels.idx1-ubyte"), 16, 8);
  DataSet[] mnistTestData = organizeMnistData(loadBytes("test-images.idx3-ubyte"), loadBytes("test-labels.idx1-ubyte"), 16, 8);

  mnistTrain = new DataManager(mnistTrainData);
  mnistTest = new DataManager(mnistTestData);

  myModel = new Network(mnistTrain, mnistTest);

  runGraphWindow();
  size(1008, 800);
}

void setup() {

  initCheckBoxes();

  board = createGraphics(width, height);
}

void draw() {

  background(0);
  image(board, 0, 0);

  if (myModel.isTesting())
    testModel();
  else if (myModel.isPredicting())
    showPrediction();
    
  showDataCount();
  showCheckBoxes();
}
