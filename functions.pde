void runGraphWindow() {

  graphs = new PlotWindow();
  String[] args = {graphs.getClass().getSimpleName()};
  runSketch(args, graphs);
}

void trainModel() {

  myModel.train();
}

void showCheckBoxes() {

  for (CheckBox cb : boxes)
    cb.show();
}

void drawInBoard() {

  board.beginDraw();

  board.stroke(255, random(130));
  board.strokeWeight(60);

  board.line(pmouseX, pmouseY, mouseX, mouseY);

  board.endDraw();
}

PImage getRandomImage() {

  return myModel.testData.getRandomImage();
}

void showImageOnBoard(PImage img) {

  if (img != null) {
    board.beginDraw();
    img.resize(width, height);
    board.image(img, 0, 0);
    board.endDraw();
  }
}

void showPrediction() {
  
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

void showDataCount() {

  push();

  fill(255);
  textAlign(CORNER);

  text("Train data: " + myModel.trainData.getUsedData() + "/" + myModel.trainData.totalData, width - 250, height - 80);
  text("Test data: " + myModel.testData.getUsedData() + "/" + myModel.testData.totalData, width - 250, height - 60);

  pop();
}

void testModel() {

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
  text("Test win rate: " + round(globalTestWinTotal * 100 * 100.0 / myModel.testData.getUsedData()) / 100.0 + "%", 450, height - 60);

  pop();
}

void resetTesting() {

  myModel.testData.resetUsedData();
  globalTestWinTotal = 0;
}

boolean overlapsAnyCheckBox(float x, float y) {

  for (CheckBox cb : boxes)
    if (cb.overlaps(x, y))
      return true;

  return false;
}

void initCheckBoxes() {

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

DataSet[] organizeMnistData(byte[] rawImages, byte[] rawLabels, int excessIm, int excessLab) {

  byte[] images = getRelevantBytes(rawImages, excessIm);
  byte[] labels = getRelevantBytes(rawLabels, excessLab);

  ArrayList<ArrayList<Byte>> organized = new ArrayList<ArrayList<Byte>>(10);

  for (int i = 0; i < 10; i++)
    organized.add(new ArrayList<Byte>());

  for (int i = 0; i < labels.length; i++)
    for (int j = 0; j < 10; j++)
      if (int(labels[i]) == j)
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

byte[] getRelevantBytes(byte[] rawBytes, int excess) {
  byte[] result = new byte[rawBytes.length - excess];

  for (int i = excess; i < rawBytes.length; i++)
    result[i - excess] = rawBytes[i];

  return result;
}
