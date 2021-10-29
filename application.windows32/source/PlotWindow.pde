class PlotWindow extends PApplet {
 
  GPointsArray trainPoints;
  GPointsArray valPoints;

  void settings() {
    size(640, 360);
  }

  void setup() {

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
  void draw() {

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
