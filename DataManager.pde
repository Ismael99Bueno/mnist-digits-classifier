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

  void prepareSets(int size) {
    
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
        smallLSet[j].set(label, 1.0);
        
        int index = i * size + j;

        sampleSet[index] = smallSSet[j];
        labelSet[index] = smallLSet[j];
      }
    }
  }
  
  Vector[] getSampleSet() {
   
    return sampleSet;
  }
  
  Vector[] getLabelSet() {
   
    return labelSet;
  }
  
  Vector[] getRandomSample() {
    
    DataSet randomSet = dataSets[rand.nextInt(dataSetsCount)];
    
    Vector label = new Vector(dataSetsCount);
    label.set(randomSet.label, 1);
    Vector sample = randomSet.getInputVector(rand.nextInt(dataSetsCount));
    
    return new Vector[] {sample, label};
  }
  
  PImage getRandomImage() {
   
    DataSet randomSet = dataSets[rand.nextInt(dataSetsCount)];
    
    return randomSet.getImage(rand.nextInt(randomSet.totalData));
  }
  
  DataSet getRandomDataSet() {
   
    return dataSets[rand.nextInt(dataSetsCount)];
  }
  
  boolean hasDataReady() {
   
    return dataReady;
  }
  
  int getUsedData() {
   
    int total = 0;
    for (DataSet ds : dataSets)
      total += ds.usedData;
      
    return total;
  }
  
  void resetUsedData() {
   
    for (DataSet ds : dataSets)
      ds.resetUsedData();
  }
}
