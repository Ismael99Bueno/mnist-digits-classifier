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

  byte[] getRelevantBytes(byte[] rawBytes, int excess) {
    byte[] result = new byte[rawBytes.length - excess];

    for (int i = excess; i < rawBytes.length; i++)
      result[i - excess] = rawBytes[i];

    return result;
  }
  
  Vector[] getNextData(int size) {
   
     Vector[] sampleSet = new Vector[size];
     for (int i = 0; i < size; i++) {
       
       int j = i + usedData;
       sampleSet[i] = getInputVector(j);
     }
     
     usedData += size;
     return sampleSet;
  }
  
  PImage getNextImage() {
   
    return getImage(usedData++);
  }
  
  Vector getInputVector(int index) {
   
    return getInputVector(getImage(index));
  }
  
  Vector getInputVector(PImage img) {
    
    Vector result = new Vector(w * h);
    img.loadPixels();
    
    for (int i = 0; i < w * h; i++) {
     
      float normalized = map(img.pixels[i], color(0), color(255), 0, 1);
      result.set(i, normalized);
    }
    
    img.updatePixels();
    return result;
  }

  PImage getImage(int index) {

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
  
  void resetUsedData() {
   
    usedData = 0;
  }
}
