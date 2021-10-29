void mouseClicked() {

  for (CheckBox cb : boxes)
    if (cb.overlaps(mouseX, mouseY) && !cb.cannotClick())
      cb.action();
}

void mouseDragged() {

  if (!overlapsAnyCheckBox(mouseX, mouseY) && !myModel.isTesting())
    drawInBoard();
}

void keyPressed() {

  if (keyCode == BACKSPACE) {
    board.beginDraw();
    board.clear();
    board.endDraw();
  } else if (keyCode == ENTER)
      showImageOnBoard(getRandomImage());
}
