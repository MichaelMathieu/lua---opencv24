#include "common.hpp"

void display(const Mat & im) {
  Mat tmp;
  im.convertTo(tmp, CV_8U);
  imshow("main", tmp);
  cvWaitKey(0);
}
