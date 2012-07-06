#include "THpp.hpp"

#include<opencv/cv.h>
#include<freak/freak.h>
#include "common.hpp"

using namespace TH;

//============================================================
// Image conversions
//

template<typename THReal>
static int TH2CVImage(lua_State* L) {
  setLuaState(L);
  Tensor<THReal> im   = FromLuaStack<Tensor<THReal> >(1);
  Tensor<ubyte>  imcv = FromLuaStack<Tensor<ubyte > >(2);

  long h = im.size(1), w = im.size(2);
  long i, j, k;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j)
      for (k = 0; k < 3; ++k)
	imcv(i,j,k) = im(2-k,i,j)*(THReal)255.;

  return 0;
}

template<typename THReal>
static int CV2THImage(lua_State* L) {
  setLuaState(L);
  Tensor<THReal> imcv = FromLuaStack<Tensor<THReal> >(1);
  Tensor<ubyte>  im   = FromLuaStack<Tensor<ubyte > >(2);

  long h = im.size(1), w = im.size(2);
  long i, j, k;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j)
      for (k = 0; k < 3; ++k)
	im(k,i,j) = ((THReal)im(i,j,2-k))/(THReal)255.;

  return 0;
}

//============================================================
// FREAK
//

extern std::vector<FREAK*> freaks_g;

template<typename THReal>
static int ComputeFREAK(lua_State* L) {
  setLuaState(L);
  Tensor<THReal>        im        = FromLuaStack<Tensor<THReal> >(1);
  Tensor<unsigned char> descs     = FromLuaStack<Tensor<unsigned char> >(2);
  Tensor<float>         positions = FromLuaStack<Tensor<float> >(3);
  float       keypoints_threshold = FromLuaStack<float>(4);
  int                   iFREAK    = FromLuaStack<int>(5);

  mat3b im_cv = TensorToMat3b(im);
  matb im_cv_gray;
  cvtColor(im_cv, im_cv_gray, CV_BGR2GRAY);

  // keypoints
  vector<KeyPoint> keypoints;
  FAST(im_cv_gray, keypoints, keypoints_threshold, true);
  
  // descriptors
  FREAK & freak = *(freaks_g[iFREAK]);
  Mat descs_cv;
  freak.compute(im_cv_gray, keypoints, descs_cv);
  
  // output
  positions.resize(keypoints.size(), 4);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const KeyPoint & kpt = keypoints[i];
    positions(i, 0) = kpt.pt.x;
    positions(i, 1) = kpt.pt.y;
    positions(i, 2) = kpt.size;
    positions(i, 3) = kpt.angle;
  }
  descs.resize(descs_cv.size().height, descs_cv.size().width);
  descs_cv.copyTo(TensorToMat(descs));
  
  return 0;
}

template<typename THReal>
static int TrainFREAK(lua_State* L) {
  setLuaState(L);
  vector<Tensor<THReal> > images = FromLuaStack<vector<Tensor<THReal> > >(1);
  Tensor<int> pairs_out = FromLuaStack<Tensor<int> >(2);
  size_t iFREAK = FromLuaStack<size_t>(3);
  float keypoints_threshold   = FromLuaStack<float>(4);
  double corrThres = FromLuaStack<double>(5);

  vector<Mat> images_cv;
  vector<vector<KeyPoint> > keypoints;
  for (size_t i = 0; i < images.size(); ++i) {
    Mat im_gray;
    cvtColor(TensorToMat3b(images[i]), im_gray, CV_BGR2GRAY);
    images_cv.push_back(im_gray);
    keypoints.push_back(vector<KeyPoint>());
    FAST(im_gray, keypoints.back(), keypoints_threshold, true);
  }

  FREAK & freak = *(freaks_g[iFREAK]);
  vector<int> pairs = freak.selectPairs(images_cv, keypoints, corrThres, false);

  pairs_out.resize(pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i)
    pairs_out(i) = pairs[i];

  return 0;
}
				   
//============================================================
// Register functions in LUA
//

template<typename THReal>
vector<luaL_reg> libopencv24_Main__() {
  
  static const luaL_reg reg[] = {
    {"TH2CVImage", TH2CVImage<THReal>},
    {"CV2THImage", CV2THImage<THReal>},
    {"ComputeFREAK", ComputeFREAK<THReal>},
    {"TrainFREAK", TrainFREAK<THReal>},
    {NULL, NULL}  /* sentinel */
  };
  
  return vector<luaL_reg>(reg, reg+sizeof(reg)/sizeof(luaL_reg));
}
