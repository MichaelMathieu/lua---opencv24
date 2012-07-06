#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.cpp"
#else

//======================================================================
// File: opencv.cpp
//
// Description: OpenCV 2.4 bindings
//
// Created: July 5th, 2012
//
// Author: Michael Mathieu // michael.mathieu@ens.fr
//======================================================================

#include "THpp.hpp"

#include<opencv/cv.h>
#include "common.hpp"

using namespace TH;

//============================================================
// Image conversions
//

static int libopencv24_(TH2CVImage)(lua_State* L) {
  setLuaState(L);
  Tensor<real> im   = FromLuaStack<Tensor<real> >(1);
  Tensor<ubyte>  imcv = FromLuaStack<Tensor<ubyte > >(2);

  long h = im.size(1), w = im.size(2);
  long i, j, k;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j)
      for (k = 0; k < 3; ++k)
	imcv(i,j,k) = im(2-k,i,j)*(real)255.;

  return 0;
}

static int libopencv24_(CV2THImage)(lua_State* L) {
  setLuaState(L);
  Tensor<real> imcv = FromLuaStack<Tensor<real> >(1);
  Tensor<ubyte>  im   = FromLuaStack<Tensor<ubyte > >(2);

  long h = im.size(1), w = im.size(2);
  long i, j, k;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j)
      for (k = 0; k < 3; ++k)
	im(k,i,j) = ((real)im(i,j,2-k))/(real)255.;

  return 0;
}

//============================================================
// Dense Optical Flow
//

static int libopencv24_(DenseOpticalFlow)(lua_State *L) {
  setLuaState(L);
  Tensor<ubyte> im1  = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<ubyte> im2  = FromLuaStack<Tensor<ubyte> >(2);
  Tensor<real>  flow = FromLuaStack<Tensor<real> >(L, 3);
  
  matb im1_cv_gray, im2_cv_gray;
  if (im1.nDimension() == 3) { //color images
    cvtColor(TensorToMat3b(im1), im1_cv_gray, CV_BGR2GRAY);
    cvtColor(TensorToMat3b(im2), im2_cv_gray, CV_BGR2GRAY);
  } else {
    im1_cv_gray = TensorToMat(im1);
    im2_cv_gray = TensorToMat(im2);
  }
  int h = im1_cv_gray.size().height, w = im1_cv_gray.size().width;

  flow.resize(h, w, 2);
#ifdef TH_REAL_IS_FLOAT  
  Mat flow_cv = TensorToMat(flow);
#else
  Mat flow_cv(h, w, CV_32FC2);
#endif

  calcOpticalFlowFarneback(im1_cv_gray, im2_cv_gray, flow_cv, 0.5, 5, 11, 10, 5, 1.1, 0);

#ifndef TH_REAL_IS_FLOAT
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      *(Vec2f*)(&(flow(i, j, 0))) = flow_cv.at<Vec2f>(i, j);
#endif
  
  return 0;
}

//============================================================
// Register functions in LUA
//

static const luaL_reg libopencv24_(Main__) [] = {
  {"TH2CVImage", libopencv24_(TH2CVImage)},
  {"CV2THImage", libopencv24_(CV2THImage)},
  {"DenseOpticalFlow", libopencv24_(DenseOpticalFlow)},
  {NULL, NULL}  /* sentinel */
};

LUA_EXTERNC DLL_EXPORT int libopencv24_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libopencv24_(Main__), "libopencv24");
  return 1;
}

#endif
