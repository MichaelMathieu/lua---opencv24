#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.cpp"
#else

//======================================================================
// File: opencv.cpp
//
// Description: OpenCV 2.4 bindings
//
// Created: April 28th, 2012
//
// Author: Michael Mathieu // michael.mathieu@ens.fr
//======================================================================

#include "THpp.hpp"

#include<opencv/cv.h>
#include<freak/freak.h>
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
// Register functions in LUA
//

static const luaL_reg libopencv24_(Main__) [] = {
  {"TH2CVImage", libopencv24_(TH2CVImage)},
  {"CV2THImage", libopencv24_(CV2THImage)},
  {NULL, NULL}  /* sentinel */
};

LUA_EXTERNC DLL_EXPORT int libopencv24_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libopencv24_(Main__), "libopencv24");
  return 1;
}

#endif
