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

				   
//============================================================
// Register functions in LUA
//

template<typename THReal>
vector<luaL_reg> libopencv24_Main__() {
  
  static const luaL_reg reg[] = {
    {"TH2CVImage", TH2CVImage<THReal>},
    {"CV2THImage", CV2THImage<THReal>},
    {NULL, NULL}  /* sentinel */
  };
  
  return vector<luaL_reg>(reg, reg+sizeof(reg)/sizeof(luaL_reg));
}
