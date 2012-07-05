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

LUA_EXTERNC DLL_EXPORT int libopencv24_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libopencv24_Main__<real>().data(), "libopencv24");
  return 1;
}

#endif
