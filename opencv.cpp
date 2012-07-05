extern "C" {
#include <TH.h>
#include <luaT.h>
}

#include "THpp.hpp"

#include<opencv/cv.h>
#include<freak/freak.h>
#include "common.hpp"

using namespace TH;

vector<FREAK*> freaks_g;
static int CreateFREAK(lua_State* L) {
  setLuaState(L);
  bool  orientedNormalization = FromLuaStack<bool >(L, 1);
  bool  scaleNormalization    = FromLuaStack<bool >(L, 2);
  float patternSize           = FromLuaStack<float>(L, 3);
  int   nOctave               = FromLuaStack<int  >(L, 4);
  Tensor<int> trainedPairs    = FromLuaStack<Tensor<int> >(L, 5);
  
  vector<int> pairs;
  if (trainedPairs.nDimension() != 0)
    for (int i = 0; i < trainedPairs.size(0); ++i)
      pairs.push_back(trainedPairs(i));

  freaks_g.push_back(new FREAK(orientedNormalization, scaleNormalization,
			       patternSize, nOctave, pairs));
  PushOnLuaStack<int>(L, freaks_g.size()-1);
  return 1;
}

static int DeleteFREAK(lua_State* L) {
  setLuaState(L);
  int iFREAK = FromLuaStack<int  >(L, 1);
  delete freaks_g[iFREAK];
  return 0;
}

inline size_t HammingDistance(unsigned long long int* p1, unsigned long long int* p2,
			      size_t len) {
  size_t dist = 0;
  for (size_t i = 0; i < len; ++i) {
    dist += __builtin_popcountll(p1[i] ^ p2[i]);
  }
  return dist;
}

static int MatchFREAK(lua_State* L) {
  setLuaState(L);
  Tensor<unsigned char> descs1 = FromLuaStack<Tensor<unsigned char> >(L, 1);
  Tensor<unsigned char> descs2 = FromLuaStack<Tensor<unsigned char> >(L, 2);
  Tensor<long         > matches= FromLuaStack<Tensor<long         > >(L, 3);
  size_t threshold = FromLuaStack<size_t>(L, 4);

  descs1.newContiguous();
  descs2.newContiguous();
  unsigned char* descs1_p = descs1.data();
  unsigned char* descs2_p = descs2.data();
  const long* s1 = descs1.stride();
  const long* s2 = descs2.stride();
  matches.resize(descs1.size(0), 2);

  THassert(descs1.size(1) % sizeof(unsigned long long int) == 0);
  size_t bestj, bestdist, dist;
  long iMatches = 0;
  for (long i = 0; i < descs1.size(0); ++i) {
    bestj = 0;
    bestdist = descs1.size(1)*8;
    for (long j = i; j < descs2.size(0); ++j) {
      dist = HammingDistance((unsigned long long*)(descs1_p + i*s1[0]),
			     (unsigned long long*)(descs2_p + j*s2[0]),
			     descs1.size(1)/sizeof(unsigned long long));
      if (dist < bestdist) {
	bestj = j;
	bestdist = dist;
      }
    }
    if (bestdist < threshold) {
      matches(iMatches, 0) = i;
      matches(iMatches, 1) = bestj;
      ++iMatches;
    }
  }
  PushOnLuaStack<int>(L, iMatches);
  return 1;
}

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define libopencv24_(NAME) TH_CONCAT_3(libopencv24_, Real, NAME)

static const luaL_reg libopencv24_init [] =
  {
    {"CreateFREAK", CreateFREAK},
    {"DeleteFREAK", DeleteFREAK},
    {"MatchFREAK", MatchFREAK},
    {NULL, NULL}
  };

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "genericpp/opencv.cpp"

#include "generic/opencv.cpp"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libopencv24(lua_State *L)
{
  luaL_register(L, "libopencv24", libopencv24_init);
  
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  libopencv24_FloatMain_init(L);
  libopencv24_DoubleMain_init(L);
  
  return 1;
}
