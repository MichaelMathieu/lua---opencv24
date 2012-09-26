extern "C" {
#include <TH.h>
#include <luaT.h>
}

#include "THpp.hpp"

#include<opencv/cv.h>
#include "common.hpp"

using namespace TH;

//============================================================
// Tracking
//

static int TrackPoints(lua_State* L) {
  setLuaState(L);
  Tensor<ubyte> im1          = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<ubyte> im2          = FromLuaStack<Tensor<ubyte> >(2);
  Tensor<float> corresps     = FromLuaStack<Tensor<float> >(3);
  size_t        maxCorners   = FromLuaStack<size_t>        (4);
  float         qualityLevel = FromLuaStack<float>         (5);
  float         minDistance  = FromLuaStack<float>         (6);
  int           blockSize    = FromLuaStack<int>           (7);
  int           winSize      = FromLuaStack<int>           (8);
  int           maxLevel     = FromLuaStack<int>           (9);
  bool          useHarris    = FromLuaStack<bool>          (10);
  
  Mat im1_cv, im2_cv, im1_cv_gray;
  if (im1.nDimension() == 3) { //color images
    im1_cv = TensorToMat3b(im1);
    im2_cv = TensorToMat3b(im2);
    cvtColor(im1_cv, im1_cv_gray, CV_BGR2GRAY);
  } else {
    im1_cv = TensorToMat(im1);
    im2_cv = TensorToMat(im2);
    im1_cv_gray = im1_cv;
  }
  matf corresps_cv = TensorToMat(corresps);

  const Size winSize2(winSize, winSize);
  const TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 0.1);
  vector<Point2f> points1, points2;
  vector<ubyte> status;
  vector<float> err;
  Mat mask;
  goodFeaturesToTrack(im1_cv_gray, points1, maxCorners, qualityLevel, minDistance,
		      mask, blockSize, useHarris, 0.04f);
  calcOpticalFlowPyrLK(im1_cv, im2_cv, points1, points2, status, err, winSize2,
		       maxLevel, criteria, 0, 0);

  size_t i, iCorresps = 0;
  for (i = 0; i < points2.size(); ++i)
    if (status[i]) {
      corresps(iCorresps, 0) = points1[i].x;
      corresps(iCorresps, 1) = points1[i].y;
      corresps(iCorresps, 2) = points2[i].x;
      corresps(iCorresps, 3) = points2[i].y;
      ++iCorresps;
    }
  THFloatTensor_narrow(corresps, corresps, 0, 0, iCorresps);
  
  return 0;
}

//============================================================
// FREAK
//

vector<FREAK*> freaks_g;

static int CreateFREAK(lua_State* L) {
  setLuaState(L);
  bool  orientedNormalization = FromLuaStack<bool >(1);
  bool  scaleNormalization    = FromLuaStack<bool >(2);
  float patternSize           = FromLuaStack<float>(3);
  int   nOctave               = FromLuaStack<int  >(4);
  Tensor<int> trainedPairs    = FromLuaStack<Tensor<int> >(5);
  
  vector<int> pairs;
  if (trainedPairs.nDimension() != 0)
    for (int i = 0; i < trainedPairs.size(0); ++i)
      pairs.push_back(trainedPairs(i));

  freaks_g.push_back(new FREAK(orientedNormalization, scaleNormalization,
			       patternSize, nOctave, pairs));
  PushOnLuaStack<int>(freaks_g.size()-1);
  return 1;
}

static int DeleteFREAK(lua_State* L) {
  setLuaState(L);
  int iFREAK = FromLuaStack<int  >(1);
  delete freaks_g[iFREAK];
  return 0;
}

static int ComputeFREAKfromKeyPoints(lua_State* L){
  setLuaState(L);
  Tensor<ubyte>         im        = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<unsigned char> descs     = FromLuaStack<Tensor<unsigned char> >(2);
  Tensor<float>         positions = FromLuaStack<Tensor<float> >(3);
  int                   iFREAK    = FromLuaStack<int>(5);

  matb im_cv_gray;
  if (im.nDimension() == 3) //color images
    cvtColor(TensorToMat3b(im), im_cv_gray, CV_BGR2GRAY);
  else
    im_cv_gray = TensorToMat(im);

  cout << positions.size() << endl;
  
  vector<KeyPoint> keypoints (positions.size()[0]);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    KeyPoint & kpt = keypoints[i];
    kpt.pt.x = positions(i,0);
    kpt.pt.y = positions(i,1);
  }
  Mat descs_cv;
  FREAK & freak = *(freaks_g[iFREAK]);
  freak.compute(im_cv_gray, keypoints, descs_cv);
  
  descs.resize(descs_cv.size().height, descs_cv.size().width);
  descs_cv.copyTo(TensorToMat(descs));
  
  return 0; 
}

static int ComputeFREAK(lua_State* L) {
  setLuaState(L);
  Tensor<ubyte>         im        = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<unsigned char> descs     = FromLuaStack<Tensor<unsigned char> >(2);
  Tensor<float>         positions = FromLuaStack<Tensor<float> >(3);
  float       keypoints_threshold = FromLuaStack<float>(4);
  int                   iFREAK    = FromLuaStack<int>(5);

  matb im_cv_gray;
  if (im.nDimension() == 3) //color images
    cvtColor(TensorToMat3b(im), im_cv_gray, CV_BGR2GRAY);
  else
    im_cv_gray = TensorToMat(im);

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

static int TrainFREAK(lua_State* L) {
  setLuaState(L);
  vector<Tensor<ubyte> > images  = FromLuaStack<vector<Tensor<ubyte> > >(1);
  Tensor<int> pairs_out           = FromLuaStack<Tensor<int> >(2);
  size_t      iFREAK              = FromLuaStack<size_t>(3);
  float       keypoints_threshold = FromLuaStack<float>(4);
  double      corrThres           = FromLuaStack<double>(5);

  vector<Mat> images_cv;
  vector<vector<KeyPoint> > keypoints;
  for (size_t i = 0; i < images.size(); ++i) {
    matb im_gray;
    if (images[i].nDimension() == 3)
      cvtColor(TensorToMat3b(images[i]), im_gray, CV_BGR2GRAY);
    else
      im_gray = TensorToMat(images[i]);
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

// Just compute the FAST keypoints
static int ComputeFAST(lua_State* L) {
  setLuaState(L);
  Tensor<ubyte>         im        = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<float>         positions = FromLuaStack<Tensor<float> >(2);
  float       keypoints_threshold = FromLuaStack<float>(3);

  matb im_cv_gray;
  if (im.nDimension() == 3) //color images
    cvtColor(TensorToMat3b(im), im_cv_gray, CV_BGR2GRAY);
  else
    im_cv_gray = TensorToMat(im);

  // keypoints
  vector<KeyPoint> keypoints;
  FAST(im_cv_gray, keypoints, keypoints_threshold, true);
  
  // output
  positions.resize(keypoints.size(), 5);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const KeyPoint & kpt = keypoints[i];
    positions(i, 0) = kpt.pt.x;
    positions(i, 1) = kpt.pt.y;
    positions(i, 2) = kpt.size;
    positions(i, 3) = kpt.angle;
    positions(i, 4) = kpt.response;
  }
  
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
  Tensor<unsigned char> descs1 = FromLuaStack<Tensor<unsigned char> >(1);
  Tensor<unsigned char> descs2 = FromLuaStack<Tensor<unsigned char> >(2);
  Tensor<long         > matches= FromLuaStack<Tensor<long         > >(3);
  size_t threshold = FromLuaStack<size_t>(4);

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
  PushOnLuaStack<int>(iMatches);
  return 1;
}


static int version (lua_State* L) {
  printf("%d.%d.%d\n",
         CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
  return 0;
}

//============================================================
// Register functions in LUA
//


#define torch_(NAME)        TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libopencv24_(NAME)  TH_CONCAT_3(libopencv24_, Real, NAME)

static const luaL_reg libopencv24_init [] =
  {
    {"TrackPoints",  TrackPoints},
    {"CreateFREAK",  CreateFREAK},
    {"DeleteFREAK",  DeleteFREAK},
    {"ComputeFREAK", ComputeFREAK},
    {"ComputeFREAKfromKeyPoints", ComputeFREAKfromKeyPoints},
    {"TrainFREAK",   TrainFREAK},
    {"MatchFREAK",   MatchFREAK},
    {"ComputeFAST",  ComputeFAST}, 
    {"Version",      version},
    {NULL, NULL}
  };


#include "generic/opencv.cpp"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libopencv24(lua_State *L)
{
  luaL_register(L, "libopencv24", libopencv24_init);
  
  libopencv24_FloatMain_init(L);
  libopencv24_DoubleMain_init(L);
  
  return 1;
}
