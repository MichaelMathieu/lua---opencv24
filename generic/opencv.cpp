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
#include "opencv2/nonfree/features2d.hpp"
#include "common.hpp"

using namespace TH;

//============================================================
// Image conversions
//

static int libopencv24_(TH2CVImage)(lua_State* L) {
  setLuaState(L);
  Tensor<real > im   = FromLuaStack<Tensor<real > >(1);
  Tensor<ubyte> imcv = FromLuaStack<Tensor<ubyte> >(2);

  im.newContiguous();

  if (im.nDimension() == 2) {
    long h = im.size(0), w = im.size(1);
    imcv.resize(h, w);
    Mat im_cv = TensorToMat(imcv);
    TensorToMat(im).convertTo(im_cv, CV_8U, 255., 0.);
  } else {
    long h = im.size(1), w = im.size(2);
    imcv.resize(h, w, 3);
    
    long i, j, k;
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j)
	for (k = 0; k < 3; ++k)
	  imcv(i,j,k) = (ubyte)(im(2-k,i,j)*(real)255. + (real)(0.5));
  }

  return 0;
}

static int libopencv24_(CV2THImage)(lua_State* L) {
  setLuaState(L);
  Tensor<ubyte> imcv = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<real > im   = FromLuaStack<Tensor<real > >(2);

  imcv.newContiguous();

  long h = imcv.size(0), w = imcv.size(1);
  if (imcv.nDimension() == 2) {
    im.resize(h, w);
    Mat im_cv = TensorToMat(im);
    TensorToMat(imcv).convertTo(im_cv, DataType<real>::type, 1./255.);
  } else {    
    im.resize(3, h, w);
    long i, j, k;
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j)
	for (k = 0; k < 3; ++k)
	  im(k,i,j) = ((real)imcv(i,j,2-k))/(real)255.;
  }

  return 0;
}

//============================================================
// Dense Optical Flow
//

static int libopencv24_(DenseOpticalFlow)(lua_State *L) {
  setLuaState(L);
  Tensor<ubyte> im1  = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<ubyte> im2  = FromLuaStack<Tensor<ubyte> >(2);
  Tensor<real>  flow = FromLuaStack<Tensor<real > >(3);
  double pyr_scale   = FromLuaStack<double>(4);
  int    levels      = FromLuaStack<int   >(5);
  int    winsize     = FromLuaStack<int   >(6);
  int    iterations  = FromLuaStack<int   >(7);
  int    poly_n      = FromLuaStack<int   >(8);
  double poly_sigma  = FromLuaStack<double>(9);
  
  matb im1_cv_gray, im2_cv_gray;
  if (im1.nDimension() == 3) { //color images
    cvtColor(TensorToMat3b(im1), im1_cv_gray, CV_BGR2GRAY);
    cvtColor(TensorToMat3b(im2), im2_cv_gray, CV_BGR2GRAY);
  } else {
    im1_cv_gray = TensorToMat(im1);
    im2_cv_gray = TensorToMat(im2);
  }

#ifdef TH_REAL_IS_FLOAT  
  Mat flow_cv = TensorToMat(flow);
#else
  int h = im1_cv_gray.size().height, w = im1_cv_gray.size().width;
  Mat flow_cv(h, w, CV_32FC2);
#endif

  calcOpticalFlowFarneback(im1_cv_gray, im2_cv_gray, flow_cv, pyr_scale, levels, winsize,
			   iterations, poly_n, poly_sigma, 0);

#ifndef TH_REAL_IS_FLOAT
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      *(Vec2f*)(&(flow(i, j, 0))) = flow_cv.at<Vec2f>(i, j);
#endif
  
  return 0;
}

//============================================================
// Detect Extract
// 
// -- use opencv functions to detect using goodfeatures (or Canny) and
// extract a sparse set of features at those detected locations.
//


static int libopencv24_(DetectExtract)(lua_State *L) {
  setLuaState(L);
  Tensor<ubyte> img          = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<ubyte> mask         = FromLuaStack<Tensor<ubyte> >(2);
  Tensor<real>  positions    = FromLuaStack<Tensor<real>  >(3); 
  Tensor<real>  feat         = FromLuaStack<Tensor<real > >(4);
  const char *  dtype        = lua_tostring(L,5);
  const char *  etype        = lua_tostring(L,6);
  size_t        maxPoints    = FromLuaStack<size_t>        (7);

  string detectorType(dtype);
  string extractorType(etype);
  int h,w;

  matb img_cv_gray, mask_cv;
  if (img.nDimension() == 3) { //color images
    cvtColor(TensorToMat3b(img), img_cv_gray, CV_BGR2GRAY);
  } else {
    img_cv_gray = TensorToMat(img);
  }
  h = img_cv_gray.size().height;
  w = img_cv_gray.size().width;
  
  if ((mask.nDimension() == 2) &&
      (mask.size(0) == h) && (mask.size(1) == w)) {
      mask_cv = TensorToMat(mask);
  } else {
    mask_cv = Mat();
  }

  Mat feat_cv;
// #ifdef TH_REAL_IS_FLOAT  
//   Mat feat_cv = TensorToMat(feat);
// #else
//   Mat feat_cv(maxPoints, 128, CV_32FC2);
// #endif

  vector<KeyPoint>         keyPoints;
  Ptr<FeatureDetector>     detector;
  Ptr<DescriptorExtractor> extractor;
  KeyPointsFilter          kpFilt;
  
  // detecting keypoints
  // "FAST", "STAR", "SIFT", "SURF", "ORB",
  // "MSER", "GFTT", "HARRIS", "Dense", "SimpleBlob",
  // Also combined format: 
  // "Grid" – GridAdaptedFeatureDetector,
  // "Pyramid" – PyramidAdaptedFeatureDetector )
  // for example: "GridFAST", "PyramidSTAR" .

  detector = FeatureDetector::create(detectorType);
  detector->detect(img_cv_gray,keyPoints,mask_cv);

  if (keyPoints.size() < 1){
    THError("No KeyPoints Found.");
  }
  // Sort the keypoints, return only the top maxPoints
  /*
  kpFilt.retainBest(keyPoints, maxPoints);
  */
  sort(keyPoints.begin(), keyPoints.end(), keyPointCompare());
  vector<KeyPoint>::const_iterator first = keyPoints.begin();
  vector<KeyPoint>::const_iterator last  =
    keyPoints.begin() + min(keyPoints.size(), maxPoints);
  vector<KeyPoint> topKeyPoints(first, last);
  
  
  // computing descriptors

  /*
    The create() function does not seem to work.  The features aren't
    computed properly so I have replaced it with the messier if else
    cases below.
    
    extractor = DescriptorExtractor::create(extractorType);
  */
  cout << "Extractor Type: " << extractorType << endl;
  if (extractorType.compare("SURF") == 0) {
    extractor = new SurfDescriptorExtractor; 
  } else if (extractorType.compare("SIFT") == 0) { 
    extractor = new SiftDescriptorExtractor;
  } else if (extractorType.compare("BRIEF") == 0) { 
    extractor = new BriefDescriptorExtractor;
  } else if (extractorType.compare("ORB") == 0) { 
    extractor = new OrbDescriptorExtractor;
  } else {
    printf("Warning unrecognized DescriptorExtractor (%s) using SURF\n",
           etype);
    extractor = new SurfDescriptorExtractor;
  }
  /*
    } else if (extractorType.compare("OpponentSIFT") == 0) { 
    extractor = new OpponentSiftDescriptorExtractor;
    } else if (extractorType.compare("BOW") == 0) { 
    extractor = new BOWImgDescriptorExtractor;
    } else if (extractorType.compare("FREAK") == 0) { 
    extractor = new FreakDescriptorExtractor;
  */

  extractor->compute(img_cv_gray, topKeyPoints, feat_cv);
  cout << "Features: " << feat_cv.rows << " x " << feat_cv.cols <<endl;

  feat.resize(feat_cv.rows,feat_cv.cols);
  positions.resize(maxPoints, 2);

  for (size_t i = 0; i < maxPoints; ++i) {
    const KeyPoint & kpt = topKeyPoints[i];
    positions(i, 0) = kpt.pt.x;
    positions(i, 1) = kpt.pt.y;
  }

  // DEBUG
  for(int i = 0; i < 1; i++){ //feat_cv.rows; i++){
    for(int j = 0; j < feat_cv.cols; j++){
      printf("%f, ",feat_cv.at<real>(i,j));
    }
    printf("\n");
  }
#ifndef TH_REAL_IS_FLOAT
  for (int i = 0; i < feat_cv.rows; ++i)
    for (int j = 0; j < feat_cv.cols; ++j)
      *(Vec2f*)(&(feat(i, j, 0))) = feat_cv.at<Vec2f>(i, j);
#endif
 
  return 0;
}

static int libopencv24_(CornerHarris)(lua_State *L) {
  setLuaState(L);
  Tensor<ubyte> src  = FromLuaStack<Tensor<ubyte> >(1);
  Tensor<real>  dst  = FromLuaStack<Tensor<real > >(2);
  int     blocksize  = FromLuaStack<int   >(3);
  int         ksize  = FromLuaStack<int   >(4);
  double          k  = FromLuaStack<double>(5);
  int    borderType  = BORDER_REPLICATE;
  
  matb src_cv_gray;
  if (src.nDimension() == 3) { //color images
    cvtColor(TensorToMat3b(src), src_cv_gray, CV_BGR2GRAY);
  } else {
    src_cv_gray = TensorToMat(src);
  }

#ifdef TH_REAL_IS_FLOAT  
  Mat dst_cv = TensorToMat(dst);
#else
  int h = src_cv_gray.size().height, w = src_cv_gray.size().width;
  Mat dst_cv(h, w, CV_32FC2);
#endif

  /* cornerHarris(InputArray src,
     OutputArray dst,
     int blockSize,
     int ksize, double k, int borderType=BORDER_DEFAULT ) */
  cornerHarris(src_cv_gray,dst_cv,blocksize,ksize,k,borderType);

#ifndef TH_REAL_IS_FLOAT
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      *(Vec2f*)(&(dst(i, j, 0))) = dst_cv.at<Vec2f>(i, j);
#endif

  return 0;
}
//============================================================
// Register functions in LUA
//

static const luaL_reg libopencv24_(Main__) [] = {
  {"TH2CVImage",       libopencv24_(TH2CVImage)},
  {"CV2THImage",       libopencv24_(CV2THImage)},
  {"DenseOpticalFlow", libopencv24_(DenseOpticalFlow)},
  {"DetectExtract",    libopencv24_(DetectExtract)},
  {"CornerHarris",     libopencv24_(CornerHarris)},
  {NULL, NULL}  /* sentinel */
};

LUA_EXTERNC DLL_EXPORT int libopencv24_(Main_init) (lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, libopencv24_(Main__), "libopencv24");
  lua_pop(L,1);
  return 1;
}

#endif
