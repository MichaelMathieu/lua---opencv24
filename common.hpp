#ifndef __COMMON_H__
#define __COMMON_H__

#include<cstdio>
#include<cmath>
#include<cassert>
#include<iostream>
#include<vector>
#include<string>
using namespace std;

#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

#include "THpp.hpp"

#if CV_MAJOR_VERSION != 2
#error OpenCV version must be 2.x.x
#endif
#if CV_MINOR_VERSION < 4
#error OpenCV version must be >= 2.4.0
#endif

typedef Mat_<float> matf;
typedef Mat_<double> matd;
typedef matf matr;
typedef Mat_<unsigned char> matb;
typedef Mat_<Vec3b> mat3b;
typedef unsigned char ubyte;

template<typename T, typename T2> inline bool epsEqual(T a, T2 b, double eps = 0.01) {
  return (a-eps < b) && (b < a+eps);
}

void display(const Mat & im);

template<typename Treal>
mat3b TensorToMat3b(const TH::Tensor<Treal> & im) {
  if (im.size(0) == 3) {
    long h = im.size(1);
    long w = im.size(2);
    const long* is = im.stride();
    const Treal* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*2+is[1]*i+is[2]*j]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]  +is[1]*i+is[2]*j]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*0+is[1]*i+is[2]*j]*255.)));
    return ret;
  } else if (im.size(2) == 3) {
    long h = im.size(0);
    long w = im.size(1);
    const long* is = im.stride();
    const Treal* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]*2]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]  ]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]*0]*255.)));
    return ret;
  } else {
    THerror("TensorToMat3b: tensor must be 3xHxW or HxWx3");
  }
  return mat3b(0,0); //remove warning
}

// byte case : we may not have to copy
template<>
mat3b TensorToMat3b<ubyte>(const TH::Tensor<ubyte> & im) {
  if (im.size(0) == 3) {
    long h = im.size(1);
    long w = im.size(2);
    const long* is = im.stride();
    const ubyte* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(im_p[is[0]*2+is[1]*i+is[2]*j],
		       im_p[is[0]  +is[1]*i+is[2]*j],
		       im_p[is[0]*0+is[1]*i+is[2]*j]);
    return ret;
  } else if (im.size(2) == 3) {
    return mat3b(im.size(0), im.size(1), (Vec3b*)im.data());
  } else {
    THerror("TensorToMat3b: tensor must be 3xHxW or HxWx3");
  }
  return mat3b(0,0); //remove warning
}

template<typename Treal>
Mat TensorToMat(TH::Tensor<Treal> & T) {
  T = T.newContiguous();
  if (T.nDimension() == 1) {
    return Mat(T.size(0), 1, DataType<Treal>::type, (void*)T.data());
  } else if (T.nDimension() == 2) {
    return Mat(T.size(0), T.size(1), DataType<Treal>::type, (void*)T.data());
  } else if (T.nDimension() == 3) {
    if (T.size(2) == 3) {
      return Mat(T.size(0), T.size(1), DataType<Vec<Treal, 3> >::type, (void*)T.data());
    }
  }
  THerror("TensorToMat: N-d tensors not implemented");
  return matf(0,0); //remove warning
}

template<typename Treal>
inline Mat_<Treal> TensorToMat2d(TH::Tensor<Treal> & T) {
  return (Mat_<Treal>)TensorToMat(T);
}

template<typename Treal>
inline Mat_<Vec<Treal, 3> > TensorToMatImage(TH::Tensor<Treal> & T) {
  return (Mat_<Vec<Treal, 3> >)TensorToMat(T);
}

#endif
