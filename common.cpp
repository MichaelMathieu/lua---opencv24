#include "common.hpp"

void display(const Mat & im) {
  Mat tmp;
  im.convertTo(tmp, CV_8U);
  imshow("main", tmp);
  cvWaitKey(0);
}

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
