#ifndef TH_GENERIC_FILE_TEMPLATE
#define TH_GENERIC_FILE_TEMPLATE "generic/THTemplateType.hpp"
#else

//#define ARRAY_CHECKS

namespace TH {

  template<> class Types<realT> {
  public:
    typedef TH_CONCAT_3(TH, RealT, Tensor) CTensor;
    typedef TH_CONCAT_3(TH, RealT, Storage) CStorage;
    typedef realT THReal;
  };

  template<> class Storage<realT> {
  public:
    typedef Types<realT>::CTensor CTensor;
    typedef Types<realT>::CStorage CStorage;
    typedef Types<realT>::THReal THReal;
  private:
    CStorage* cstorage;
  public:
    inline Storage(CStorage* cstorage)
      :cstorage(cstorage) {};
  };

  template<> class Tensor<realT> {
  public:
    typedef Types<realT>::CTensor CTensor;
    typedef Types<realT>::CStorage CStorage;
    typedef Types<realT>::THReal THReal;
  private:
    CTensor* ctensor;
    mutable bool hasToBeFreed;
  public: // constructor etc.
    inline Tensor(CTensor* ctensor, bool hasToBeFreed = false)
      :ctensor(ctensor), hasToBeFreed(hasToBeFreed) {};
    inline Tensor(const Tensor & src)
      :ctensor(src.ctensor), hasToBeFreed(src.hasToBeFreed) {
      if (hasToBeFreed)
	retain();
    };
    inline ~Tensor() {
      if (hasToBeFreed) this->free();
    };
    inline Tensor & operator=(const Tensor & src) {
      if (&src != this) {
	if (hasToBeFreed)
	  this->free();
	ctensor = src.ctensor;
	hasToBeFreed = src.hasToBeFreed;
	if (hasToBeFreed)
	  retain();
      }
      return *this;
    };
    inline operator CTensor*() {
      return ctensor;
    }
    inline operator const CTensor*() const {
      return ctensor;
    }
  public: // methods
    inline void retain() {
      THTensor_(retain)(ctensor);
    };
    inline void free() {
      THTensor_(free)(ctensor);
      hasToBeFreed = false;
    };
    
    inline Storage<THReal> storage() {
      return Storage<THReal>(THTensor_(storage)(ctensor));
    };
    
    inline THReal* data() {
      return THTensor_(data)(ctensor);
    };
    inline const THReal* data() const {
      return THTensor_(data)(ctensor);
    };
    
    inline long nDimension() const {
      return THTensor_(nDimension)(ctensor);
    };
    
    inline const long* stride() const {
      return ctensor->stride;
    };
    inline long stride(int i) const {
#ifdef ARRAY_CHECKS
      assert((0 <= i) && (i < nDimension()));
#endif
      return ctensor->stride[i];
    };
    
    inline const long* size() const {
      return ctensor->size;
    };
    inline long size(int i) const {
#ifdef ARRAY_CHECKS
      assert((0 <= i) && (i < nDimension()));
#endif
      return ctensor->size[i];
    };

    inline void resize(long s0, long s1 = -1, long s2 = -1, long s3 = -1, long s4 = -1) {
      if (s1 == -1)
	THTensor_(resize1d)(ctensor, s0);
      else if (s2 == -1)
	THTensor_(resize2d)(ctensor, s0, s1);
      else if (s3 == -1)
	THTensor_(resize3d)(ctensor, s0, s1, s2);
      else if (s4 == -1)
	THTensor_(resize4d)(ctensor, s0, s1, s2, s3);
      else
	THTensor_(resize5d)(ctensor, s0, s1, s2, s3, s4);
    }
    
    inline bool isContiguous() const {
      return (bool)(THTensor_(isContiguous)(ctensor));
    };
    inline Tensor newContiguous() const {
      hasToBeFreed = true;
      return THTensor_(newContiguous)(ctensor);
    };

    // accessors
    inline THReal & operator[](int i) {
#ifdef ARRAY_CHECKS
      assert(isContiguous());
#endif
      return data()[i];
    }
    inline const THReal & operator[](int i) const {
#ifdef ARRAY_CHECKS
      assert(isContiguous());
#endif
      return data()[i];
    }
    inline THReal & operator()(int i) {
      return data()[stride(0)*i];
    }
    inline const THReal & operator()(int i) const {
      return data()[stride(0)*i];
    }
    inline THReal & operator()(int i, int j) {
      return data()[stride(0)*i+stride(1)*j];
    }
    inline const THReal & operator()(int i, int j) const {
      return data()[stride(0)*i+stride(1)*j];
    }
    inline THReal & operator()(int i, int j, int k) {
      return data()[stride(0)*i+stride(1)*j+stride(2)*k];
    }
    inline const THReal & operator()(int i, int j, int k) const {
      return data()[stride(0)*i+stride(1)*j+stride(2)*k];
    }
    inline THReal & operator()(int i, int j, int k, int l) {
      return data()[stride(0)*i+stride(1)*j+stride(2)*k+stride(3)*l];
    }
    inline const THReal & operator()(int i, int j, int k, int l) const {
      return data()[stride(0)*i+stride(1)*j+stride(2)*k+stride(3)*l];
    }
  };

}

#endif
