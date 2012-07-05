#ifndef __TH_TEMPLATE_GENERATE_TYPES_H__
#define __TH_TEMPLATE_GENERATE_TYPES_H__

#include "THTemplateTypes.hpp"

#include "generic/THTemplateType.hpp"

#ifdef THTensor_
#undef THTensor_
#endif
#define THTensor_(NAME) TH_CONCAT_4(TH, RealT, Tensor_, NAME)

#define realT unsigned char
#define accrealT long
#define RealT Byte
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_BYTE

#define realT char
#define accrealT long
#define RealT Char
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_CHAR

#define realT short
#define accrealT long
#define RealT Short
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_SHORT

#define realT int
#define accrealT long
#define RealT Int
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_INT

#define realT long
#define accrealT long
#define RealT Long
#define TH_REAL_IS_LONG
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_LONG

#define realT float
#define accrealT double
#define RealT Float
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_FLOAT

#define realT double
#define accrealT double
#define RealT Double
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE_TEMPLATE
#include TH_GENERIC_FILE_TEMPLATE
#undef realT
#undef accrealT
#undef RealT
#undef TH_REAL_IS_DOUBLE

#undef THTensor_
#define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME)

#endif
