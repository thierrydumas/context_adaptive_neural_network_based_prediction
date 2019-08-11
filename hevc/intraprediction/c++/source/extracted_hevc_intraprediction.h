#include <iostream>
#include <algorithm>
#include <assert.h>
#include <stdexcept>
#include <stdint.h>
#include <string>

#define PLANAR_IDX 0
#define VER_IDX 26
#define HOR_IDX 10
#define DC_IDX 1
#define MAXIMUM_INTRA_FILTERED_WIDTH 16
#define MAXIMUM_INTRA_FILTERED_HEIGHT 16
#define MAX_CU_DEPTH 6
#define MAX_CU_SIZE (1<<(MAX_CU_DEPTH))

template <typename T>
inline T Clip3(const T minVal, const T maxVal, const T a)
{
    return std::min<T>(std::max<T>(minVal, a), maxVal);
}

void hevc_intraprediction(const int& height_intra_pattern,
                          const int& width_intra_pattern,
                          const int& size_block,
                          const uint8_t* intra_pattern,
                          uint8_t* predicted_block,
                          const unsigned int& direction_mode);

void xPredIntraAng(int bitDepth,
                   const int* pSrc,
                   int srcStride,
                   int* pTrueDst,
                   int dstStrideTrue,
                   unsigned int uiWidth,
                   unsigned int uiHeight,
                   bool isLuma,
                   unsigned int dirMode,
                   bool blkAboveAvailable,
                   bool blkLeftAvailable);

int predIntraGetPredValDC(const int* pSrc,
                          int iSrcStride,
                          unsigned int iWidth,
                          unsigned int iHeight,
                          bool bAbove,
                          bool bLeft);

void xPredIntraPlanar(const int* pSrc,
                      int srcStride,
                      int* rpDst,
                      int dstStride,
                      unsigned int width,
                      unsigned int height);

void xDCPredFiltering(const int* pSrc,
                      int iSrcStride,
                      int*& rpDst,
                      int iDstStride,
                      int iWidth,
                      int iHeight,
                      bool isLuma);


