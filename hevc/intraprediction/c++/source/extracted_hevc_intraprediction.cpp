#include "extracted_hevc_intraprediction.h"

void hevc_intraprediction(const int& height_intra_pattern,
                          const int& width_intra_pattern,
                          const int& size_block,
                          const uint8_t* intra_pattern,
                          uint8_t* predicted_block,
                          const unsigned int& direction_mode)
{
    if (!intra_pattern)
    {
        throw std::invalid_argument("NULL pointer.");
    }
    if (!predicted_block)
    {
        throw std::invalid_argument("NULL pointer.");
    }
    if (direction_mode > 34)
    {
        throw std::invalid_argument("The direction is not smaller than 34.");
    }
    if (height_intra_pattern < size_block + 1 || height_intra_pattern > 2*size_block + 1)
    {
        throw std::invalid_argument("The height of the intra pattern does not belong to [" + std::to_string(size_block + 1) + ", " + std::to_string(2*size_block + 1) + "].");
    }
    if (width_intra_pattern < size_block + 1 || width_intra_pattern > 2*size_block + 1)
    {
        throw std::invalid_argument("The width of the intra pattern does not belong to [" + std::to_string(size_block + 1) + ", " + std::to_string(2*size_block + 1) + "].");
    }
    int bitdepth(8);
    bool isLuma(true);
    bool isAboveAvailable(true);
    bool isLeftAvailable(true);
    int height_full_pattern(2*size_block + 1);
    int width_full_pattern(2*size_block + 1);
    int* full_pattern(NULL);
    full_pattern = new int[height_full_pattern*width_full_pattern];

    /*
    `temp` is a pointer to the integer
    located at `width_full_pattern` integers
    after the 1st integer of the array.
    */
    int* temp(NULL);
    temp = full_pattern + width_full_pattern;

    /*
    The 1st line and the 1st column of the
    intra pattern are copied into the full
    pattern at the corresponding locations.
    */
    for (int i(0); i < width_intra_pattern; i++)
    {
        full_pattern[i] = (int)intra_pattern[i];
    }
    for (int i(1); i < height_intra_pattern; i++)
    {
        temp[0] = (int)intra_pattern[i*width_intra_pattern];
        temp += width_full_pattern;
    }

    /*
    If the width of the intra pattern is strictly
    smaller than `2*size_block + 1`, the top right
    pixel of the intra pattern pads the top right
    part of the full pattern.
    */
    int padding_value(full_pattern[width_intra_pattern - 1]);
    for (int i(width_intra_pattern); i < width_full_pattern; i++)
    {
        full_pattern[i] = padding_value;
    }

    /*
    If the height of the intra pattern is strictly
    smaller than `2*size_block + 1`, the bottom left
    pixel of the intra pattern pads the bottom left
    part of the full pattern.
    */
    padding_value = full_pattern[(height_intra_pattern - 1)*width_full_pattern];
    for (int i(height_intra_pattern); i < height_full_pattern; i++, temp += width_full_pattern)
    {
        temp[0] = padding_value;
    }

    /*
    The height of the predicted block is
    `size_block`. Its width is `size_block`.
    */
    int* prediction(NULL);
    prediction = new int[size_block*size_block];
    if (direction_mode == PLANAR_IDX)
    {
        xPredIntraPlanar(full_pattern + width_full_pattern + 1,
                         width_full_pattern,
                         prediction,
                         size_block,
                         size_block,
                         size_block);
    }
    else
    {
        xPredIntraAng(bitdepth,
                      full_pattern + width_full_pattern + 1,
                      width_full_pattern,
                      prediction,
                      size_block,
                      size_block,
                      size_block,
                      isLuma,
                      direction_mode,
                      isAboveAvailable,
                      isLeftAvailable);
        if ((direction_mode == DC_IDX) && isAboveAvailable && isLeftAvailable)
        {
            xDCPredFiltering(full_pattern + width_full_pattern + 1,
                             width_full_pattern,
                             prediction,
                             size_block,
                             size_block,
                             size_block,
                             isLuma);
        }
    }

    /*
    The predicted values are cast from `int`
    to `uint8_t`.
    */
    for (int i(0); i < size_block*size_block; i++)
    {
        predicted_block[i] = (uint8_t)prediction[i];
    }
    delete[] prediction;
    delete[] full_pattern;
}

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
                   bool blkLeftAvailable)
{
    int width = int(uiWidth);
    int height = int(uiHeight);
    assert(dirMode != PLANAR_IDX);
    const bool modeDC = dirMode == DC_IDX;

    // Do the DC prediction
    if (modeDC)
    {
        const int dcval = predIntraGetPredValDC(pSrc,
                                                srcStride,
                                                width,
                                                height,
                                                blkAboveAvailable,
                                                blkLeftAvailable);
        for (int y = height; y > 0; y--, pTrueDst += dstStrideTrue)
        {
            for (int x = 0; x < width;)
            {
                pTrueDst[x++] = dcval;
            }
        }
    }
    
    // Do angular predictions
    else
    {
        const bool bIsModeVer = (dirMode >= 18);
        const int intraPredAngleMode = (bIsModeVer) ? (int)dirMode - VER_IDX : -((int)dirMode - HOR_IDX);
        const int absAngMode = abs(intraPredAngleMode);
        const int signAng = intraPredAngleMode < 0 ? -1 : 1;
        const bool edgeFilter = isLuma && (width <= MAXIMUM_INTRA_FILTERED_WIDTH) && (height <= MAXIMUM_INTRA_FILTERED_HEIGHT);
        static const int angTable[9] = {0, 2, 5, 9, 13, 17, 21, 26, 32};
        static const int invAngTable[9] = {0, 4096, 1638, 910, 630, 482, 390, 315, 256};
        int invAngle = invAngTable[absAngMode];
        int absAng = angTable[absAngMode];
        int intraPredAngle = signAng * absAng;
        int* refMain;
        int* refSide;
        int  refAbove[2 * MAX_CU_SIZE + 1];
        int  refLeft[2 * MAX_CU_SIZE + 1];
        if (intraPredAngle < 0)
        {
            const int refMainOffsetPreScale = (bIsModeVer ? height : width) - 1;
            const int refMainOffset = height - 1;
            for (int x = 0; x < width + 1; x++)
            {
                refAbove[x + refMainOffset] = pSrc[x - srcStride - 1];
            }
            for (int y = 0; y < height + 1; y++)
            {
                refLeft[y + refMainOffset] = pSrc[(y - 1)*srcStride - 1];
            }
            refMain = (bIsModeVer ? refAbove : refLeft) + refMainOffset;
            refSide = (bIsModeVer ? refLeft : refAbove) + refMainOffset;
            int invAngleSum = 128;
            for (int k = -1; k > (refMainOffsetPreScale + 1)*intraPredAngle >> 5; k--)
            {
                invAngleSum += invAngle;
                refMain[k] = refSide[invAngleSum >> 8];
            }
        }
        else
        {
            for (int x = 0; x < 2 * width + 1; x++)
            {
                refAbove[x] = pSrc[x - srcStride - 1];
            }
            for (int y = 0; y < 2 * height + 1; y++)
            {
                refLeft[y] = pSrc[(y - 1)*srcStride - 1];
            }
            refMain = bIsModeVer ? refAbove : refLeft;
            refSide = bIsModeVer ? refLeft : refAbove;
        }
        
        int tempArray[MAX_CU_SIZE*MAX_CU_SIZE];
        const int dstStride = bIsModeVer ? dstStrideTrue : MAX_CU_SIZE;
        int *pDst = bIsModeVer ? pTrueDst : tempArray;
        if (!bIsModeVer)
        {
            std::swap(width, height);
        }
        if (intraPredAngle == 0)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    pDst[y*dstStride + x] = refMain[x + 1];
                }
            }
            if (edgeFilter)
            {
                for (int y = 0; y < height; y++)
                {
                    pDst[y*dstStride] = Clip3(0, ((1 << bitDepth) - 1), pDst[y*dstStride] + ((refSide[y + 1] - refSide[0]) >> 1));
                }
            }
        }
        else
        {
            int *pDsty = pDst;
            for (int y = 0, deltaPos = intraPredAngle; y < height; y++, deltaPos += intraPredAngle, pDsty += dstStride)
            {
                const int deltaInt = deltaPos >> 5;
                const int deltaFract = deltaPos & (32 - 1);
                if (deltaFract)
                {
                    const int *pRM = refMain + deltaInt + 1;
                    int lastRefMainPel = *pRM++;
                    for (int x = 0; x < width; pRM++, x++)
                    {
                        int thisRefMainPel = *pRM;
                        pDsty[x + 0] = (int)(((32 - deltaFract)*lastRefMainPel + deltaFract*thisRefMainPel + 16) >> 5);
                        lastRefMainPel = thisRefMainPel;
                    }
                }
                else
                {
                    for (int x = 0; x < width; x++)
                    {
                        pDsty[x] = refMain[x + deltaInt + 1];
                    }
                }
            }
        }
        if (!bIsModeVer)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    pTrueDst[x*dstStrideTrue] = pDst[x];
                }
                pTrueDst++;
                pDst += dstStride;
            }
        }
    }
}

int predIntraGetPredValDC(const int* pSrc,
                          int iSrcStride,
                          unsigned int iWidth,
                          unsigned int iHeight,
                          bool bAbove,
                          bool bLeft)
{
    assert(iWidth > 0 && iHeight > 0);
    unsigned int iInd = 0;
    int iSum = 0;
    int pDcVal;

    if (bAbove)
    {
        for (iInd = 0; iInd < iWidth; iInd++)
        {
            iSum += pSrc[((int)iInd) - iSrcStride];
        }
    }
    if (bLeft)
    {
        for (iInd = 0; iInd < iHeight; iInd++)
        {
            iSum += pSrc[((int)iInd)*iSrcStride - 1];
        }
    }
    if (bAbove && bLeft)
    {
        pDcVal = (iSum + iWidth)/(iWidth + iHeight);
    }
    else if (bAbove)
    {
        pDcVal = (iSum + iWidth/2)/iWidth;
    }
    else if (bLeft)
    {
        pDcVal = (iSum + iHeight/2)/iHeight;
    }
    else
    {
        pDcVal = pSrc[-1];
    }
    return pDcVal;
}

void xPredIntraPlanar(const int* pSrc,
                      int srcStride,
                      int* rpDst,
                      int dstStride,
                      unsigned int width,
                      unsigned int height)
{
    assert(width <= height);

    int leftColumn[MAX_CU_SIZE + 1], topRow[MAX_CU_SIZE + 1], bottomRow[MAX_CU_SIZE], rightColumn[MAX_CU_SIZE];
    unsigned int shift1Dhor = (unsigned int)(log(double(width))/log(2.0));
    unsigned int shift1Dver = (unsigned int)(log(double(height))/log(2.0));

    // Get left and above reference column and row
    for (unsigned int k = 0; k < width + 1; k++)
    {
        topRow[k] = pSrc[((int)k) - srcStride];
    }

    for (unsigned int k = 0; k < height + 1; k++)
    {
        leftColumn[k] = pSrc[((int)k)*srcStride - 1];
    }

    // Prepare intermediate variables used in interpolation
    int bottomLeft = leftColumn[height];
    int topRight = topRow[width];

    for (unsigned int k = 0; k < width; k++)
    {
        bottomRow[k] = bottomLeft - topRow[k];
        topRow[k] <<= shift1Dver;
    }

    for (unsigned int k = 0; k < height; k++)
    {
        rightColumn[k] = topRight - leftColumn[k];
        leftColumn[k] <<= shift1Dhor;
    }

    //#if (RExt__SQUARE_TRANSFORM_CHROMA_422 != 0)
    const unsigned int topRowShift = 0;
    //#else
    //  const UInt topRowShift = (isChroma(channelType) && (format == CHROMA_422)) ? 1 : 0;
    //#endif

    // Generate prediction signal
    for (unsigned int y = 0; y < height; y++)
    {
        int horPred = leftColumn[y] + width;
        for (unsigned int x = 0; x < width; x++)
        {
            horPred += rightColumn[y];
            topRow[x] += bottomRow[x];
            int vertPred = ((topRow[x] + topRowShift) >> topRowShift);
            rpDst[y*dstStride + x] = (horPred + vertPred) >> (shift1Dhor + 1);
        }
    }
}

void xDCPredFiltering(const int* pSrc,
                      int iSrcStride,
                      int*& rpDst,
                      int iDstStride,
                      int iWidth,
                      int iHeight,
                      bool isLuma)
{
    int* pDst = rpDst;
    int x, y, iDstStride2, iSrcStride2;
    if (isLuma && (iWidth <= MAXIMUM_INTRA_FILTERED_WIDTH) && (iHeight <= MAXIMUM_INTRA_FILTERED_HEIGHT))
    {
        pDst[0] = (int)((pSrc[-iSrcStride] + pSrc[-1] + 2 * pDst[0] + 2) >> 2);
        for (x = 1; x < iWidth; x++)
        {
            pDst[x] = (int)((pSrc[x - iSrcStride] + 3 * pDst[x] + 2) >> 2);
        }
        for (y = 1, iDstStride2 = iDstStride, iSrcStride2 = iSrcStride - 1; y < iHeight; y++, iDstStride2 += iDstStride, iSrcStride2 += iSrcStride)
        {
            pDst[iDstStride2] = (int)((pSrc[iSrcStride2] + 3 * pDst[iDstStride2] + 2) >> 2);
        }
    }
    return;
}


