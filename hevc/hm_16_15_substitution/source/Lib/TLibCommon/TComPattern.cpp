/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "TComPic.h"
#include "TComPattern.h"
#include "TComDataCU.h"
#include "TComTU.h"
#include "Debug.h"
#include "TComPrediction.h"

// #define CHECK_OVERLAP_INTRA_PATTERN_MASKED_CONTEXT_PORTIONS

// Below are the forward declarations.
Void fillReferenceSamples(const Int bitDepth, 
#if O0043_BEST_EFFORT_DECODING
                          const Int bitDepthDelta, 
#endif
                          const Pel* piRoiOrigin, 
                          Pel* piIntraTemp,
                          const Bool* bNeighborFlags,
                          const Int iNumIntraNeighbor, 
                          const Int unitWidth, 
                          const Int unitHeight, 
                          const Int iAboveUnits, 
                          const Int iLeftUnits,
                          const UInt uiWidth, 
                          const UInt uiHeight, 
                          const Int iPicStride);

Bool isAboveLeftAvailable(const TComDataCU* pcCU,
                          UInt uiPartIdxLT);

Int isAboveAvailable(const TComDataCU* pcCU,
                     UInt uiPartIdxLT,
                     UInt uiPartIdxRT,
                     Bool* bValidFlags);

Int isLeftAvailable(const TComDataCU* pcCU,
                    UInt uiPartIdxLT,
                    UInt uiPartIdxLB,
                    Bool* bValidFlags);

Int isAboveRightAvailable(const TComDataCU* pcCU,
                          UInt uiPartIdxLT,
                          UInt uiPartIdxRT,
                          Bool* bValidFlags);

Int isBelowLeftAvailable(const TComDataCU* pcCU,
                         UInt uiPartIdxLT,
                         UInt uiPartIdxLB,
                         Bool* bValidFlags);

Void TComPatternParam::setPatternParamPel(Pel* piTexture,
                                          Int iRoiWidth,
                                          Int iRoiHeight,
                                          Int iStride,
                                          Int bitDepth)
{
    m_piROIOrigin = piTexture;
    m_iROIWidth = iRoiWidth;
    m_iROIHeight = iRoiHeight;
    m_iPatternStride = iStride;
    m_bitDepth = bitDepth;
}

Void TComPattern::initPattern(Pel* piY,
                              Int iRoiWidth,
                              Int iRoiHeight,
                              Int iStride,
                              Int bitDepthLuma)
{
    m_cPatternY.setPatternParamPel(piY,
                                   iRoiWidth,
                                   iRoiHeight,
                                   iStride,
                                   bitDepthLuma);
}

bool is_context_available(TComDataCU* pcCU,
                          const int& nbUnitsVerticalBetweenAnchorCurrentTU,
                          const int& nbUnitsHorizontalBetweenAnchorCurrentTU,
                          const int& uiPartIdxLT)
{
    /*
    The current CTU contains 256 units. `uiRasterIdxLT`
    is the raster-scan index in the current CTU of the
    unit located at the top-left corner of the current
    TU. `uiRasterIdxLT` belongs to [|0, 255|].
    */
    const int uiRasterIdxLT(g_auiZscanToRaster[uiPartIdxLT]);

    /*
    `numUnitsPerRow` is the number of units per row
    in a CTU. `numUnitsPerRow` is equal to 16.
    */
    const int numUnitsPerRow(pcCU->getPic()->getNumPartInCtuWidth());
    
    /*
    `isLeftCtu` is true if the anchor pixel is located
    in a CTU on the left side of the current CTU.
    */
    const bool isLeftCtu(RasterAddress::lessThanCol(uiRasterIdxLT, nbUnitsHorizontalBetweenAnchorCurrentTU, numUnitsPerRow));
    
    /*
    `isAboveCtu` is true if the anchor pixel is located
    in a CTU located above the current CTU.
    */
    const bool isAboveCtu(RasterAddress::lessThanRow(uiRasterIdxLT, nbUnitsVerticalBetweenAnchorCurrentTU, numUnitsPerRow));
    
    /*
    `addrCtuAnchor` is the address of the CTU that
    contains the anchor pixel. The anchor pixel is
    the pixel in the decoded image whose position
    corresponds to the top-left of the context we
    feed into the prediction neural networks.
    */
    TComDataCU* addrCtuAnchor(NULL);
    if (isLeftCtu)
    {
        if (isAboveCtu)
        {
            addrCtuAnchor = pcCU->getCtuAboveLeft();
        }
        else
        {
            addrCtuAnchor = pcCU->getCtuLeft();
        }
    }
    else
    {
        if (isAboveCtu)
        {
            addrCtuAnchor = pcCU->getCtuAbove();
        }
        else
        {
            /*
            Here, the CTU that contains the anchor pixel
            is the current CTU.
            */
            return true;
        }
    }
    return addrCtuAnchor ? true : false;
}

Void TComPrediction::initIntraPatternChType(TComTU& rTu,
                                            bool& contextFlag,
                                            const ComponentID compID,
                                            const Bool bFilterRefSamples DEBUG_STRING_FN_DECLARE(sDebug))
{
    const ChannelType chType(toChannelType(compID));
    TComDataCU* pcCU(rTu.getCU());
    const TComSPS& sps(*(pcCU->getSlice()->getSPS()));
    const UInt uiZorderIdxInPart(rTu.GetAbsPartIdxTU());
    
    /*
    `uiTuWidth` is the width of the current TU in pixels.
    `uiTuHeight` is the height of the current TU in pixels.
    `uiTuWidth` and `uiTuHeight` belong to {4, 8, 16, 32, 64}
    pixels for the luminance channel. `uiTuWidth` and `uiTuHeight`
    belong to {4, 8, 16} for the two chrominance channels
    (is it a bug?).
    */
    const UInt uiTuWidth(rTu.getRect(compID).width);
    const UInt uiTuHeight(rTu.getRect(compID).height);
    const UInt uiTuWidth2(uiTuWidth << 1);
    const UInt uiTuHeight2(uiTuHeight << 1);
    
    /*
    `sps.getMaxCUWidth()` returns the maximum CU width. The
    maximum CU width is determined by a configuration parameter,
    usually equal to 64.
    `sps.getMaxTotalCUDepth()` returns the maximum depth. The
    maximum depth is also determined by a configuration parameter,
    usually equal to 4.
    */
    const Int iBaseUnitSize(sps.getMaxCUWidth() >> sps.getMaxTotalCUDepth());
    
    /*
    For the luminance channel, `pcCU->getPic()->getPicYuvRec()->getComponentScaleX(compID)`
    and `pcCU->getPic()->getPicYuvRec()->getComponentScaleY(compID)` return 0. For the two
    chrominance channels, `pcCU->getPic()->getPicYuvRec()->getComponentScaleX(compID)`
    and `pcCU->getPic()->getPicYuvRec()->getComponentScaleY(compID)` return 1.
    For the luminance channel, `iUnitWidth` and `iUnitHeight` are equal to 4 pixels.
    For the two chrominance channels, `iUnitWidth` and `iUnitHeight` are equal to
    2 pixels.
    */
    const Int iUnitWidth(iBaseUnitSize >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(compID));
    const Int iUnitHeight(iBaseUnitSize >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(compID));
    
    /*
    `iTUWidthInUnits` is the width of the current TU in units.
    `iTUHeightInUnits` is the height of the current TU in units.
    `iTUWidthInUnits` and `iTUHeightInUnits` belong to {1, 2, 4, 8, 16}
    units for the luminance channel. `iTUWidthInUnits` and `iTUHeightInUnits`
    belong to {2, 4, 8} for the two chrominance channels.
    */
    const Int iTUWidthInUnits(uiTuWidth/iUnitWidth);
    const Int iTUHeightInUnits(uiTuHeight/iUnitHeight);
    const Int iAboveUnits(iTUWidthInUnits << 1);
    const Int iLeftUnits(iTUHeightInUnits << 1);
    const Int bitDepthForChannel(sps.getBitDepth(chType));
    const Int iPartIdxStride(pcCU->getPic()->getNumPartInCtuWidth());
    
    /*
    `uiPartIdxLT` is the Z-order index in the current CTU
    of the unit at the top-left corner of the current TU.
    */
    const UInt uiPartIdxLT(pcCU->getZorderIdxInCtu() + uiZorderIdxInPart);
    
    /*
    `uiPartIdxRT` is the Z-order index in the current CTU
    of the unit at the top-right corner of the current TU.
    */
    const UInt uiPartIdxRT(g_auiRasterToZscan[g_auiZscanToRaster[uiPartIdxLT] + iTUWidthInUnits - 1]);
    
    /*
    `uiPartIdxLB` is the Z-order index in the current CTU
    of the unit at the bottom-left corner of the current TU.
    */
    const UInt uiPartIdxLB(g_auiRasterToZscan[g_auiZscanToRaster[uiPartIdxLT] + ((iTUHeightInUnits - 1)*iPartIdxStride)]);
    Int iPicStride(pcCU->getPic()->getStride(compID));
    
    /*
    For the luminance channel, if `uiTuWidth` is equal to 64
    pixels and `uiTuHeight` is equal to 64 pixels, the number
    of neighbouring units is equal to 65.
    */
    Bool bNeighborFlags[4*MAX_NUM_PART_IDXS_IN_CTU_WIDTH + 1];
    Int iNumIntraNeighbor(0);
    bNeighborFlags[iLeftUnits] = isAboveLeftAvailable(pcCU,
                                                      uiPartIdxLT);
    iNumIntraNeighbor += bNeighborFlags[iLeftUnits] ? 1 : 0;
    iNumIntraNeighbor += isAboveAvailable(pcCU,
                                          uiPartIdxLT,
                                          uiPartIdxRT,
                                          (bNeighborFlags + iLeftUnits + 1));
    iNumIntraNeighbor += isAboveRightAvailable(pcCU,
                                               uiPartIdxLT,
                                               uiPartIdxRT,
                                               (bNeighborFlags + iLeftUnits + 1 + iTUWidthInUnits));
    iNumIntraNeighbor += isLeftAvailable(pcCU,
                                         uiPartIdxLT,
                                         uiPartIdxLB,
                                         (bNeighborFlags + iLeftUnits - 1));
    iNumIntraNeighbor += isBelowLeftAvailable(pcCU,
                                              uiPartIdxLT,
                                              uiPartIdxLB,
                                              (bNeighborFlags + iLeftUnits - 1 - iTUHeightInUnits));
    
    /*
    `uiROIHeight` is the height of the current intra
    pattern in pixels. `uiROIWidth` is the width of
    the current intra pattern in pixels.
    */
    const UInt uiROIWidth(uiTuWidth2 + 1);
    const UInt uiROIHeight(uiTuHeight2 + 1);

#if DEBUG_STRING
  std::stringstream ss(stringstream::out);
#endif

{
    /*
    `piIntraTemp` is a pointer to the intra
    pattern buffer.
    */
    Pel* piIntraTemp(m_piYuvExt[compID][PRED_BUF_UNFILTERED]);
    
    /*
    `piRoiOrigin` points to the pixel at the
    top-left corner of the current TU.
    */
    Pel* piRoiOrigin(pcCU->getPic()->getPicYuvRec()->getAddr(compID, pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + uiZorderIdxInPart));
    
    /*
    The intra pattern for the regular HEVC intra prediction
    modes is filled before filling the masked context portions
    for the prediction neural networks as the filled intra pattern
    is needed for checking.
    */
#if O0043_BEST_EFFORT_DECODING
    const Int bitDepthForChannelInStream(sps.getStreamBitDepth(chType));
    fillReferenceSamples(bitDepthForChannelInStream,
                         bitDepthForChannelInStream - bitDepthForChannel,
#else
    fillReferenceSamples(bitDepthForChannel,
#endif
                         piRoiOrigin,
                         piIntraTemp,
                         bNeighborFlags,
                         iNumIntraNeighbor,
                         iUnitWidth,
                         iUnitHeight,
                         iAboveUnits,
                         iLeftUnits,
                         uiROIWidth,
                         uiROIHeight,
                         iPicStride);
    
    /*
    If the context for the prediction neural network
    is available, `contextFlag` is true.
    */
    contextFlag = is_context_available(pcCU,
                                       iTUHeightInUnits,
                                       iTUWidthInUnits,
                                       uiPartIdxLT);
    if (contextFlag)
    {
        float* piPortionAbove(NULL);
        float* piPortionLeft(NULL);
        if (uiTuWidth <= 8)
        {
            const int index_4_8(static_cast<int>(log2(uiTuWidth) - 2.));
            
            /*
            The precedence of . (member access) is lower than the precedence
            of [] (subscript).
            */
            piPortionAbove = m_tensors_flattened_context.at(index_4_8).flat<float>().data();
            piPortionLeft = piPortionAbove + 3*uiTuWidth*uiTuWidth;
        }
        else
        {
            const int index_16_32_64(static_cast<int>(log2(uiTuWidth) - 4.));
            piPortionAbove = m_tensors_portion_above.at(index_16_32_64).flat<float>().data();
            piPortionLeft = m_tensors_portion_left.at(index_16_32_64).flat<float>().data();
        }
        if (bitDepthForChannel != 8)
        {
            std::cerr << "HEVC with the substitution of the mode of index 18 with the neural networks mode is not coded yet for bitdepths different from 8." << std::endl;
            assert(false);
        }
        int error_code(0);
        error_code = extract_context_portions(piRoiOrigin,
                                              piPortionAbove,
                                              piPortionLeft,
                                              bNeighborFlags,
                                              iNumIntraNeighbor,
                                              iUnitWidth,
                                              iUnitHeight,
                                              iAboveUnits,
                                              iLeftUnits,
                                              static_cast<int>(uiTuWidth),
                                              static_cast<int>(uiTuHeight),
                                              iPicStride,
                                              m_meanTraining);
        assert(error_code >= 0);
        
#ifdef CHECK_OVERLAP_INTRA_PATTERN_MASKED_CONTEXT_PORTIONS
        error_code = check_overlapping_intra_pattern_context_portions(piPortionAbove,
                                                                      piPortionLeft,
                                                                      piIntraTemp,
                                                                      uiTuWidth,
                                                                      m_meanTraining);
        assert(error_code >= 0);
#endif
    }

#if DEBUG_STRING
    if (DebugOptionList::DebugString_Pred.getInt() & DebugStringGetPredModeMask(MODE_INTRA))
    {
        ss << "###: generating Ref Samples for channel " << compID << " and " << rTu.getRect(compID).width << " x " << rTu.getRect(compID).height << "\n";
        for (UInt y = 0; y < uiROIHeight; y++)
        {
            ss << "###: - ";
            for (UInt x = 0; x < uiROIWidth; x++)
            {
                if (x == 0 || y == 0)
                {
                    ss << piIntraTemp[y*uiROIWidth + x] << ", ";
                }
            }
            ss << "\n";
        }
    }
#endif
    if (bFilterRefSamples)
    {
        Int stride = uiROIWidth;
        const Pel* piSrcPtr = piIntraTemp + (stride*uiTuHeight2);
        Pel* piDestPtr = m_piYuvExt[compID][PRED_BUF_FILTERED] + (stride*uiTuHeight2);
        Bool useStrongIntraSmoothing = isLuma(chType) && sps.getUseStrongIntraSmoothing();
        const Pel bottomLeft = piIntraTemp[stride*uiTuHeight2];
        const Pel topLeft = piIntraTemp[0];
        const Pel topRight = piIntraTemp[uiTuWidth2];
        if (useStrongIntraSmoothing)
        {
#if O0043_BEST_EFFORT_DECODING
            const Int  threshold = 1 << (bitDepthForChannelInStream - 5);
#else
            const Int  threshold = 1 << (bitDepthForChannel - 5);
#endif
            const Bool bilinearLeft = abs((bottomLeft + topLeft) - (2*piIntraTemp[stride*uiTuHeight])) < threshold;
            const Bool bilinearAbove = abs((topLeft + topRight) - (2*piIntraTemp[uiTuWidth ])) < threshold;
            if ((uiTuWidth < 32) || (!bilinearLeft) || (!bilinearAbove))
            {
                useStrongIntraSmoothing = false;
            }
        }
        *piDestPtr = *piSrcPtr;
        piDestPtr -= stride;
        piSrcPtr -= stride;
        if (useStrongIntraSmoothing)
        {
            const Int shift = g_aucConvertToBit[uiTuHeight] + 3;
            for(UInt i = 1; i < uiTuHeight2; i++, piDestPtr -= stride)
            {
                *piDestPtr = (((uiTuHeight2 - i)*bottomLeft) + (i*topLeft) + uiTuHeight) >> shift;
            }
            piSrcPtr -= stride*(uiTuHeight2 - 1);
        }
        else
        {
            for(UInt i = 1; i < uiTuHeight2; i++, piDestPtr -= stride, piSrcPtr -= stride)
            {
                *piDestPtr = (piSrcPtr[stride] + 2*piSrcPtr[0] + piSrcPtr[-stride] + 2) >> 2;
            }
        }
        if (useStrongIntraSmoothing)
        {
            *piDestPtr = piSrcPtr[0];
        }
        else
        {
            *piDestPtr = (piSrcPtr[stride] + 2*piSrcPtr[0] + piSrcPtr[1] + 2) >> 2;
        }
        piDestPtr += 1;
        piSrcPtr += 1;
        if (useStrongIntraSmoothing)
        {
            const Int shift = g_aucConvertToBit[uiTuWidth] + 3;
            for(UInt i = 1; i < uiTuWidth2; i++, piDestPtr++)
            {
                *piDestPtr = (((uiTuWidth2 - i)*topLeft) + (i*topRight) + uiTuWidth) >> shift;
            }
            piSrcPtr += uiTuWidth2 - 1;
        }
        else
        {
            for(UInt i = 1; i < uiTuWidth2; i++, piDestPtr++, piSrcPtr++)
            {
                *piDestPtr = (piSrcPtr[1] + 2*piSrcPtr[0] + piSrcPtr[-1] + 2) >> 2;
            }
        }
        *piDestPtr = *piSrcPtr;
#if DEBUG_STRING
        if (DebugOptionList::DebugString_Pred.getInt() & DebugStringGetPredModeMask(MODE_INTRA))
        {
            ss << "###: filtered result for channel " << compID <<"\n";
            for (UInt y = 0; y < uiROIHeight; y++)
            {
                ss << "###: - ";
                for (UInt x = 0; x < uiROIWidth; x++)
                {
                    if (x == 0 || y == 0)
                    {
                        ss << m_piYuvExt[compID][PRED_BUF_FILTERED][y*uiROIWidth + x] << ", ";
                    }
                }
                ss << "\n";
            }
        }
#endif
    }
}
    DEBUG_STRING_APPEND(sDebug, ss.str())
}

Void fillReferenceSamples( const Int bitDepth, 
#if O0043_BEST_EFFORT_DECODING
                           const Int bitDepthDelta, 
#endif
                           const Pel* piRoiOrigin, 
                                 Pel* piIntraTemp,
                           const Bool* bNeighborFlags,
                           const Int iNumIntraNeighbor, 
                           const Int unitWidth, 
                           const Int unitHeight, 
                           const Int iAboveUnits, 
                           const Int iLeftUnits,
                           const UInt uiWidth, 
                           const UInt uiHeight, 
                           const Int iPicStride )
{
  const Pel* piRoiTemp;
  Int  i, j;
  Int  iDCValue = 1 << (bitDepth - 1);
  const Int iTotalUnits = iAboveUnits + iLeftUnits + 1; //+1 for top-left

  if (iNumIntraNeighbor == 0)
  {
    // Fill border with DC value
    for (i=0; i<uiWidth; i++)
    {
      piIntraTemp[i] = iDCValue;
    }
    for (i=1; i<uiHeight; i++)
    {
      piIntraTemp[i*uiWidth] = iDCValue;
    }
  }
  else if (iNumIntraNeighbor == iTotalUnits)
  {
    // Fill top-left border and top and top right with rec. samples
    piRoiTemp = piRoiOrigin - iPicStride - 1;

    for (i=0; i<uiWidth; i++)
    {
#if O0043_BEST_EFFORT_DECODING
      piIntraTemp[i] = piRoiTemp[i] << bitDepthDelta;
#else
      piIntraTemp[i] = piRoiTemp[i];
#endif
    }

    // Fill left and below left border with rec. samples
    piRoiTemp = piRoiOrigin - 1;

    for (i=1; i<uiHeight; i++)
    {
#if O0043_BEST_EFFORT_DECODING
      piIntraTemp[i*uiWidth] = (*(piRoiTemp)) << bitDepthDelta;
#else
      piIntraTemp[i*uiWidth] = *(piRoiTemp);
#endif
      piRoiTemp += iPicStride;
    }
  }
  else // reference samples are partially available
  {
    // all above units have "unitWidth" samples each, all left/below-left units have "unitHeight" samples each
    const Int  iTotalSamples = (iLeftUnits * unitHeight) + ((iAboveUnits + 1) * unitWidth);
    Pel  piIntraLine[5 * MAX_CU_SIZE];
    Pel  *piIntraLineTemp;
    const Bool *pbNeighborFlags;


    // Initialize
    for (i=0; i<iTotalSamples; i++)
    {
      piIntraLine[i] = iDCValue;
    }

    // Fill top-left sample
    piRoiTemp = piRoiOrigin - iPicStride - 1;
    piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight);
    pbNeighborFlags = bNeighborFlags + iLeftUnits;
    if (*pbNeighborFlags)
    {
#if O0043_BEST_EFFORT_DECODING
      Pel topLeftVal=piRoiTemp[0] << bitDepthDelta;
#else
      Pel topLeftVal=piRoiTemp[0];
#endif
      for (i=0; i<unitWidth; i++)
      {
        piIntraLineTemp[i] = topLeftVal;
      }
    }

    // Fill left & below-left samples (downwards)
    piRoiTemp += iPicStride;
    piIntraLineTemp--;
    pbNeighborFlags--;

    for (j=0; j<iLeftUnits; j++)
    {
      if (*pbNeighborFlags)
      {
        for (i=0; i<unitHeight; i++)
        {
#if O0043_BEST_EFFORT_DECODING
          piIntraLineTemp[-i] = piRoiTemp[i*iPicStride] << bitDepthDelta;
#else
          piIntraLineTemp[-i] = piRoiTemp[i*iPicStride];
#endif
        }
      }
      piRoiTemp += unitHeight*iPicStride;
      piIntraLineTemp -= unitHeight;
      pbNeighborFlags--;
    }

    // Fill above & above-right samples (left-to-right) (each unit has "unitWidth" samples)
    piRoiTemp = piRoiOrigin - iPicStride;
    // offset line buffer by iNumUints2*unitHeight (for left/below-left) + unitWidth (for above-left)
    piIntraLineTemp = piIntraLine + (iLeftUnits * unitHeight) + unitWidth;
    pbNeighborFlags = bNeighborFlags + iLeftUnits + 1;
    for (j=0; j<iAboveUnits; j++)
    {
      if (*pbNeighborFlags)
      {
        for (i=0; i<unitWidth; i++)
        {
#if O0043_BEST_EFFORT_DECODING
          piIntraLineTemp[i] = piRoiTemp[i] << bitDepthDelta;
#else
          piIntraLineTemp[i] = piRoiTemp[i];
#endif
        }
      }
      piRoiTemp += unitWidth;
      piIntraLineTemp += unitWidth;
      pbNeighborFlags++;
    }

    // Pad reference samples when necessary
    Int iCurrJnit = 0;
    Pel  *piIntraLineCur   = piIntraLine;
    const UInt piIntraLineTopRowOffset = iLeftUnits * (unitHeight - unitWidth);

    if (!bNeighborFlags[0])
    {
      // very bottom unit of bottom-left; at least one unit will be valid.
      {
        Int   iNext = 1;
        while (iNext < iTotalUnits && !bNeighborFlags[iNext])
        {
          iNext++;
        }
        Pel *piIntraLineNext = piIntraLine + ((iNext < iLeftUnits) ? (iNext * unitHeight) : (piIntraLineTopRowOffset + (iNext * unitWidth)));
        const Pel refSample = *piIntraLineNext;
        // Pad unavailable samples with new value
        Int iNextOrTop = std::min<Int>(iNext, iLeftUnits);
        // fill left column
        while (iCurrJnit < iNextOrTop)
        {
          for (i=0; i<unitHeight; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += unitHeight;
          iCurrJnit++;
        }
        // fill top row
        while (iCurrJnit < iNext)
        {
          for (i=0; i<unitWidth; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += unitWidth;
          iCurrJnit++;
        }
      }
    }

    // pad all other reference samples.
    while (iCurrJnit < iTotalUnits)
    {
      if (!bNeighborFlags[iCurrJnit]) // samples not available
      {
        {
          const Int numSamplesInCurrUnit = (iCurrJnit >= iLeftUnits) ? unitWidth : unitHeight;
          const Pel refSample = *(piIntraLineCur-1);
          for (i=0; i<numSamplesInCurrUnit; i++)
          {
            piIntraLineCur[i] = refSample;
          }
          piIntraLineCur += numSamplesInCurrUnit;
          iCurrJnit++;
        }
      }
      else
      {
        piIntraLineCur += (iCurrJnit >= iLeftUnits) ? unitWidth : unitHeight;
        iCurrJnit++;
      }
    }

    // Copy processed samples

    piIntraLineTemp = piIntraLine + uiHeight + unitWidth - 2;
    // top left, top and top right samples
    for (i=0; i<uiWidth; i++)
    {
      piIntraTemp[i] = piIntraLineTemp[i];
    }

    piIntraLineTemp = piIntraLine + uiHeight - 1;
    for (i=1; i<uiHeight; i++)
    {
      piIntraTemp[i*uiWidth] = piIntraLineTemp[-i];
    }
  }
}

Bool TComPrediction::filteringIntraReferenceSamples(const ComponentID compID, UInt uiDirMode, UInt uiTuChWidth, UInt uiTuChHeight, const ChromaFormat chFmt, const Bool intraReferenceSmoothingDisabled)
{
    Bool bFilter;
    if (!filterIntraReferenceSamples(toChannelType(compID), chFmt, intraReferenceSmoothingDisabled))
    {
        bFilter = false;
    }
    else
    {
        assert(uiTuChWidth >= 4 && uiTuChHeight >= 4 && uiTuChWidth < 128 && uiTuChHeight < 128);
        if (uiDirMode == DC_IDX)
        {
            bFilter = false;
        }
        else
        {
            Int diff = min<Int>(abs((Int) uiDirMode - HOR_IDX), abs((Int)uiDirMode - VER_IDX));
            UInt sizeIndex=g_aucConvertToBit[uiTuChWidth];
            assert(sizeIndex < MAX_INTRA_FILTER_DEPTHS);
            bFilter = diff > m_aucIntraFilter[toChannelType(compID)][sizeIndex];
        }
    }
    return bFilter;
}

Bool isAboveLeftAvailable(const TComDataCU* pcCU, UInt uiPartIdxLT)
{
    Bool bAboveLeftFlag;
    UInt uiPartAboveLeft;
    const TComDataCU* pcCUAboveLeft = pcCU->getPUAboveLeft(uiPartAboveLeft, uiPartIdxLT);
    if (pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
    {
        bAboveLeftFlag = (pcCUAboveLeft && pcCUAboveLeft->isIntra(uiPartAboveLeft));
    }
    else
    {
        bAboveLeftFlag = (pcCUAboveLeft ? true : false);
    }
    return bAboveLeftFlag;
}

Int isAboveAvailable(const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool *bValidFlags)
{
    const UInt uiRasterPartBegin = g_auiZscanToRaster[uiPartIdxLT];
    const UInt uiRasterPartEnd = g_auiZscanToRaster[uiPartIdxRT]+1;
    const UInt uiIdxStep = 1;
    Bool *pbValidFlags = bValidFlags;
    Int iNumIntra = 0;
    
    for (UInt uiRasterPart = uiRasterPartBegin; uiRasterPart < uiRasterPartEnd; uiRasterPart += uiIdxStep)
    {
        UInt uiPartAbove;
        const TComDataCU* pcCUAbove = pcCU->getPUAbove(uiPartAbove, g_auiRasterToZscan[uiRasterPart]);
        if (pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
        {
            if (pcCUAbove && pcCUAbove->isIntra(uiPartAbove))
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        else
        {
            if (pcCUAbove)
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        pbValidFlags++;
    }
    return iNumIntra;
}

Int isLeftAvailable(const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool *bValidFlags)
{
    const UInt uiRasterPartBegin = g_auiZscanToRaster[uiPartIdxLT];
    const UInt uiRasterPartEnd = g_auiZscanToRaster[uiPartIdxLB]+1;
    const UInt uiIdxStep = pcCU->getPic()->getNumPartInCtuWidth();
    Bool *pbValidFlags = bValidFlags;
    Int iNumIntra = 0;
    
    for (UInt uiRasterPart = uiRasterPartBegin; uiRasterPart < uiRasterPartEnd; uiRasterPart += uiIdxStep)
    {
        UInt uiPartLeft;
        const TComDataCU* pcCULeft = pcCU->getPULeft(uiPartLeft, g_auiRasterToZscan[uiRasterPart]);
        if (pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
        {
            if (pcCULeft && pcCULeft->isIntra(uiPartLeft))
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        else
        {
            if (pcCULeft)
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        pbValidFlags--;
    }
    return iNumIntra;
}

Int isAboveRightAvailable(const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxRT, Bool *bValidFlags)
{
    const UInt uiNumUnitsInPU = g_auiZscanToRaster[uiPartIdxRT] - g_auiZscanToRaster[uiPartIdxLT] + 1;
    Bool *pbValidFlags = bValidFlags;
    Int iNumIntra = 0;
    
    for (UInt uiOffset = 1; uiOffset <= uiNumUnitsInPU; uiOffset++)
    {
        UInt uiPartAboveRight;
        const TComDataCU* pcCUAboveRight = pcCU->getPUAboveRight(uiPartAboveRight, uiPartIdxRT, uiOffset);
        if (pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
        {
            if (pcCUAboveRight && pcCUAboveRight->isIntra(uiPartAboveRight))
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        else
        {
            if (pcCUAboveRight)
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        pbValidFlags++;
    }
    return iNumIntra;
}

Int isBelowLeftAvailable(const TComDataCU* pcCU, UInt uiPartIdxLT, UInt uiPartIdxLB, Bool *bValidFlags)
{
    const UInt uiNumUnitsInPU = (g_auiZscanToRaster[uiPartIdxLB] - g_auiZscanToRaster[uiPartIdxLT])/pcCU->getPic()->getNumPartInCtuWidth() + 1;
    Bool *pbValidFlags = bValidFlags;
    Int iNumIntra = 0;
    
    for (UInt uiOffset = 1; uiOffset <= uiNumUnitsInPU; uiOffset++)
    {
        UInt uiPartBelowLeft;
        const TComDataCU* pcCUBelowLeft = pcCU->getPUBelowLeft(uiPartBelowLeft, uiPartIdxLB, uiOffset);
        if (pcCU->getSlice()->getPPS()->getConstrainedIntraPred())
        {
            if (pcCUBelowLeft && pcCUBelowLeft->isIntra(uiPartBelowLeft))
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        else
        {
            if (pcCUBelowLeft)
            {
                iNumIntra++;
                *pbValidFlags = true;
            }
            else
            {
                *pbValidFlags = false;
            }
        }
        pbValidFlags--;
    }
    return iNumIntra;
}
