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

#ifndef __TCOMPREDICTION__
#define __TCOMPREDICTION__

#include "TComYuv.h"
#include "TComInterpolationFilter.h"
#include "TComWeightPrediction.h"

/*
Note that, in "integration_prediction_neural_network.h", the
macro `NOMINMAX` is defined when using Windows.
*/
#include "integration_prediction_neural_network.h"
#include "interface_c_python.h"
#include "tools.h"

class TComMv;
class TComTU;

typedef enum PRED_BUF_E
{
    PRED_BUF_UNFILTERED=0,
    PRED_BUF_FILTERED=1,
    NUM_PRED_BUF=2
} PRED_BUF;

static const UInt MAX_INTRA_FILTER_DEPTHS=5;

class TComPrediction : public TComWeightPrediction
{
private:
    static const UChar m_aucIntraFilter[MAX_NUM_CHANNEL_TYPE][MAX_INTRA_FILTER_DEPTHS];

protected:
    std::vector<std::unique_ptr<tensorflow::Session>> m_vector_unique_ptrs_session; /**< Vector of unique pointers to sessions. */
    std::vector<tensorflow::Tensor> m_tensors_flattened_context;                    /**< Vector of flattened masked contexts when the width of the target patch is in {4, 8}. */
    std::vector<tensorflow::Tensor> m_tensors_portion_above;                        /**< Vector of masked context portions located above the target patch when the width of the target patch is in {16, 32, 64}. */
    std::vector<tensorflow::Tensor> m_tensors_portion_left;                         /**< Vector of masked context portions located on the left side of the target patch when the width of the target patch is in {16, 32, 64}. */
    float m_meanTraining;                                                           /**< Mean pixels luminance computed over different luminance images. */
    
    Pel* m_piYuvExt[MAX_NUM_COMPONENT][NUM_PRED_BUF];
    Int m_iYuvExtSize;
    TComYuv m_acYuvPred[NUM_REF_PIC_LIST_01];
    TComYuv m_cYuvPredTemp;
    TComYuv m_filteredBlock[LUMA_INTERPOLATION_FILTER_SUB_SAMPLE_POSITIONS][LUMA_INTERPOLATION_FILTER_SUB_SAMPLE_POSITIONS];
    TComYuv m_filteredBlockTmp[LUMA_INTERPOLATION_FILTER_SUB_SAMPLE_POSITIONS];
    TComInterpolationFilter m_if;
    Pel* m_pLumaRecBuffer;
    Int m_iLumaRecStride;
    
    Void xPredIntraAng(Int bitDepth,
                       const Pel* pSrc,
                       Int srcStride,
                       Pel* pDst,
                       Int dstStride,
                       UInt width,
                       UInt height,
                       ChannelType channelType,
                       UInt dirMode,
                       const Bool bEnableEdgeFilters);
    
    Void xPredIntraPlanar(const Pel* pSrc,
                          Int srcStride,
                          Pel* rpDst,
                          Int dstStride,
                          UInt width,
                          UInt height);

    Void xPredInterUni(TComDataCU* pcCU,
                       UInt uiPartAddr,
                       Int iWidth,
                       Int iHeight,
                       RefPicList eRefPicList,
                       TComYuv* pcYuvPred,
                       Bool bi=false);
    
    Void xPredInterBi(TComDataCU* pcCU,
                      UInt uiPartAddr,
                      Int iWidth,
                      Int iHeight,
                      TComYuv* pcYuvPred);
    
    Void xPredInterBlk(const ComponentID compID,
                       TComDataCU* cu,
                       TComPicYuv* refPic,
                       UInt partAddr,
                       TComMv* mv,
                       Int width,
                       Int height,
                       TComYuv* dstPic,
                       Bool bi,
                       const Int bitDepth);
    
    Void xWeightedAverage(TComYuv* pcYuvSrc0,
                          TComYuv* pcYuvSrc1,
                          Int iRefIdx0,
                          Int iRefIdx1,
                          UInt uiPartAddr,
                          Int iWidth,
                          Int iHeight,
                          TComYuv* pcYuvDst,
                          const BitDepths& clipBitDepths);
    
    Void xGetLLSPrediction(const Pel* pSrc0,
                           Int iSrcStride,
                           Pel* pDst0,
                           Int iDstStride,
                           UInt uiWidth,
                           UInt uiHeight,
                           UInt uiExt0,
                           const ChromaFormat chFmt DEBUG_STRING_FN_DECLARE(sDebug));
    
    Void xDCPredFiltering(const Pel* pSrc,
                          Int iSrcStride,
                          Pel* pDst,
                          Int iDstStride,
                          Int iWidth,
                          Int iHeight,
                          ChannelType channelType);
    
    Bool xCheckIdenticalMotion(TComDataCU* pcCU,
                               UInt PartAddr);
    
    Void destroy();

public:
    TComPrediction();
    
    virtual ~TComPrediction();
    
    /*
    It is a modification of the HEVC code for integrating
    the prediction neural networks.
    */
    Void initTempBuff(ChromaFormat chromaFormatIDC,
                      const std::string& path_to_additional_directory,
                      const std::string& path_to_mean_training,
                      const std::string& path_to_file_paths_to_graphs_output,
                      const int& qp_selection);
    
    ChromaFormat getChromaFormat() const { return m_cYuvPredTemp.getChromaFormat(); }
    
    Void motionCompensation(TComDataCU* pcCU,
                            TComYuv* pcYuvPred,
                            RefPicList eRefPicList=REF_PIC_LIST_X,
                            Int iPartIdx=-1);
    
    Void getMvPredAMVP(TComDataCU* pcCU,
                       UInt uiPartIdx,
                       UInt uiPartAddr,
                       RefPicList eRefPicList,
                       TComMv& rcMvPred);
    
    Void predIntraAng(const ComponentID compID,
                      UInt uiDirMode,
                      Pel* piOrg /* Will be null for decoding */,
                      UInt uiOrgStride,
                      Pel* piPred,
                      UInt uiStride,
                      TComTU &rTu,
                      const bool& contextFlag,
                      const Bool bUseFilteredPredSamples,
                      const Bool bUseLosslessDPCM=false);
    
    Pel predIntraGetPredValDC(const Pel* pSrc,
                              Int iSrcStride,
                              UInt iWidth,
                              UInt iHeight);
    
    Pel* getPredictorPtr(const ComponentID compID,
                         const Bool bUseFilteredPredictions)
    {
        return m_piYuvExt[compID][bUseFilteredPredictions?PRED_BUF_FILTERED:PRED_BUF_UNFILTERED];
    }
    
    Void initIntraPatternChType(TComTU &rTu,
                                bool& contextFlag,
                                const ComponentID compID,
                                const Bool bFilterRefSamples DEBUG_STRING_FN_DECLARE(sDebug));
    
    static Bool filteringIntraReferenceSamples(const ComponentID compID,
                                               UInt uiDirMode,
                                               UInt uiTuChWidth,
                                               UInt uiTuChHeight,
                                               const ChromaFormat chFmt,
                                               const Bool intraReferenceSmoothingDisabled);
    
    static Bool UseDPCMForFirstPassIntraEstimation(TComTU &rTu,
                                                   const UInt uiDirMode);
};

#endif
