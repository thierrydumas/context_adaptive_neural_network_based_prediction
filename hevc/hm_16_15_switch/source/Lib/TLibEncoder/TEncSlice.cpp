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

#include "TEncTop.h"
#include "TEncSlice.h"
#include <math.h>

TEncSlice::TEncSlice() :
    m_ptr_map_luminance(NULL),
    m_ptr_map_chrominance(NULL),
    m_encCABACTableIdx(I_SLICE)
{}

TEncSlice::~TEncSlice()
{
    destroy();
}

/*
It is a modification of the HEVC code for collecting
the frequency of use of each HEVC intra prediction mode.
*/
UInt** TEncSlice::getNbFastSelections() const
{
    return m_pcCuEncoder->getNbFastSelections();
}

UInt** TEncSlice::getNbFastWins() const
{
    return m_pcCuEncoder->getNbFastWins();
}

UInt** TEncSlice::getNbRDWins() const
{
    return m_pcCuEncoder->getNbRDWins();
}

UInt* TEncSlice::getNbFastRDRuns() const
{
    return m_pcCuEncoder->getNbFastRDRuns();
}

UInt* TEncSlice::getCumNbModesForRD() const
{
    return m_pcCuEncoder->getCumNbModesForRD();
}

Void TEncSlice::create(Int iWidth,
                       Int iHeight,
                       ChromaFormat chromaFormat,
                       UInt iMaxCUWidth,
                       UInt iMaxCUHeight,
                       UChar uhTotalDepth)
{
    m_picYuvPred.create(iWidth,
                        iHeight,
                        chromaFormat,
                        iMaxCUWidth,
                        iMaxCUHeight,
                        uhTotalDepth,
                        true);
    m_picYuvResi.create(iWidth,
                        iHeight,
                        chromaFormat,
                        iMaxCUWidth,
                        iMaxCUHeight,
                        uhTotalDepth,
                        true);
}

Void TEncSlice::destroy()
{
    m_picYuvPred.destroy();
    m_picYuvResi.destroy();
    m_vdRdPicLambda.clear();
    m_vdRdPicQp.clear();
    m_viRdPicQp.clear();
    
    /*
    The line below deallocates the memory of the array
    of intra prediction modes for the luminance channel.
    */
    if (m_ptr_map_luminance)
    {
        xFree(m_ptr_map_luminance);
        m_ptr_map_luminance = NULL;
    }
    
    /*
    The line below deallocates the memory of the array
    of intra prediction modes for the two chrominance
    channels.
    */
    if (m_ptr_map_chrominance)
    {
        xFree(m_ptr_map_chrominance);
        m_ptr_map_chrominance = NULL;
    }
}

Void TEncSlice::init(TEncTop* pcEncTop)
{
    m_pcCfg = pcEncTop;
    m_pcListPic = pcEncTop->getListPic();
    m_pcGOPEncoder = pcEncTop->getGOPEncoder();
    m_pcCuEncoder = pcEncTop->getCuEncoder();
    m_pcPredSearch = pcEncTop->getPredSearch();
    m_pcEntropyCoder = pcEncTop->getEntropyCoder();
    m_pcSbacCoder = pcEncTop->getSbacCoder();
    m_pcBinCABAC = pcEncTop->getBinCABAC();
    m_pcTrQuant = pcEncTop->getTrQuant();
    m_pcRdCost = pcEncTop->getRdCost();
    m_pppcRDSbacCoder = pcEncTop->getRDSbacCoder();
    m_pcRDGoOnSbacCoder = pcEncTop->getRDGoOnSbacCoder();
    m_vdRdPicLambda.resize(m_pcCfg->getDeltaQpRD()*2 + 1);
    m_vdRdPicQp.resize(m_pcCfg->getDeltaQpRD()*2 + 1);
    m_viRdPicQp.resize(m_pcCfg->getDeltaQpRD()*2 + 1);
    m_pcRateCtrl = pcEncTop->getRateCtrl();
    
    /*
    The memory of the array for the map of intra prediction
    modes for the luminance channel is allocated in the method
    `TEncSlice::init` as `m_pcCfg` must be initialized before
    this memory is allocated. The same goes for the memory of
    the array for the map of intra prediction modes for the
    two chrominance channels.
    */
    /*
    The precedence of . (member access by reference) is
    higher than the precedence of !... (logical NOT).
    */
    const std::string pathToThresholdedMapModesLuminance(m_pcCfg->getPathToThresholdedMapModesLuminance());
    if (!pathToThresholdedMapModesLuminance.empty())
    {
        /*
        The precedence of ... -> ... (member access by pointer)
        is higher than the precedence of *... (dereference) and
        the precedence of ... * ... (multiplication).
        */
        m_ptr_map_luminance = (unsigned char*)xMalloc(unsigned char, m_pcCfg->getSourceHeight()*m_pcCfg->getSourceWidth());
    }
    const std::string pathToThresholdedMapModesChrominance(m_pcCfg->getPathToThresholdedMapModesChrominance());
    
    /*
    If only a luminance channel is encoded via HEVC, no memory is allocated
    for the array for the map of intra prediction modes for the two chrominance
    channels.
    */
    if (!pathToThresholdedMapModesChrominance.empty() && m_picYuvPred.getChromaFormat() != CHROMA_400)
    {
        m_ptr_map_chrominance = (unsigned char*)xMalloc(unsigned char, m_pcCfg->getSourceHeight()*m_pcCfg->getSourceWidth());
    }
}

Void TEncSlice::updateLambda(TComSlice* pSlice, Double dQP)
{
    Int iQP = (Int)dQP;
    Double dLambda = calculateLambda(pSlice,
                                     m_gopID,
                                     pSlice->getDepth(),
                                     pSlice->getSliceQp(),
                                     dQP,
                                     iQP);
    setUpLambda(pSlice,
                dLambda,
                iQP);
}

Void TEncSlice::setUpLambda(TComSlice* slice, const Double dLambda, Int iQP)
{
    m_pcRdCost ->setLambda(dLambda,
                           slice->getSPS()->getBitDepths());
    Double dLambdas[MAX_NUM_COMPONENT] = {dLambda};
    for (UInt compIdx = 1; compIdx < MAX_NUM_COMPONENT; compIdx++)
    {
        const ComponentID compID = ComponentID(compIdx);
        Int chromaQPOffset = slice->getPPS()->getQpOffset(compID) + slice->getSliceChromaQpDelta(compID);
        Int qpc = (iQP + chromaQPOffset < 0) ? iQP : getScaledChromaQP(iQP + chromaQPOffset, m_pcCfg->getChromaFormatIdc());
        Double tmpWeight = pow(2.0, (iQP - qpc)/3.0);
        m_pcRdCost->setDistortionWeight(compID,
                                        tmpWeight);
        dLambdas[compIdx] = dLambda/tmpWeight;
    }
#if RDOQ_CHROMA_LAMBDA
    m_pcTrQuant->setLambdas(dLambdas);
#else
    m_pcTrQuant->setLambda(dLambda);
#endif
    slice->setLambdas(dLambdas);
}

Void TEncSlice::initEncSlice(TComPic* pcPic, const Int pocLast, const Int pocCurr, const Int iGOPid, TComSlice*& rpcSlice, const Bool isField)
{
  Double dQP;
  Double dLambda;

  rpcSlice = pcPic->getSlice(0);
  rpcSlice->setSliceBits(0);
  rpcSlice->setPic( pcPic );
  rpcSlice->initSlice();
  rpcSlice->setPicOutputFlag( true );
  rpcSlice->setPOC( pocCurr );
  pcPic->setField(isField);
  m_gopID = iGOPid;

  // depth computation based on GOP size
  Int depth;
  {
    Int poc = rpcSlice->getPOC();
    if(isField)
    {
      poc = (poc/2) % (m_pcCfg->getGOPSize()/2);
    }
    else
    {
      poc = poc % m_pcCfg->getGOPSize();   
    }

    if ( poc == 0 )
    {
      depth = 0;
    }
    else
    {
      Int step = m_pcCfg->getGOPSize();
      depth    = 0;
      for( Int i=step>>1; i>=1; i>>=1 )
      {
        for ( Int j=i; j<m_pcCfg->getGOPSize(); j+=step )
        {
          if ( j == poc )
          {
            i=0;
            break;
          }
        }
        step >>= 1;
        depth++;
      }
    }

    if(m_pcCfg->getHarmonizeGopFirstFieldCoupleEnabled() && poc != 0)
    {
      if (isField && ((rpcSlice->getPOC() % 2) == 1))
      {
        depth ++;
      }
    }
  }

  // slice type
  SliceType eSliceType;

  eSliceType=B_SLICE;
  if(!(isField && pocLast == 1) || !m_pcCfg->getEfficientFieldIRAPEnabled())
  {
    if(m_pcCfg->getDecodingRefreshType() == 3)
    {
      eSliceType = (pocLast == 0 || pocCurr % m_pcCfg->getIntraPeriod() == 0             || m_pcGOPEncoder->getGOPSize() == 0) ? I_SLICE : eSliceType;
    }
    else
    {
      eSliceType = (pocLast == 0 || (pocCurr - (isField ? 1 : 0)) % m_pcCfg->getIntraPeriod() == 0 || m_pcGOPEncoder->getGOPSize() == 0) ? I_SLICE : eSliceType;
    }
  }

  rpcSlice->setSliceType    ( eSliceType );

  // ------------------------------------------------------------------------------------------------------------------
  // Non-referenced frame marking
  // ------------------------------------------------------------------------------------------------------------------

  if(pocLast == 0)
  {
    rpcSlice->setTemporalLayerNonReferenceFlag(false);
  }
  else
  {
    rpcSlice->setTemporalLayerNonReferenceFlag(!m_pcCfg->getGOPEntry(iGOPid).m_refPic);
  }
  rpcSlice->setReferenced(true);

  // ------------------------------------------------------------------------------------------------------------------
  // QP setting
  // ------------------------------------------------------------------------------------------------------------------

#if X0038_LAMBDA_FROM_QP_CAPABILITY
  dQP = m_pcCfg->getQPForPicture(iGOPid, rpcSlice);
#else
  dQP = m_pcCfg->getQP();
  if(eSliceType!=I_SLICE)
  {
    if (!(( m_pcCfg->getMaxDeltaQP() == 0) && (!m_pcCfg->getLumaLevelToDeltaQPMapping().isEnabled()) && (dQP == -rpcSlice->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA) ) && (rpcSlice->getPPS()->getTransquantBypassEnabledFlag())))
    {
      dQP += m_pcCfg->getGOPEntry(iGOPid).m_QPOffset;
    }
  }

  // modify QP
  const Int* pdQPs = m_pcCfg->getdQPs();
  if ( pdQPs )
  {
    dQP += pdQPs[ rpcSlice->getPOC() ];
  }

  if (m_pcCfg->getCostMode()==COST_LOSSLESS_CODING)
  {
    dQP=LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP;
    m_pcCfg->setDeltaQpRD(0);
  }
#endif

  // ------------------------------------------------------------------------------------------------------------------
  // Lambda computation
  // ------------------------------------------------------------------------------------------------------------------

#if X0038_LAMBDA_FROM_QP_CAPABILITY
  const Int temporalId=m_pcCfg->getGOPEntry(iGOPid).m_temporalId;
#endif
  Int iQP;
  Double dOrigQP = dQP;

  // pre-compute lambda and QP values for all possible QP candidates
  for ( Int iDQpIdx = 0; iDQpIdx < 2 * m_pcCfg->getDeltaQpRD() + 1; iDQpIdx++ )
  {
    // compute QP value
    dQP = dOrigQP + ((iDQpIdx+1)>>1)*(iDQpIdx%2 ? -1 : 1);
    dLambda = calculateLambda(rpcSlice, iGOPid, depth, dQP, dQP, iQP );

    m_vdRdPicLambda[iDQpIdx] = dLambda;
    m_vdRdPicQp    [iDQpIdx] = dQP;
    m_viRdPicQp    [iDQpIdx] = iQP;
  }

  // obtain dQP = 0 case
  dLambda = m_vdRdPicLambda[0];
  dQP     = m_vdRdPicQp    [0];
  iQP     = m_viRdPicQp    [0];

#if !X0038_LAMBDA_FROM_QP_CAPABILITY
  const Int temporalId=m_pcCfg->getGOPEntry(iGOPid).m_temporalId;
  const std::vector<Double> &intraLambdaModifiers=m_pcCfg->getIntraLambdaModifier();
#endif

  if(rpcSlice->getPPS()->getSliceChromaQpFlag())
  {
    const Bool bUseIntraOrPeriodicOffset = rpcSlice->getSliceType()==I_SLICE || (m_pcCfg->getSliceChromaOffsetQpPeriodicity()!=0 && (rpcSlice->getPOC()%m_pcCfg->getSliceChromaOffsetQpPeriodicity())==0);
    Int cbQP = bUseIntraOrPeriodicOffset? m_pcCfg->getSliceChromaOffsetQpIntraOrPeriodic(false) : m_pcCfg->getGOPEntry(iGOPid).m_CbQPoffset;
    Int crQP = bUseIntraOrPeriodicOffset? m_pcCfg->getSliceChromaOffsetQpIntraOrPeriodic(true)  : m_pcCfg->getGOPEntry(iGOPid).m_CrQPoffset;

    cbQP = Clip3( -12, 12, cbQP + rpcSlice->getPPS()->getQpOffset(COMPONENT_Cb) ) - rpcSlice->getPPS()->getQpOffset(COMPONENT_Cb); 
    crQP = Clip3( -12, 12, crQP + rpcSlice->getPPS()->getQpOffset(COMPONENT_Cr) ) - rpcSlice->getPPS()->getQpOffset(COMPONENT_Cr); 
    rpcSlice->setSliceChromaQpDelta(COMPONENT_Cb, Clip3( -12, 12, cbQP));
    assert(rpcSlice->getSliceChromaQpDelta(COMPONENT_Cb)+rpcSlice->getPPS()->getQpOffset(COMPONENT_Cb)<=12 && rpcSlice->getSliceChromaQpDelta(COMPONENT_Cb)+rpcSlice->getPPS()->getQpOffset(COMPONENT_Cb)>=-12);
    rpcSlice->setSliceChromaQpDelta(COMPONENT_Cr, Clip3( -12, 12, crQP));
    assert(rpcSlice->getSliceChromaQpDelta(COMPONENT_Cr)+rpcSlice->getPPS()->getQpOffset(COMPONENT_Cr)<=12 && rpcSlice->getSliceChromaQpDelta(COMPONENT_Cr)+rpcSlice->getPPS()->getQpOffset(COMPONENT_Cr)>=-12);
  }
  else
  {
    rpcSlice->setSliceChromaQpDelta( COMPONENT_Cb, 0 );
    rpcSlice->setSliceChromaQpDelta( COMPONENT_Cr, 0 );
  }

#if !X0038_LAMBDA_FROM_QP_CAPABILITY
  Double lambdaModifier;
  if( rpcSlice->getSliceType( ) != I_SLICE || intraLambdaModifiers.empty())
  {
    lambdaModifier = m_pcCfg->getLambdaModifier( temporalId );
  }
  else
  {
    lambdaModifier = intraLambdaModifiers[ (temporalId < intraLambdaModifiers.size()) ? temporalId : (intraLambdaModifiers.size()-1) ];
  }

  dLambda *= lambdaModifier;
#endif

  setUpLambda(rpcSlice, dLambda, iQP);

  if (m_pcCfg->getFastMEForGenBLowDelayEnabled())
  {
    // restore original slice type

    if(!(isField && pocLast == 1) || !m_pcCfg->getEfficientFieldIRAPEnabled())
    {
      if(m_pcCfg->getDecodingRefreshType() == 3)
      {
        eSliceType = (pocLast == 0 || (pocCurr)                     % m_pcCfg->getIntraPeriod() == 0 || m_pcGOPEncoder->getGOPSize() == 0) ? I_SLICE : eSliceType;
      }
      else
      {
        eSliceType = (pocLast == 0 || (pocCurr - (isField ? 1 : 0)) % m_pcCfg->getIntraPeriod() == 0 || m_pcGOPEncoder->getGOPSize() == 0) ? I_SLICE : eSliceType;
      }
    }

    rpcSlice->setSliceType        ( eSliceType );
  }

  if (m_pcCfg->getUseRecalculateQPAccordingToLambda())
  {
    dQP = xGetQPValueAccordingToLambda( dLambda );
    iQP = max( -rpcSlice->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA), min( MAX_QP, (Int) floor( dQP + 0.5 ) ) );
  }

  rpcSlice->setSliceQp           ( iQP );
#if ADAPTIVE_QP_SELECTION
  rpcSlice->setSliceQpBase       ( iQP );
#endif
  rpcSlice->setSliceQpDelta      ( 0 );
  rpcSlice->setUseChromaQpAdj( rpcSlice->getPPS()->getPpsRangeExtension().getChromaQpOffsetListEnabledFlag() );
  rpcSlice->setNumRefIdx(REF_PIC_LIST_0,m_pcCfg->getGOPEntry(iGOPid).m_numRefPicsActive);
  rpcSlice->setNumRefIdx(REF_PIC_LIST_1,m_pcCfg->getGOPEntry(iGOPid).m_numRefPicsActive);

  if ( m_pcCfg->getDeblockingFilterMetric() )
  {
    rpcSlice->setDeblockingFilterOverrideFlag(true);
    rpcSlice->setDeblockingFilterDisable(false);
    rpcSlice->setDeblockingFilterBetaOffsetDiv2( 0 );
    rpcSlice->setDeblockingFilterTcOffsetDiv2( 0 );
  }
  else if (rpcSlice->getPPS()->getDeblockingFilterControlPresentFlag())
  {
    rpcSlice->setDeblockingFilterOverrideFlag( rpcSlice->getPPS()->getDeblockingFilterOverrideEnabledFlag() );
    rpcSlice->setDeblockingFilterDisable( rpcSlice->getPPS()->getPPSDeblockingFilterDisabledFlag() );
    if ( !rpcSlice->getDeblockingFilterDisable())
    {
      if ( rpcSlice->getDeblockingFilterOverrideFlag() && eSliceType!=I_SLICE)
      {
        rpcSlice->setDeblockingFilterBetaOffsetDiv2( m_pcCfg->getGOPEntry(iGOPid).m_betaOffsetDiv2 + m_pcCfg->getLoopFilterBetaOffset()  );
        rpcSlice->setDeblockingFilterTcOffsetDiv2( m_pcCfg->getGOPEntry(iGOPid).m_tcOffsetDiv2 + m_pcCfg->getLoopFilterTcOffset() );
      }
      else
      {
        rpcSlice->setDeblockingFilterBetaOffsetDiv2( m_pcCfg->getLoopFilterBetaOffset() );
        rpcSlice->setDeblockingFilterTcOffsetDiv2( m_pcCfg->getLoopFilterTcOffset() );
      }
    }
  }
  else
  {
    rpcSlice->setDeblockingFilterOverrideFlag( false );
    rpcSlice->setDeblockingFilterDisable( false );
    rpcSlice->setDeblockingFilterBetaOffsetDiv2( 0 );
    rpcSlice->setDeblockingFilterTcOffsetDiv2( 0 );
  }

  rpcSlice->setDepth            ( depth );

  pcPic->setTLayer( temporalId );
  if(eSliceType==I_SLICE)
  {
    pcPic->setTLayer(0);
  }
  rpcSlice->setTLayer( pcPic->getTLayer() );

  pcPic->setPicYuvPred( &m_picYuvPred );
  pcPic->setPicYuvResi( &m_picYuvResi );
  rpcSlice->setSliceMode            ( m_pcCfg->getSliceMode()            );
  rpcSlice->setSliceArgument        ( m_pcCfg->getSliceArgument()        );
  rpcSlice->setSliceSegmentMode     ( m_pcCfg->getSliceSegmentMode()     );
  rpcSlice->setSliceSegmentArgument ( m_pcCfg->getSliceSegmentArgument() );
  rpcSlice->setMaxNumMergeCand      ( m_pcCfg->getMaxNumMergeCand()      );
}


Double TEncSlice::calculateLambda( const TComSlice* slice,
                                   const Int        GOPid, // entry in the GOP table
                                   const Int        depth, // slice GOP hierarchical depth.
                                   const Double     refQP, // initial slice-level QP
                                   const Double     dQP,   // initial double-precision QP
                                         Int       &iQP )  // returned integer QP.
{
  enum   SliceType eSliceType    = slice->getSliceType();
  const  Bool      isField       = slice->getPic()->isField();
  const  Int       NumberBFrames = ( m_pcCfg->getGOPSize() - 1 );
  const  Int       SHIFT_QP      = 12;
#if X0038_LAMBDA_FROM_QP_CAPABILITY
  const Int temporalId=m_pcCfg->getGOPEntry(GOPid).m_temporalId;
  const std::vector<Double> &intraLambdaModifiers=m_pcCfg->getIntraLambdaModifier();
#endif

#if FULL_NBIT
  Int    bitdepth_luma_qp_scale = 6 * (slice->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) - 8);
#else
  Int    bitdepth_luma_qp_scale = 0;
#endif
  Double qp_temp = dQP + bitdepth_luma_qp_scale - SHIFT_QP;
  // Case #1: I or P-slices (key-frame)
  Double dQPFactor = m_pcCfg->getGOPEntry(GOPid).m_QPFactor;
  if ( eSliceType==I_SLICE )
  {
    if (m_pcCfg->getIntraQpFactor()>=0.0 && m_pcCfg->getGOPEntry(GOPid).m_sliceType != I_SLICE)
    {
      dQPFactor=m_pcCfg->getIntraQpFactor();
    }
    else
    {
#if X0038_LAMBDA_FROM_QP_CAPABILITY
      if(m_pcCfg->getLambdaFromQPEnable())
      {
        dQPFactor=0.57;
      }
      else
      {
#endif
        Double dLambda_scale = 1.0 - Clip3( 0.0, 0.5, 0.05*(Double)(isField ? NumberBFrames/2 : NumberBFrames) );
        dQPFactor=0.57*dLambda_scale;
#if X0038_LAMBDA_FROM_QP_CAPABILITY
      }
#endif
    }
  }
#if X0038_LAMBDA_FROM_QP_CAPABILITY
  else if( m_pcCfg->getLambdaFromQPEnable() )
  {
    dQPFactor=0.57;
  }
#endif

  Double dLambda = dQPFactor*pow( 2.0, qp_temp/3.0 );

#if X0038_LAMBDA_FROM_QP_CAPABILITY
  if( !(m_pcCfg->getLambdaFromQPEnable()) && depth>0 )
#else
  if ( depth>0 )
#endif
  {
#if FULL_NBIT
      Double qp_temp_ref_orig = refQP - SHIFT_QP;
      dLambda *= Clip3( 2.00, 4.00, (qp_temp_ref_orig / 6.0) ); // (j == B_SLICE && p_cur_frm->layer != 0 )
#else
      Double qp_temp_ref = refQP + bitdepth_luma_qp_scale - SHIFT_QP;
      dLambda *= Clip3( 2.00, 4.00, (qp_temp_ref / 6.0) ); // (j == B_SLICE && p_cur_frm->layer != 0 )
#endif
  }

  // if hadamard is used in ME process
  if ( !m_pcCfg->getUseHADME() && slice->getSliceType( ) != I_SLICE )
  {
    dLambda *= 0.95;
  }

#if X0038_LAMBDA_FROM_QP_CAPABILITY
  Double lambdaModifier;
  if( eSliceType != I_SLICE || intraLambdaModifiers.empty())
  {
    lambdaModifier = m_pcCfg->getLambdaModifier( temporalId );
  }
  else
  {
    lambdaModifier = intraLambdaModifiers[ (temporalId < intraLambdaModifiers.size()) ? temporalId : (intraLambdaModifiers.size()-1) ];
  }
  dLambda *= lambdaModifier;
#endif

  iQP = max( -slice->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA), min( MAX_QP, (Int) floor( dQP + 0.5 ) ) );
  
  // NOTE: the lambda modifiers that are sometimes applied later might be best always applied in here.
  return dLambda;
}

Void TEncSlice::resetQP( TComPic* pic, Int sliceQP, Double lambda )
{
  TComSlice* slice = pic->getSlice(0);

  // store lambda
  slice->setSliceQp( sliceQP );
#if ADAPTIVE_QP_SELECTION
  slice->setSliceQpBase ( sliceQP );
#endif
  setUpLambda(slice, lambda, sliceQP);
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

Void TEncSlice::setSearchRange(TComSlice* pcSlice)
{
  Int iCurrPOC = pcSlice->getPOC();
  Int iRefPOC;
  Int iGOPSize = m_pcCfg->getGOPSize();
  Int iOffset = (iGOPSize >> 1);
  Int iMaxSR = m_pcCfg->getSearchRange();
  Int iNumPredDir = pcSlice->isInterP() ? 1 : 2;
  for (Int iDir = 0; iDir < iNumPredDir; iDir++)
  {
      RefPicList e = (iDir ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
      for (Int iRefIdx = 0; iRefIdx < pcSlice->getNumRefIdx(e); iRefIdx++)
      {
          iRefPOC = pcSlice->getRefPic(e, iRefIdx)->getPOC();
          Int newSearchRange = Clip3(m_pcCfg->getMinSearchWindow(), iMaxSR, (iMaxSR*ADAPT_SR_SCALE*abs(iCurrPOC - iRefPOC)+iOffset)/iGOPSize);
          m_pcPredSearch->setAdaptiveSearchRange(iDir, iRefIdx, newSearchRange);
      }
  }
}

Void TEncSlice::precompressSlice(TComPic* pcPic)
{
  /*
  The precompression of the slice is not performed
  if DeltaQPRD is not used.
  */
  if (m_pcCfg->getDeltaQpRD() == 0)
  {
      return;
  }
  if (m_pcCfg->getUseRateCtrl())
  {
      printf( "\nMultiple QP optimization is not allowed when rate control is enabled.");
      assert(0);
      return;
  }
  TComSlice* pcSlice = pcPic->getSlice(getSliceIdx());
  if (pcSlice->getDependentSliceSegmentFlag())
  {
      return;
  }
  if (pcSlice->getSliceMode() == FIXED_NUMBER_OF_BYTES)
  {
      printf( "\nUnable to optimise Slice-level QP if Slice Mode is set to FIXED_NUMBER_OF_BYTES\n" );
      assert(0);
      return;
  }
  Double dPicRdCostBest = MAX_DOUBLE;
  UInt uiQpIdxBest = 0;
  Double dFrameLambda;
#if FULL_NBIT
  Int SHIFT_QP = 12 + 6*(pcSlice->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) - 8);
#else
  Int SHIFT_QP = 12;
#endif
  if (m_pcCfg->getGOPSize() > 1)
  {
      dFrameLambda = 0.68*pow(2, (m_viRdPicQp[0] - SHIFT_QP)/3.0)*(pcSlice->isInterB() ? 2 : 1);
  }
  else
  {
      dFrameLambda = 0.68*pow (2, (m_viRdPicQp[0] - SHIFT_QP)/3.0);
  }
  m_pcRdCost->setFrameLambda(dFrameLambda);
  for (UInt uiQpIdx = 0; uiQpIdx < 2*m_pcCfg->getDeltaQpRD() + 1; uiQpIdx++)
  {
      pcSlice->setSliceQp(m_viRdPicQp[uiQpIdx]);
#if ADAPTIVE_QP_SELECTION
      pcSlice->setSliceQpBase(m_viRdPicQp[uiQpIdx]);
#endif
      setUpLambda(pcSlice, m_vdRdPicLambda[uiQpIdx], m_viRdPicQp[uiQpIdx]);
      compressSlice(pcPic, true, m_pcCfg->getFastDeltaQp());
      UInt64 uiPicDist = m_uiPicDist;
      Double dPicRdCost = m_pcRdCost->calcRdCost((Double)m_uiPicTotalBits, (Double)uiPicDist, DF_SSE_FRAME);
      if (dPicRdCost < dPicRdCostBest)
      {
          uiQpIdxBest = uiQpIdx;
          dPicRdCostBest = dPicRdCost;
      }
  }
  pcSlice->setSliceQp(m_viRdPicQp[uiQpIdxBest]);
#if ADAPTIVE_QP_SELECTION
  pcSlice->setSliceQpBase(m_viRdPicQp[uiQpIdxBest]);
#endif
  setUpLambda(pcSlice, m_vdRdPicLambda[uiQpIdxBest], m_viRdPicQp[uiQpIdxBest]);
}

Void TEncSlice::calCostSliceI(TComPic* pcPic)
{
  Double iSumHadSlice = 0;
  TComSlice* const pcSlice = pcPic->getSlice(getSliceIdx());
  const TComSPS &sps = *(pcSlice->getSPS());
  const Int shift = sps.getBitDepth(CHANNEL_TYPE_LUMA) - 8;
  const Int offset = (shift > 0) ? (1 << (shift - 1)) : 0;
  pcSlice->setSliceSegmentBits(0);
  UInt startCtuTsAddr, boundingCtuTsAddr;
  xDetermineStartAndBoundingCtuTsAddr(startCtuTsAddr,
                                      boundingCtuTsAddr,
                                      pcPic);
  for (UInt ctuTsAddr = startCtuTsAddr, ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTsAddr); ctuTsAddr < boundingCtuTsAddr; ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(++ctuTsAddr))
  {
      TComDataCU* pCtu = pcPic->getCtu(ctuRsAddr);
      pCtu->initCtu(pcPic, ctuRsAddr);
      Int height = min(sps.getMaxCUHeight(),sps.getPicHeightInLumaSamples() - ctuRsAddr/pcPic->getFrameWidthInCtus()*sps.getMaxCUHeight());
      Int width = min(sps.getMaxCUWidth(), sps.getPicWidthInLumaSamples() - ctuRsAddr % pcPic->getFrameWidthInCtus()*sps.getMaxCUWidth());
      Int iSumHad = m_pcCuEncoder->updateCtuDataISlice(pCtu, width, height);
      (m_pcRateCtrl->getRCPic()->getLCU(ctuRsAddr)).m_costIntra = (iSumHad + offset) >> shift;
      iSumHadSlice += (m_pcRateCtrl->getRCPic()->getLCU(ctuRsAddr)).m_costIntra;
  }
  m_pcRateCtrl->getRCPic()->setTotalIntraCost(iSumHadSlice);
}

Void TEncSlice::compressSlice(TComPic* pcPic, const Bool bCompressEntireSlice, const Bool bFastDeltaQP)
{
  UInt startCtuTsAddr;
  UInt boundingCtuTsAddr;
  TComSlice* const pcSlice = pcPic->getSlice(getSliceIdx());
  pcSlice->setSliceSegmentBits(0);
  xDetermineStartAndBoundingCtuTsAddr(startCtuTsAddr, boundingCtuTsAddr, pcPic);
  if (bCompressEntireSlice)
  {
      boundingCtuTsAddr = pcSlice->getSliceCurEndCtuTsAddr();
      pcSlice->setSliceSegmentCurEndCtuTsAddr(boundingCtuTsAddr);
  }
  m_uiPicTotalBits = 0;
  m_dPicRdCost = 0;
  m_uiPicDist = 0;
  m_pcEntropyCoder->setEntropyCoder(m_pppcRDSbacCoder[0][CI_CURR_BEST]);
  m_pcEntropyCoder->resetEntropy(pcSlice);
  TEncBinCABAC* pRDSbacCoder = (TEncBinCABAC*)m_pppcRDSbacCoder[0][CI_CURR_BEST]->getEncBinIf();
  pRDSbacCoder->setBinCountingEnableFlag(false);
  pRDSbacCoder->setBinsCoded(0);
  TComBitCounter tempBitCounter;
  const UInt frameWidthInCtus = pcPic->getPicSym()->getFrameWidthInCtus();
  m_pcCuEncoder->setFastDeltaQp(bFastDeltaQP);
  if (pcSlice->getPPS()->getUseWP() || pcSlice->getPPS()->getWPBiPred())
  {
      xCalcACDCParamSlice(pcSlice);
  }
  const Bool bWp_explicit = (pcSlice->getSliceType() == P_SLICE && pcSlice->getPPS()->getUseWP()) || (pcSlice->getSliceType() == B_SLICE && pcSlice->getPPS()->getWPBiPred());
  if (bWp_explicit)
  {
    if (pcSlice->getSliceMode() == FIXED_NUMBER_OF_BYTES || pcSlice->getSliceSegmentMode() == FIXED_NUMBER_OF_BYTES)
    {
        printf("Weighted Prediction is not supported with slice mode determined by max number of bins.\n");
        exit(0);
    }
    xEstimateWPParamSlice(pcSlice, m_pcCfg->getWeightedPredictionMethod());
    pcSlice->initWpScaling(pcSlice->getSPS());
    xCheckWPEnable(pcSlice);
  }
#if ADAPTIVE_QP_SELECTION
  if (m_pcCfg->getUseAdaptQpSelect() && !(pcSlice->getDependentSliceSegmentFlag()))
  {
      m_pcTrQuant->clearSliceARLCnt();
      if(pcSlice->getSliceType() != I_SLICE)
      {
          Int qpBase = pcSlice->getSliceQpBase();
          pcSlice->setSliceQp(qpBase + m_pcTrQuant->getQpDelta(qpBase));
      }
  }
#endif
  {
      const UInt ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTsAddr);
      const UInt currentTileIdx = pcPic->getPicSym()->getTileIdxMap(ctuRsAddr);
      const TComTile* pCurrentTile = pcPic->getPicSym()->getTComTile(currentTileIdx);
      const UInt firstCtuRsAddrOfTile = pCurrentTile->getFirstCtuRsAddr();
      if (pcSlice->getDependentSliceSegmentFlag() && ctuRsAddr != firstCtuRsAddrOfTile)
      {
          if( pCurrentTile->getTileWidthInCtus() >= 2 || !m_pcCfg->getEntropyCodingSyncEnabledFlag() )
          {
              m_pppcRDSbacCoder[0][CI_CURR_BEST]->loadContexts(&m_lastSliceSegmentEndContextState);
          }
      }
  }
  
  /*
  In the standard cases, `startCtuTsAddr` is equal to 0.
  `boundingCtuTsAddr` is equal to the height of the image
  divided by the CTU width times the width of the image
  divided by the CTU width.
  */
  // Below is the raster-scanning of the CTUs.
  for (UInt ctuTsAddr = startCtuTsAddr; ctuTsAddr < boundingCtuTsAddr; ++ctuTsAddr)
  {
      const UInt ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(ctuTsAddr);
      TComDataCU* pCtu = pcPic->getCtu(ctuRsAddr);
      
      /*
      In `TComDataCU::initCtu`, the memory of the buffer
      storing the selected intra prediction mode for each
      PU in the current CTU is allocated.
      The raster scan address of the current CTU `ctuRsAddr`
      is stored in the private attribute `m_ctuRsAddr` of
      the instance of class `TComDataCU` pointed by `pCtu`.
      */
      pCtu->initCtu(pcPic, ctuRsAddr);
      const UInt firstCtuRsAddrOfTile = pcPic->getPicSym()->getTComTile(pcPic->getPicSym()->getTileIdxMap(ctuRsAddr))->getFirstCtuRsAddr();
      const UInt tileXPosInCtus = firstCtuRsAddrOfTile % frameWidthInCtus;
      const UInt ctuXPosInCtus = ctuRsAddr % frameWidthInCtus;
      if (ctuRsAddr == firstCtuRsAddrOfTile)
      {
          m_pppcRDSbacCoder[0][CI_CURR_BEST]->resetEntropy(pcSlice);
      }
      else if (ctuXPosInCtus == tileXPosInCtus && m_pcCfg->getEntropyCodingSyncEnabledFlag())
      {
          m_pppcRDSbacCoder[0][CI_CURR_BEST]->resetEntropy(pcSlice);
          TComDataCU* pCtuUp = pCtu->getCtuAbove();
          if (pCtuUp && ((ctuRsAddr % frameWidthInCtus + 1) < frameWidthInCtus))
          {
              TComDataCU* pCtuTR = pcPic->getCtu(ctuRsAddr - frameWidthInCtus + 1);
              if (pCtu->CUIsFromSameSliceAndTile(pCtuTR))
              {
                  m_pppcRDSbacCoder[0][CI_CURR_BEST]->loadContexts(&m_entropyCodingSyncContextState);
              }
          }
      }
      m_pcEntropyCoder->setEntropyCoder(m_pcRDGoOnSbacCoder);
      m_pcEntropyCoder->setBitstream(&tempBitCounter);
      tempBitCounter.resetBits();
      m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[0][CI_CURR_BEST]);
      ((TEncBinCABAC*)m_pcRDGoOnSbacCoder->getEncBinIf())->setBinCountingEnableFlag(true);
      Double oldLambda = m_pcRdCost->getLambda();
      if (m_pcCfg->getUseRateCtrl())
      {
          Int estQP = pcSlice->getSliceQp();
          Double estLambda = -1.0;
          Double bpp = -1.0;
          if ((pcPic->getSlice(0)->getSliceType() == I_SLICE && m_pcCfg->getForceIntraQP()) || !m_pcCfg->getLCULevelRC())
          {
              estQP = pcSlice->getSliceQp();
          }
          else
          {
              bpp = m_pcRateCtrl->getRCPic()->getLCUTargetBpp(pcSlice->getSliceType());
              if (pcPic->getSlice(0)->getSliceType() == I_SLICE)
              {
                  estLambda = m_pcRateCtrl->getRCPic()->getLCUEstLambdaAndQP(bpp, pcSlice->getSliceQp(), &estQP);
              }
              else
              {
                  estLambda = m_pcRateCtrl->getRCPic()->getLCUEstLambda(bpp);
                  estQP = m_pcRateCtrl->getRCPic()->getLCUEstQP(estLambda, pcSlice->getSliceQp());
              }
              estQP = Clip3(-pcSlice->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, estQP);
              m_pcRdCost->setLambda(estLambda, pcSlice->getSPS()->getBitDepths());
              
#if RDOQ_CHROMA_LAMBDA
              const Double chromaLambda = estLambda/m_pcRdCost->getChromaWeight();
              const Double lambdaArray[MAX_NUM_COMPONENT] = {estLambda, chromaLambda, chromaLambda};
              m_pcTrQuant->setLambdas(lambdaArray);
#else
              m_pcTrQuant->setLambda(estLambda);
#endif
          }
          m_pcRateCtrl->setRCQP(estQP);
#if ADAPTIVE_QP_SELECTION
          pCtu->getSlice()->setSliceQpBase(estQP);
#endif
      }

      /*
      The decisions for compressing the current CTU, such
      as the partitioning of the current CTU or the intra
      prediction mode of each PU in the current CU, are made.
      */
      /*
      The method `TEncCU::compressCtu` is called by the method
      `TEncSlice::compressSlice` exclusively.
      */
      m_pcCuEncoder->compressCtu(pCtu);
      m_pcEntropyCoder->setEntropyCoder(m_pppcRDSbacCoder[0][CI_CURR_BEST]);
      m_pcEntropyCoder->setBitstream(&tempBitCounter);
      pRDSbacCoder->setBinCountingEnableFlag(true);
      m_pppcRDSbacCoder[0][CI_CURR_BEST]->resetBits();
      pRDSbacCoder->setBinsCoded(0);
      
      /*
      The method `TEncCU::encodeCtu` is called by the method
      `TEncSlice::compressSlice` and the method `TEncSlice::encodeSlice`
      exclusively.
      */
      m_pcCuEncoder->encodeCtu(pCtu, false);
      
      pRDSbacCoder->setBinCountingEnableFlag(false);
      const Int numberOfWrittenBits = m_pcEntropyCoder->getNumberOfWrittenBits();
      const UInt validEndOfSliceCtuTsAddr = ctuTsAddr + (ctuTsAddr == startCtuTsAddr ? 1 : 0);
      if (pcSlice->getSliceMode() == FIXED_NUMBER_OF_BYTES && pcSlice->getSliceBits() + numberOfWrittenBits > (pcSlice->getSliceArgument() << 3))
      {
          pcSlice->setSliceSegmentCurEndCtuTsAddr(validEndOfSliceCtuTsAddr);
          pcSlice->setSliceCurEndCtuTsAddr(validEndOfSliceCtuTsAddr);
          boundingCtuTsAddr=validEndOfSliceCtuTsAddr;
      }
      else if ((!bCompressEntireSlice) && pcSlice->getSliceSegmentMode() == FIXED_NUMBER_OF_BYTES && pcSlice->getSliceSegmentBits() + numberOfWrittenBits > (pcSlice->getSliceSegmentArgument() << 3))
      {
          pcSlice->setSliceSegmentCurEndCtuTsAddr(validEndOfSliceCtuTsAddr);
          boundingCtuTsAddr = validEndOfSliceCtuTsAddr;
      }
      if (boundingCtuTsAddr <= ctuTsAddr)
      {
          break;
      }
      pcSlice->setSliceBits((UInt)(pcSlice->getSliceBits() + numberOfWrittenBits));
      pcSlice->setSliceSegmentBits(pcSlice->getSliceSegmentBits() + numberOfWrittenBits);
      if (ctuXPosInCtus == tileXPosInCtus + 1 && m_pcCfg->getEntropyCodingSyncEnabledFlag())
      {
          m_entropyCodingSyncContextState.loadContexts(m_pppcRDSbacCoder[0][CI_CURR_BEST]);
      }
      if (m_pcCfg->getUseRateCtrl())
      {
          Int actualQP = g_RCInvalidQPValue;
          Double actualLambda = m_pcRdCost->getLambda();
          Int actualBits = pCtu->getTotalBits();
          Int numberOfEffectivePixels = 0;
          for (Int idx = 0; idx < pcPic->getNumPartitionsInCtu(); idx++)
          {
              if (pCtu->getPredictionMode(idx) != NUMBER_OF_PREDICTION_MODES && (!pCtu->isSkipped(idx)))
              {
                  numberOfEffectivePixels = numberOfEffectivePixels + 16;
                  break;
              }
          }
          if (numberOfEffectivePixels == 0)
          {
              actualQP = g_RCInvalidQPValue;
          }
          else
          {
              actualQP = pCtu->getQP(0);
          }
          m_pcRdCost->setLambda(oldLambda, pcSlice->getSPS()->getBitDepths());
          m_pcRateCtrl->getRCPic()->updateAfterCTU(m_pcRateCtrl->getRCPic()->getLCUCoded(),
                                                   actualBits,
                                                   actualQP,
                                                   actualLambda,
                                                   pCtu->getSlice()->getSliceType() == I_SLICE ? 0 : m_pcCfg->getLCULevelRC());
      }
      m_uiPicTotalBits += pCtu->getTotalBits();
      m_dPicRdCost += pCtu->getTotalCost();
      m_uiPicDist += pCtu->getTotalDistortion();
  }
  if (pcSlice->getPPS()->getDependentSliceSegmentsEnabledFlag())
  {
      m_lastSliceSegmentEndContextState.loadContexts(m_pppcRDSbacCoder[0][CI_CURR_BEST]);
  }
  m_pppcRDSbacCoder[0][CI_CURR_BEST]->setBitstream(NULL);
  m_pcRDGoOnSbacCoder->setBitstream(NULL);
}

Void TEncSlice::encodeSlice(TComPic* pcPic, TComOutputBitstream* pcSubstreams, UInt &numBinsCoded)
{
  TComSlice* const pcSlice = pcPic->getSlice(getSliceIdx());
  const UInt startCtuTsAddr = pcSlice->getSliceSegmentCurStartCtuTsAddr();
  const UInt boundingCtuTsAddr = pcSlice->getSliceSegmentCurEndCtuTsAddr();
  const UInt frameWidthInCtus = pcPic->getPicSym()->getFrameWidthInCtus();
  const Bool depSliceSegmentsEnabled = pcSlice->getPPS()->getDependentSliceSegmentsEnabledFlag();
  const Bool wavefrontsEnabled = pcSlice->getPPS()->getEntropyCodingSyncEnabledFlag();
  m_pcSbacCoder->init((TEncBinIf*)m_pcBinCABAC);
  m_pcEntropyCoder->setEntropyCoder(m_pcSbacCoder);
  m_pcEntropyCoder->resetEntropy(pcSlice);
  numBinsCoded = 0;
  m_pcBinCABAC->setBinCountingEnableFlag(true);
  m_pcBinCABAC->setBinsCoded(0);
#if ENC_DEC_TRACE
  g_bJustDoIt = g_bEncDecTraceEnable;
#endif
  DTRACE_CABAC_VL(g_nSymbolCounter++);
  DTRACE_CABAC_T("\tPOC: ");
  DTRACE_CABAC_V(pcPic->getPOC());
  DTRACE_CABAC_T("\n");
#if ENC_DEC_TRACE
  g_bJustDoIt = g_bEncDecTraceDisable;
#endif
  if (depSliceSegmentsEnabled)
  {
      const UInt ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTsAddr);
      const UInt currentTileIdx=pcPic->getPicSym()->getTileIdxMap(ctuRsAddr);
      const TComTile *pCurrentTile=pcPic->getPicSym()->getTComTile(currentTileIdx);
      const UInt firstCtuRsAddrOfTile = pCurrentTile->getFirstCtuRsAddr();
      if (pcSlice->getDependentSliceSegmentFlag() && ctuRsAddr != firstCtuRsAddrOfTile)
      {
          if (pCurrentTile->getTileWidthInCtus() >= 2 || !wavefrontsEnabled)
          {
              m_pcSbacCoder->loadContexts(&m_lastSliceSegmentEndContextState);
          }
      }
  }
  
  for (UInt ctuTsAddr = startCtuTsAddr; ctuTsAddr < boundingCtuTsAddr; ++ctuTsAddr)
  {
      const UInt ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(ctuTsAddr);
      const TComTile &currentTile = *(pcPic->getPicSym()->getTComTile(pcPic->getPicSym()->getTileIdxMap(ctuRsAddr)));
      const UInt firstCtuRsAddrOfTile = currentTile.getFirstCtuRsAddr();
      const UInt tileXPosInCtus = firstCtuRsAddrOfTile % frameWidthInCtus;
      const UInt tileYPosInCtus = firstCtuRsAddrOfTile/frameWidthInCtus;
      const UInt ctuXPosInCtus = ctuRsAddr % frameWidthInCtus;
      const UInt ctuYPosInCtus = ctuRsAddr/frameWidthInCtus;
      const UInt uiSubStrm = pcPic->getSubstreamForCtuAddr(ctuRsAddr, true, pcSlice);
      TComDataCU* pCtu = pcPic->getCtu(ctuRsAddr);
      m_pcEntropyCoder->setBitstream(&pcSubstreams[uiSubStrm]);
      if (ctuRsAddr == firstCtuRsAddrOfTile)
      {
          if (ctuTsAddr != startCtuTsAddr)
          {
              m_pcEntropyCoder->resetEntropy(pcSlice);
          }
      }
      else if (ctuXPosInCtus == tileXPosInCtus && wavefrontsEnabled)
      {
          if (ctuTsAddr != startCtuTsAddr)
          {
              m_pcEntropyCoder->resetEntropy(pcSlice);
          }
          TComDataCU *pCtuUp = pCtu->getCtuAbove();
          if (pCtuUp && ((ctuRsAddr % frameWidthInCtus + 1) < frameWidthInCtus))
          {
              TComDataCU *pCtuTR = pcPic->getCtu(ctuRsAddr - frameWidthInCtus + 1);
              if (pCtu->CUIsFromSameSliceAndTile(pCtuTR))
              {
                  m_pcSbacCoder->loadContexts(&m_entropyCodingSyncContextState);
              }
          }
      }
      if (pcSlice->getSPS()->getUseSAO())
      {
          Bool bIsSAOSliceEnabled = false;
          Bool sliceEnabled[MAX_NUM_COMPONENT];
          for (Int comp = 0; comp < MAX_NUM_COMPONENT; comp++)
          {
              ComponentID compId = ComponentID(comp);
              sliceEnabled[compId] = pcSlice->getSaoEnabledFlag(toChannelType(compId)) && (comp < pcPic->getNumberValidComponents());
              if (sliceEnabled[compId])
              {
                  bIsSAOSliceEnabled=true;
              }
          }
          if (bIsSAOSliceEnabled)
          {
              SAOBlkParam& saoblkParam = (pcPic->getPicSym()->getSAOBlkParam())[ctuRsAddr];
              Bool leftMergeAvail = false;
              Bool aboveMergeAvail= false;
              Int rx = (ctuRsAddr % frameWidthInCtus);
              if (rx > 0)
              {
                  leftMergeAvail = pcPic->getSAOMergeAvailability(ctuRsAddr, ctuRsAddr - 1);
              }
              Int ry = (ctuRsAddr/frameWidthInCtus);
              if (ry > 0)
              {
                  aboveMergeAvail = pcPic->getSAOMergeAvailability(ctuRsAddr, ctuRsAddr - frameWidthInCtus);
              }
              m_pcEntropyCoder->encodeSAOBlkParam(saoblkParam, pcPic->getPicSym()->getSPS().getBitDepths(), sliceEnabled, leftMergeAvail, aboveMergeAvail);
          }
      }
    
#if ENC_DEC_TRACE
      g_bJustDoIt = g_bEncDecTraceEnable;
#endif
      
      /*
      The method `TEncCU::encodeCtu` is called by the method
      `TEncSlice::compressSlice` and the method `TEncSlice::encodeSlice`
      exclusively.
      */
      m_pcCuEncoder->encodeCtu(pCtu, true);
#if ENC_DEC_TRACE
      g_bJustDoIt = g_bEncDecTraceDisable;
#endif
      if (ctuXPosInCtus == tileXPosInCtus + 1 && wavefrontsEnabled)
      {
          m_entropyCodingSyncContextState.loadContexts( m_pcSbacCoder );
      }
      if (ctuTsAddr + 1 == boundingCtuTsAddr || (ctuXPosInCtus + 1 == tileXPosInCtus + currentTile.getTileWidthInCtus() && (ctuYPosInCtus + 1 == tileYPosInCtus + currentTile.getTileHeightInCtus() || wavefrontsEnabled)))
      {
          m_pcEntropyCoder->encodeTerminatingBit(1);
          m_pcEntropyCoder->encodeSliceFinish();
          pcSubstreams[uiSubStrm].writeByteAlignment();
          if (ctuTsAddr + 1 != boundingCtuTsAddr)
          {
              pcSlice->addSubstreamSize((pcSubstreams[uiSubStrm].getNumberOfWrittenBits() >> 3) + pcSubstreams[uiSubStrm].countStartCodeEmulations());
          }
      }
  }
  
  if (m_ptr_map_luminance || m_ptr_map_chrominance)
  {
      int error_code(0);
      const unsigned int size_arrays_thresholds(4);
      unsigned char ptr_array_thresholds[size_arrays_thresholds] = {1, 17, 18, 34};
      
      /*
      A PB with either the DC mode or the planar mode is colored
      in red. A PB with the mode of index 18 is colored in sky blue.
      A PB with the mode of index 35, i.e. the prediction neural
      networks mode, is colored in blue. A PB with any other mode
      is colored in orange.
      */
      unsigned int ptr_array_colors[size_arrays_thresholds + 1][3] = {
          {255, 0, 0},
          {255, 165, 0},
          {135, 206, 250},
          {255, 165, 0},
          {0, 0, 255}
      };
      
      /*
      `m_ptr_map_luminance` is NULL if the path to the saved thresholded map
      of intra prediction modes for the luminance channel is not provided.
      */
      if (m_ptr_map_luminance)
      {
          error_code = visualize_thresholded_channel<unsigned char>(m_ptr_map_luminance,
                                                                    pcSlice->getSPS()->getPicHeightInLumaSamples(),
                                                                    pcSlice->getSPS()->getPicWidthInLumaSamples(),
                                                                    ptr_array_thresholds,
                                                                    size_arrays_thresholds,
                                                                    ptr_array_colors,
                                                                    255,
                                                                    m_pcCfg->getPathToThresholdedMapModesLuminance());
          assert(error_code >= 0);
      }
      
      /*
      `m_ptr_map_chrominance` is NULL if the path to the saved thresholded map
      of intra prediction modes for the two chrominance channels is not provided.
      */
      if (m_ptr_map_chrominance)
      {
          const std::string pathToBinaryMapDirectMode(m_pcCfg->getPathToBinaryMapDirectMode());
          
          /*
          The binary map for the direct mode is created before
          replacing each direct mode in the map of intra prediction
          modes for the two chrominance channels.
          */
          if (!pathToBinaryMapDirectMode.empty())
          {
              const unsigned int size_arrays_threshold_direct(1);
              unsigned char ptr_array_threshold_direct[size_arrays_threshold_direct] = {DM_CHROMA_IDX - 1};
              unsigned int ptr_array_colors_direct[size_arrays_threshold_direct + 1][3] = {
                  {200, 200, 200},
                  {0, 0, 0}
              };
               error_code = visualize_thresholded_channel<unsigned char>(m_ptr_map_chrominance,
                                                                         pcSlice->getSPS()->getPicHeightInLumaSamples(),
                                                                         pcSlice->getSPS()->getPicWidthInLumaSamples(),
                                                                         ptr_array_threshold_direct,
                                                                         size_arrays_threshold_direct,
                                                                         ptr_array_colors_direct,
                                                                         255,
                                                                         m_pcCfg->getPathToBinaryMapDirectMode());
              assert(error_code >= 0);
          }
          
          /*
          The precedence of ...->... (member access by pointer) is
          higher than the precedence of *... (dereference).
          */
          error_code = replace_in_array_by_value<unsigned char>(m_ptr_map_chrominance,
                                                                m_ptr_map_luminance,
                                                                pcSlice->getSPS()->getPicHeightInLumaSamples()*pcSlice->getSPS()->getPicWidthInLumaSamples(),
                                                                DM_CHROMA_IDX);
          assert(error_code >= 0);
          error_code = visualize_thresholded_channel<unsigned char>(m_ptr_map_chrominance,
                                                                    pcSlice->getSPS()->getPicHeightInLumaSamples(),
                                                                    pcSlice->getSPS()->getPicWidthInLumaSamples(),
                                                                    ptr_array_thresholds,
                                                                    size_arrays_thresholds,
                                                                    ptr_array_colors,
                                                                    255,
                                                                    m_pcCfg->getPathToThresholdedMapModesChrominance());
          assert(error_code >= 0);
      }
  }
  
  if (depSliceSegmentsEnabled)
  {
      m_lastSliceSegmentEndContextState.loadContexts(m_pcSbacCoder);
  }
#if ADAPTIVE_QP_SELECTION
  if (m_pcCfg->getUseAdaptQpSelect())
  {
      m_pcTrQuant->storeSliceQpNext(pcSlice);
  }
#endif
  if (pcSlice->getPPS()->getCabacInitPresentFlag() && !pcSlice->getPPS()->getDependentSliceSegmentsEnabledFlag())
  {
      m_encCABACTableIdx = m_pcEntropyCoder->determineCabacInitIdx(pcSlice);
  }
  else
  {
      m_encCABACTableIdx = pcSlice->getSliceType();
  }
  numBinsCoded = m_pcBinCABAC->getBinsCoded();
}

Void TEncSlice::calculateBoundingCtuTsAddrForSlice(UInt &startCtuTSAddrSlice, UInt &boundingCtuTSAddrSlice, Bool &haveReachedTileBoundary,
                                                   TComPic* pcPic, const Int sliceMode, const Int sliceArgument)
{
  TComSlice* pcSlice = pcPic->getSlice(getSliceIdx());
  const UInt numberOfCtusInFrame = pcPic->getNumberOfCtusInFrame();
  const TComPPS &pps=*(pcSlice->getPPS());
  boundingCtuTSAddrSlice=0;
  haveReachedTileBoundary=false;

  switch (sliceMode)
  {
    case FIXED_NUMBER_OF_CTU:
      {
        UInt ctuAddrIncrement    = sliceArgument;
        boundingCtuTSAddrSlice  = ((startCtuTSAddrSlice + ctuAddrIncrement) < numberOfCtusInFrame) ? (startCtuTSAddrSlice + ctuAddrIncrement) : numberOfCtusInFrame;
      }
      break;
    case FIXED_NUMBER_OF_BYTES:
      boundingCtuTSAddrSlice  = numberOfCtusInFrame; // This will be adjusted later if required.
      break;
    case FIXED_NUMBER_OF_TILES:
      {
        const UInt tileIdx        = pcPic->getPicSym()->getTileIdxMap( pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTSAddrSlice) );
        const UInt tileTotalCount = (pcPic->getPicSym()->getNumTileColumnsMinus1()+1) * (pcPic->getPicSym()->getNumTileRowsMinus1()+1);
        UInt ctuAddrIncrement   = 0;

        for(UInt tileIdxIncrement = 0; tileIdxIncrement < sliceArgument; tileIdxIncrement++)
        {
          if((tileIdx + tileIdxIncrement) < tileTotalCount)
          {
            UInt tileWidthInCtus   = pcPic->getPicSym()->getTComTile(tileIdx + tileIdxIncrement)->getTileWidthInCtus();
            UInt tileHeightInCtus  = pcPic->getPicSym()->getTComTile(tileIdx + tileIdxIncrement)->getTileHeightInCtus();
            ctuAddrIncrement    += (tileWidthInCtus * tileHeightInCtus);
          }
        }

        boundingCtuTSAddrSlice  = ((startCtuTSAddrSlice + ctuAddrIncrement) < numberOfCtusInFrame) ? (startCtuTSAddrSlice + ctuAddrIncrement) : numberOfCtusInFrame;
      }
      break;
    default:
      boundingCtuTSAddrSlice    = numberOfCtusInFrame;
      break;
  }

  // Adjust for tiles and wavefronts.
  const Bool wavefrontsAreEnabled = pps.getEntropyCodingSyncEnabledFlag();

  if ((sliceMode == FIXED_NUMBER_OF_CTU || sliceMode == FIXED_NUMBER_OF_BYTES) &&
      (pps.getNumTileRowsMinus1() > 0 || pps.getNumTileColumnsMinus1() > 0))
  {
    const UInt ctuRSAddr                  = pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTSAddrSlice);
    const UInt startTileIdx               = pcPic->getPicSym()->getTileIdxMap(ctuRSAddr);

    const TComTile *pStartingTile         = pcPic->getPicSym()->getTComTile(startTileIdx);
    const UInt tileStartTsAddr            = pcPic->getPicSym()->getCtuRsToTsAddrMap(pStartingTile->getFirstCtuRsAddr());
    const UInt tileStartWidth             = pStartingTile->getTileWidthInCtus();
    const UInt tileStartHeight            = pStartingTile->getTileHeightInCtus();
    const UInt tileLastTsAddr_excl        = tileStartTsAddr + tileStartWidth*tileStartHeight;
    const UInt tileBoundingCtuTsAddrSlice = tileLastTsAddr_excl;

    const UInt ctuColumnOfStartingTile    = ((startCtuTSAddrSlice-tileStartTsAddr)%tileStartWidth);
    if (wavefrontsAreEnabled && ctuColumnOfStartingTile!=0)
    {
      // WPP: if a slice does not start at the beginning of a CTB row, it must end within the same CTB row
      const UInt numberOfCTUsToEndOfRow            = tileStartWidth - ctuColumnOfStartingTile;
      const UInt wavefrontTileBoundingCtuAddrSlice = startCtuTSAddrSlice + numberOfCTUsToEndOfRow;
      if (wavefrontTileBoundingCtuAddrSlice < boundingCtuTSAddrSlice)
      {
        boundingCtuTSAddrSlice = wavefrontTileBoundingCtuAddrSlice;
      }
    }

    if (tileBoundingCtuTsAddrSlice < boundingCtuTSAddrSlice)
    {
      boundingCtuTSAddrSlice = tileBoundingCtuTsAddrSlice;
      haveReachedTileBoundary = true;
    }
  }
  else if ((sliceMode == FIXED_NUMBER_OF_CTU || sliceMode == FIXED_NUMBER_OF_BYTES) && wavefrontsAreEnabled && ((startCtuTSAddrSlice % pcPic->getFrameWidthInCtus()) != 0))
  {
    // Adjust for wavefronts (no tiles).
    // WPP: if a slice does not start at the beginning of a CTB row, it must end within the same CTB row
    boundingCtuTSAddrSlice = min(boundingCtuTSAddrSlice, startCtuTSAddrSlice - (startCtuTSAddrSlice % pcPic->getFrameWidthInCtus()) + (pcPic->getFrameWidthInCtus()));
  }
}

/** Determines the starting and bounding CTU address of current slice / dependent slice
 * \param [out] startCtuTsAddr
 * \param [out] boundingCtuTsAddr
 * \param [in]  pcPic

 * Updates startCtuTsAddr, boundingCtuTsAddr with appropriate CTU address
 */
Void TEncSlice::xDetermineStartAndBoundingCtuTsAddr  ( UInt& startCtuTsAddr, UInt& boundingCtuTsAddr, TComPic* pcPic )
{
  TComSlice* pcSlice                 = pcPic->getSlice(getSliceIdx());

  // Non-dependent slice
  UInt startCtuTsAddrSlice           = pcSlice->getSliceCurStartCtuTsAddr();
  Bool haveReachedTileBoundarySlice  = false;
  UInt boundingCtuTsAddrSlice;
  calculateBoundingCtuTsAddrForSlice(startCtuTsAddrSlice, boundingCtuTsAddrSlice, haveReachedTileBoundarySlice, pcPic,
                                     m_pcCfg->getSliceMode(), m_pcCfg->getSliceArgument());
  pcSlice->setSliceCurEndCtuTsAddr(   boundingCtuTsAddrSlice );
  pcSlice->setSliceCurStartCtuTsAddr( startCtuTsAddrSlice    );

  // Dependent slice
  UInt startCtuTsAddrSliceSegment          = pcSlice->getSliceSegmentCurStartCtuTsAddr();
  Bool haveReachedTileBoundarySliceSegment = false;
  UInt boundingCtuTsAddrSliceSegment;
  calculateBoundingCtuTsAddrForSlice(startCtuTsAddrSliceSegment, boundingCtuTsAddrSliceSegment, haveReachedTileBoundarySliceSegment, pcPic,
                                     m_pcCfg->getSliceSegmentMode(), m_pcCfg->getSliceSegmentArgument());
  if (boundingCtuTsAddrSliceSegment>boundingCtuTsAddrSlice)
  {
    boundingCtuTsAddrSliceSegment = boundingCtuTsAddrSlice;
  }
  pcSlice->setSliceSegmentCurEndCtuTsAddr( boundingCtuTsAddrSliceSegment );
  pcSlice->setSliceSegmentCurStartCtuTsAddr(startCtuTsAddrSliceSegment);

  // Make a joint decision based on reconstruction and dependent slice bounds
  startCtuTsAddr    = max(startCtuTsAddrSlice   , startCtuTsAddrSliceSegment   );
  boundingCtuTsAddr = boundingCtuTsAddrSliceSegment;
}

Double TEncSlice::xGetQPValueAccordingToLambda ( Double lambda )
{
  return 4.2005*log(lambda) + 13.7122;
}
