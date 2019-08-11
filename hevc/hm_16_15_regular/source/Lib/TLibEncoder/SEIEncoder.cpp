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

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/SEI.h"
#include "TEncGOP.h"
#include "TEncTop.h"

//! \ingroup TLibEncoder
//! \{

Void SEIEncoder::initSEIActiveParameterSets (SEIActiveParameterSets *seiActiveParameterSets, const TComVPS *vps, const TComSPS *sps)
{
  assert (m_isInitialized);
  assert (seiActiveParameterSets!=NULL);
  assert (vps!=NULL);
  assert (sps!=NULL);

  seiActiveParameterSets->activeVPSId = vps->getVPSId(); 
  seiActiveParameterSets->m_selfContainedCvsFlag = false;
  seiActiveParameterSets->m_noParameterSetUpdateFlag = false;
  seiActiveParameterSets->numSpsIdsMinus1 = 0;
  seiActiveParameterSets->activeSeqParameterSetId.resize(seiActiveParameterSets->numSpsIdsMinus1 + 1);
  seiActiveParameterSets->activeSeqParameterSetId[0] = sps->getSPSId();
}

Void SEIEncoder::initSEIFramePacking(SEIFramePacking *seiFramePacking, Int currPicNum)
{
  assert (m_isInitialized);
  assert (seiFramePacking!=NULL);

  seiFramePacking->m_arrangementId = m_pcCfg->getFramePackingArrangementSEIId();
  seiFramePacking->m_arrangementCancelFlag = 0;
  seiFramePacking->m_arrangementType = m_pcCfg->getFramePackingArrangementSEIType();
  assert((seiFramePacking->m_arrangementType > 2) && (seiFramePacking->m_arrangementType < 6) );
  seiFramePacking->m_quincunxSamplingFlag = m_pcCfg->getFramePackingArrangementSEIQuincunx();
  seiFramePacking->m_contentInterpretationType = m_pcCfg->getFramePackingArrangementSEIInterpretation();
  seiFramePacking->m_spatialFlippingFlag = 0;
  seiFramePacking->m_frame0FlippedFlag = 0;
  seiFramePacking->m_fieldViewsFlag = (seiFramePacking->m_arrangementType == 2);
  seiFramePacking->m_currentFrameIsFrame0Flag = ((seiFramePacking->m_arrangementType == 5) && (currPicNum&1) );
  seiFramePacking->m_frame0SelfContainedFlag = 0;
  seiFramePacking->m_frame1SelfContainedFlag = 0;
  seiFramePacking->m_frame0GridPositionX = 0;
  seiFramePacking->m_frame0GridPositionY = 0;
  seiFramePacking->m_frame1GridPositionX = 0;
  seiFramePacking->m_frame1GridPositionY = 0;
  seiFramePacking->m_arrangementReservedByte = 0;
  seiFramePacking->m_arrangementPersistenceFlag = true;
  seiFramePacking->m_upsampledAspectRatio = 0;
}

Void SEIEncoder::initSEISegmentedRectFramePacking(SEISegmentedRectFramePacking *seiSegmentedRectFramePacking)
{
  assert (m_isInitialized);
  assert (seiSegmentedRectFramePacking!=NULL);

  seiSegmentedRectFramePacking->m_arrangementCancelFlag = m_pcCfg->getSegmentedRectFramePackingArrangementSEICancel();
  seiSegmentedRectFramePacking->m_contentInterpretationType = m_pcCfg->getSegmentedRectFramePackingArrangementSEIType();
  seiSegmentedRectFramePacking->m_arrangementPersistenceFlag = m_pcCfg->getSegmentedRectFramePackingArrangementSEIPersistence();
}

Void SEIEncoder::initSEIDisplayOrientation(SEIDisplayOrientation* seiDisplayOrientation)
{
  assert (m_isInitialized);
  assert (seiDisplayOrientation!=NULL);

  seiDisplayOrientation->cancelFlag = false;
  seiDisplayOrientation->horFlip = false;
  seiDisplayOrientation->verFlip = false;
  seiDisplayOrientation->anticlockwiseRotation = m_pcCfg->getDisplayOrientationSEIAngle();
}

Void SEIEncoder::initSEIToneMappingInfo(SEIToneMappingInfo *seiToneMappingInfo)
{
  assert (m_isInitialized);
  assert (seiToneMappingInfo!=NULL);

  seiToneMappingInfo->m_toneMapId = m_pcCfg->getTMISEIToneMapId();
  seiToneMappingInfo->m_toneMapCancelFlag = m_pcCfg->getTMISEIToneMapCancelFlag();
  seiToneMappingInfo->m_toneMapPersistenceFlag = m_pcCfg->getTMISEIToneMapPersistenceFlag();

  seiToneMappingInfo->m_codedDataBitDepth = m_pcCfg->getTMISEICodedDataBitDepth();
  assert(seiToneMappingInfo->m_codedDataBitDepth >= 8 && seiToneMappingInfo->m_codedDataBitDepth <= 14);
  seiToneMappingInfo->m_targetBitDepth = m_pcCfg->getTMISEITargetBitDepth();
  assert(seiToneMappingInfo->m_targetBitDepth >= 1 && seiToneMappingInfo->m_targetBitDepth <= 17);
  seiToneMappingInfo->m_modelId = m_pcCfg->getTMISEIModelID();
  assert(seiToneMappingInfo->m_modelId >=0 &&seiToneMappingInfo->m_modelId<=4);

  switch( seiToneMappingInfo->m_modelId)
  {
  case 0:
    {
      seiToneMappingInfo->m_minValue = m_pcCfg->getTMISEIMinValue();
      seiToneMappingInfo->m_maxValue = m_pcCfg->getTMISEIMaxValue();
      break;
    }
  case 1:
    {
      seiToneMappingInfo->m_sigmoidMidpoint = m_pcCfg->getTMISEISigmoidMidpoint();
      seiToneMappingInfo->m_sigmoidWidth = m_pcCfg->getTMISEISigmoidWidth();
      break;
    }
  case 2:
    {
      UInt num = 1u<<(seiToneMappingInfo->m_targetBitDepth);
      seiToneMappingInfo->m_startOfCodedInterval.resize(num);
      Int* ptmp = m_pcCfg->getTMISEIStartOfCodedInterva();
      if(ptmp)
      {
        for(Int i=0; i<num;i++)
        {
          seiToneMappingInfo->m_startOfCodedInterval[i] = ptmp[i];
        }
      }
      break;
    }
  case 3:
    {
      seiToneMappingInfo->m_numPivots = m_pcCfg->getTMISEINumPivots();
      seiToneMappingInfo->m_codedPivotValue.resize(seiToneMappingInfo->m_numPivots);
      seiToneMappingInfo->m_targetPivotValue.resize(seiToneMappingInfo->m_numPivots);
      Int* ptmpcoded = m_pcCfg->getTMISEICodedPivotValue();
      Int* ptmptarget = m_pcCfg->getTMISEITargetPivotValue();
      if(ptmpcoded&&ptmptarget)
      {
        for(Int i=0; i<(seiToneMappingInfo->m_numPivots);i++)
        {
          seiToneMappingInfo->m_codedPivotValue[i]=ptmpcoded[i];
          seiToneMappingInfo->m_targetPivotValue[i]=ptmptarget[i];
        }
      }
      break;
    }
  case 4:
    {
      seiToneMappingInfo->m_cameraIsoSpeedIdc = m_pcCfg->getTMISEICameraIsoSpeedIdc();
      seiToneMappingInfo->m_cameraIsoSpeedValue = m_pcCfg->getTMISEICameraIsoSpeedValue();
      assert( seiToneMappingInfo->m_cameraIsoSpeedValue !=0 );
      seiToneMappingInfo->m_exposureIndexIdc = m_pcCfg->getTMISEIExposurIndexIdc();
      seiToneMappingInfo->m_exposureIndexValue = m_pcCfg->getTMISEIExposurIndexValue();
      assert( seiToneMappingInfo->m_exposureIndexValue !=0 );
      seiToneMappingInfo->m_exposureCompensationValueSignFlag = m_pcCfg->getTMISEIExposureCompensationValueSignFlag();
      seiToneMappingInfo->m_exposureCompensationValueNumerator = m_pcCfg->getTMISEIExposureCompensationValueNumerator();
      seiToneMappingInfo->m_exposureCompensationValueDenomIdc = m_pcCfg->getTMISEIExposureCompensationValueDenomIdc();
      seiToneMappingInfo->m_refScreenLuminanceWhite = m_pcCfg->getTMISEIRefScreenLuminanceWhite();
      seiToneMappingInfo->m_extendedRangeWhiteLevel = m_pcCfg->getTMISEIExtendedRangeWhiteLevel();
      assert( seiToneMappingInfo->m_extendedRangeWhiteLevel >= 100 );
      seiToneMappingInfo->m_nominalBlackLevelLumaCodeValue = m_pcCfg->getTMISEINominalBlackLevelLumaCodeValue();
      seiToneMappingInfo->m_nominalWhiteLevelLumaCodeValue = m_pcCfg->getTMISEINominalWhiteLevelLumaCodeValue();
      assert( seiToneMappingInfo->m_nominalWhiteLevelLumaCodeValue > seiToneMappingInfo->m_nominalBlackLevelLumaCodeValue );
      seiToneMappingInfo->m_extendedWhiteLevelLumaCodeValue = m_pcCfg->getTMISEIExtendedWhiteLevelLumaCodeValue();
      assert( seiToneMappingInfo->m_extendedWhiteLevelLumaCodeValue >= seiToneMappingInfo->m_nominalWhiteLevelLumaCodeValue );
      break;
    }
  default:
    {
      assert(!"Undefined SEIToneMapModelId");
      break;
    }
  }
}

Void SEIEncoder::initSEISOPDescription(SEISOPDescription *sopDescriptionSEI, TComSlice *slice, Int picInGOP, Int lastIdr, Int currGOPSize)
{
  assert (m_isInitialized);
  assert (sopDescriptionSEI != NULL);
  assert (slice != NULL);

  Int sopCurrPOC = slice->getPOC();
  sopDescriptionSEI->m_sopSeqParameterSetId = slice->getSPS()->getSPSId();

  Int i = 0;
  Int prevEntryId = picInGOP;
  for (Int j = picInGOP; j < currGOPSize; j++)
  {
    Int deltaPOC = m_pcCfg->getGOPEntry(j).m_POC - m_pcCfg->getGOPEntry(prevEntryId).m_POC;
    if ((sopCurrPOC + deltaPOC) < m_pcCfg->getFramesToBeEncoded())
    {
      sopCurrPOC += deltaPOC;
      sopDescriptionSEI->m_sopDescVclNaluType[i] = m_pcEncGOP->getNalUnitType(sopCurrPOC, lastIdr, slice->getPic()->isField());
      sopDescriptionSEI->m_sopDescTemporalId[i] = m_pcCfg->getGOPEntry(j).m_temporalId;
      sopDescriptionSEI->m_sopDescStRpsIdx[i] = m_pcEncTop->getReferencePictureSetIdxForSOP(sopCurrPOC, j);
      sopDescriptionSEI->m_sopDescPocDelta[i] = deltaPOC;

      prevEntryId = j;
      i++;
    }
  }

  sopDescriptionSEI->m_numPicsInSopMinus1 = i - 1;
}

Void SEIEncoder::initSEIBufferingPeriod(SEIBufferingPeriod *bufferingPeriodSEI, TComSlice *slice)
{
  assert (m_isInitialized);
  assert (bufferingPeriodSEI != NULL);
  assert (slice != NULL);

  UInt uiInitialCpbRemovalDelay = (90000/2);                      // 0.5 sec
  bufferingPeriodSEI->m_initialCpbRemovalDelay      [0][0]     = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialCpbRemovalDelayOffset[0][0]     = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialCpbRemovalDelay      [0][1]     = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialCpbRemovalDelayOffset[0][1]     = uiInitialCpbRemovalDelay;

  Double dTmp = (Double)slice->getSPS()->getVuiParameters()->getTimingInfo()->getNumUnitsInTick() / (Double)slice->getSPS()->getVuiParameters()->getTimingInfo()->getTimeScale();

  UInt uiTmp = (UInt)( dTmp * 90000.0 );
  uiInitialCpbRemovalDelay -= uiTmp;
  uiInitialCpbRemovalDelay -= uiTmp / ( slice->getSPS()->getVuiParameters()->getHrdParameters()->getTickDivisorMinus2() + 2 );
  bufferingPeriodSEI->m_initialAltCpbRemovalDelay      [0][0]  = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialAltCpbRemovalDelayOffset[0][0]  = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialAltCpbRemovalDelay      [0][1]  = uiInitialCpbRemovalDelay;
  bufferingPeriodSEI->m_initialAltCpbRemovalDelayOffset[0][1]  = uiInitialCpbRemovalDelay;

  bufferingPeriodSEI->m_rapCpbParamsPresentFlag = 0;
  //for the concatenation, it can be set to one during splicing.
  bufferingPeriodSEI->m_concatenationFlag = 0;
  //since the temporal layer HRD is not ready, we assumed it is fixed
  bufferingPeriodSEI->m_auCpbRemovalDelayDelta = 1;
  bufferingPeriodSEI->m_cpbDelayOffset = 0;
  bufferingPeriodSEI->m_dpbDelayOffset = 0;
}

//! initialize scalable nesting SEI message.
//! Note: The SEI message structures input into this function will become part of the scalable nesting SEI and will be 
//!       automatically freed, when the nesting SEI is disposed.
Void SEIEncoder::initSEIScalableNesting(SEIScalableNesting *scalableNestingSEI, SEIMessages &nestedSEIs)
{
  assert (m_isInitialized);
  assert (scalableNestingSEI != NULL);

  scalableNestingSEI->m_bitStreamSubsetFlag           = 1;      // If the nested SEI messages are picture buffering SEI messages, picture timing SEI messages or sub-picture timing SEI messages, bitstream_subset_flag shall be equal to 1
  scalableNestingSEI->m_nestingOpFlag                 = 0;
  scalableNestingSEI->m_nestingNumOpsMinus1           = 0;      //nesting_num_ops_minus1
  scalableNestingSEI->m_allLayersFlag                 = 0;
  scalableNestingSEI->m_nestingNoOpMaxTemporalIdPlus1 = 6 + 1;  //nesting_no_op_max_temporal_id_plus1
  scalableNestingSEI->m_nestingNumLayersMinus1        = 1 - 1;  //nesting_num_layers_minus1
  scalableNestingSEI->m_nestingLayerId[0]             = 0;

  scalableNestingSEI->m_nestedSEIs.clear();
  for (SEIMessages::iterator it=nestedSEIs.begin(); it!=nestedSEIs.end(); it++)
  {
    scalableNestingSEI->m_nestedSEIs.push_back((*it));
  }
}

Void SEIEncoder::initSEIRecoveryPoint(SEIRecoveryPoint *recoveryPointSEI, TComSlice *slice)
{
  assert (m_isInitialized);
  assert (recoveryPointSEI != NULL);
  assert (slice != NULL);

  recoveryPointSEI->m_recoveryPocCnt    = 0;
  recoveryPointSEI->m_exactMatchingFlag = ( slice->getPOC() == 0 ) ? (true) : (false);
  recoveryPointSEI->m_brokenLinkFlag    = false;
}

//! calculate hashes for entire reconstructed picture
Void SEIEncoder::initDecodedPictureHashSEI(SEIDecodedPictureHash *decodedPictureHashSEI, TComPic *pcPic, std::string &rHashString, const BitDepths &bitDepths)
{
  assert (m_isInitialized);
  assert (decodedPictureHashSEI!=NULL);
  assert (pcPic!=NULL);

  decodedPictureHashSEI->method = m_pcCfg->getDecodedPictureHashSEIType();
  switch (m_pcCfg->getDecodedPictureHashSEIType())
  {
    case HASHTYPE_MD5:
      {
        UInt numChar=calcMD5(*pcPic->getPicYuvRec(), decodedPictureHashSEI->m_pictureHash, bitDepths);
        rHashString = hashToString(decodedPictureHashSEI->m_pictureHash, numChar);
      }
      break;
    case HASHTYPE_CRC:
      {
        UInt numChar=calcCRC(*pcPic->getPicYuvRec(), decodedPictureHashSEI->m_pictureHash, bitDepths);
        rHashString = hashToString(decodedPictureHashSEI->m_pictureHash, numChar);
      }
      break;
    case HASHTYPE_CHECKSUM:
    default:
      {
        UInt numChar=calcChecksum(*pcPic->getPicYuvRec(), decodedPictureHashSEI->m_pictureHash, bitDepths);
        rHashString = hashToString(decodedPictureHashSEI->m_pictureHash, numChar);
      }
      break;
  }
}

Void SEIEncoder::initTemporalLevel0IndexSEI(SEITemporalLevel0Index *temporalLevel0IndexSEI, TComSlice *slice)
{
  assert (m_isInitialized);
  assert (temporalLevel0IndexSEI!=NULL);
  assert (slice!=NULL);

  if (slice->getRapPicFlag())
  {
    m_tl0Idx = 0;
    m_rapIdx = (m_rapIdx + 1) & 0xFF;
  }
  else
  {
    m_tl0Idx = (m_tl0Idx + (slice->getTLayer() ? 0 : 1)) & 0xFF;
  }
  temporalLevel0IndexSEI->tl0Idx = m_tl0Idx;
  temporalLevel0IndexSEI->rapIdx = m_rapIdx;
}

Void SEIEncoder::initSEITempMotionConstrainedTileSets (SEITempMotionConstrainedTileSets *sei, const TComPPS *pps)
{
  assert (m_isInitialized);
  assert (sei!=NULL);
  assert (pps!=NULL);

  if(pps->getTilesEnabledFlag())
  {
    sei->m_mc_all_tiles_exact_sample_value_match_flag = false;
    sei->m_each_tile_one_tile_set_flag                = false;
    sei->m_limited_tile_set_display_flag              = false;
    sei->setNumberOfTileSets((pps->getNumTileColumnsMinus1() + 1) * (pps->getNumTileRowsMinus1() + 1));

    for(Int i=0; i < sei->getNumberOfTileSets(); i++)
    {
      sei->tileSetData(i).m_mcts_id = i;  //depends the application;
      sei->tileSetData(i).setNumberOfTileRects(1);

      for(Int j=0; j<sei->tileSetData(i).getNumberOfTileRects(); j++)
      {
        sei->tileSetData(i).topLeftTileIndex(j)     = i+j;
        sei->tileSetData(i).bottomRightTileIndex(j) = i+j;
      }

      sei->tileSetData(i).m_exact_sample_value_match_flag    = false;
      sei->tileSetData(i).m_mcts_tier_level_idc_present_flag = false;
    }
  }
  else
  {
    assert(!"Tile is not enabled");
  }
}

Void SEIEncoder::initSEIKneeFunctionInfo(SEIKneeFunctionInfo *seiKneeFunctionInfo)
{
  assert (m_isInitialized);
  assert (seiKneeFunctionInfo!=NULL);

  seiKneeFunctionInfo->m_kneeId = m_pcCfg->getKneeSEIId();
  seiKneeFunctionInfo->m_kneeCancelFlag = m_pcCfg->getKneeSEICancelFlag();
  if ( !seiKneeFunctionInfo->m_kneeCancelFlag )
  {
    seiKneeFunctionInfo->m_kneePersistenceFlag = m_pcCfg->getKneeSEIPersistenceFlag();
    seiKneeFunctionInfo->m_kneeInputDrange = m_pcCfg->getKneeSEIInputDrange();
    seiKneeFunctionInfo->m_kneeInputDispLuminance = m_pcCfg->getKneeSEIInputDispLuminance();
    seiKneeFunctionInfo->m_kneeOutputDrange = m_pcCfg->getKneeSEIOutputDrange();
    seiKneeFunctionInfo->m_kneeOutputDispLuminance = m_pcCfg->getKneeSEIOutputDispLuminance();

    seiKneeFunctionInfo->m_kneeNumKneePointsMinus1 = m_pcCfg->getKneeSEINumKneePointsMinus1();
    Int* piInputKneePoint  = m_pcCfg->getKneeSEIInputKneePoint();
    Int* piOutputKneePoint = m_pcCfg->getKneeSEIOutputKneePoint();
    if(piInputKneePoint&&piOutputKneePoint)
    {
      seiKneeFunctionInfo->m_kneeInputKneePoint.resize(seiKneeFunctionInfo->m_kneeNumKneePointsMinus1+1);
      seiKneeFunctionInfo->m_kneeOutputKneePoint.resize(seiKneeFunctionInfo->m_kneeNumKneePointsMinus1+1);
      for(Int i=0; i<=seiKneeFunctionInfo->m_kneeNumKneePointsMinus1; i++)
      {
        seiKneeFunctionInfo->m_kneeInputKneePoint[i] = piInputKneePoint[i];
        seiKneeFunctionInfo->m_kneeOutputKneePoint[i] = piOutputKneePoint[i];
      }
    }
  }
}

template <typename T>
static Void readTokenValue(T            &returnedValue, /// value returned
                           Bool         &failed,        /// used and updated
                           std::istream &is,            /// stream to read token from
                           const TChar  *pToken)        /// token string
{
  returnedValue=T();
  if (failed)
  {
    return;
  }

  Int c;
  // Ignore any whitespace
  while ((c=is.get())!=EOF && isspace(c));
  // test for comment mark
  while (c=='#')
  {
    // Ignore to the end of the line
    while ((c=is.get())!=EOF && (c!=10 && c!=13));
    // Ignore any white space at the start of the next line
    while ((c=is.get())!=EOF && isspace(c));
  }
  // test first character of token
  failed=(c!=pToken[0]);
  // test remaining characters of token
  Int pos;
  for(pos=1;!failed && pToken[pos]!=0 && is.get()==pToken[pos]; pos++);
  failed|=(pToken[pos]!=0);
  // Ignore any whitespace before the ':'
  while (!failed && (c=is.get())!=EOF && isspace(c));
  failed|=(c!=':');
  // Now read the value associated with the token:
  if (!failed)
  {
    is >> returnedValue;
    failed=!is.good();
    if (!failed)
    {
      c=is.get();
      failed=(c!=EOF && !isspace(c));
    }
  }
  if (failed)
  {
    std::cerr << "Unable to read token '" << pToken << "'\n";
  }
}

template <typename T>
static Void readTokenValueAndValidate(T            &returnedValue, /// value returned
                                      Bool         &failed,        /// used and updated
                                      std::istream &is,            /// stream to read token from
                                      const TChar  *pToken,        /// token string
                                      const T      &minInclusive,  /// minimum value allowed, inclusive
                                      const T      &maxInclusive)  /// maximum value allowed, inclusive
{
  readTokenValue(returnedValue, failed, is, pToken);
  if (!failed)
  {
    if (returnedValue<minInclusive || returnedValue>maxInclusive)
    {
      failed=true;
      std::cerr << "Value for token " << pToken << " must be in the range " << minInclusive << " to " << maxInclusive << " (inclusive); value read: " << returnedValue << std::endl;
    }
  }
}

// Bool version does not have maximum and minimum values.
static Void readTokenValueAndValidate(Bool         &returnedValue, /// value returned
                                      Bool         &failed,        /// used and updated
                                      std::istream &is,            /// stream to read token from
                                      const TChar  *pToken)        /// token string
{
  readTokenValue(returnedValue, failed, is, pToken);
}

Bool SEIEncoder::initSEIColourRemappingInfo(SEIColourRemappingInfo* seiColourRemappingInfo, Int currPOC) // returns true on success, false on failure.
{
  assert (m_isInitialized);
  assert (seiColourRemappingInfo!=NULL);

  // reading external Colour Remapping Information SEI message parameters from file
  if( !m_pcCfg->getColourRemapInfoSEIFileRoot().empty())
  {
    Bool failed=false;

    // building the CRI file name with poc num in prefix "_poc.txt"
    std::string colourRemapSEIFileWithPoc(m_pcCfg->getColourRemapInfoSEIFileRoot());
    {
      std::stringstream suffix;
      suffix << "_" << currPOC << ".txt";
      colourRemapSEIFileWithPoc+=suffix.str();
    }

    std::ifstream fic(colourRemapSEIFileWithPoc.c_str());
    if (!fic.good() || !fic.is_open())
    {
      std::cerr <<  "No Colour Remapping Information SEI parameters file " << colourRemapSEIFileWithPoc << " for POC " << currPOC << std::endl;
      return false;
    }

    // TODO: identify and remove duplication with decoder parsing through abstraction.

    readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapId,         failed, fic, "colour_remap_id",        UInt(0), UInt(0x7fffffff) );
    readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapCancelFlag, failed, fic, "colour_remap_cancel_flag" );
    if( !seiColourRemappingInfo->m_colourRemapCancelFlag )
    {
      readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapPersistenceFlag,            failed, fic, "colour_remap_persistence_flag" );
      readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapVideoSignalInfoPresentFlag, failed, fic, "colour_remap_video_signal_info_present_flag");
      if( seiColourRemappingInfo->m_colourRemapVideoSignalInfoPresentFlag )
      {
        readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapFullRangeFlag,      failed, fic, "colour_remap_full_range_flag" );
        readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapPrimaries,          failed, fic, "colour_remap_primaries",           Int(0), Int(255) );
        readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapTransferFunction,   failed, fic, "colour_remap_transfer_function",   Int(0), Int(255) );
        readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapMatrixCoefficients, failed, fic, "colour_remap_matrix_coefficients", Int(0), Int(255) );
      }
      readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapInputBitDepth, failed, fic, "colour_remap_input_bit_depth",            Int(8), Int(16) );
      readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapBitDepth,      failed, fic, "colour_remap_bit_depth",                  Int(8), Int(16) );

      const Int maximumInputValue    = (1 << (((seiColourRemappingInfo->m_colourRemapInputBitDepth + 7) >> 3) << 3)) - 1;
      const Int maximumRemappedValue = (1 << (((seiColourRemappingInfo->m_colourRemapBitDepth      + 7) >> 3) << 3)) - 1;

      for( Int c=0 ; c<3 ; c++ )
      {
        readTokenValueAndValidate(seiColourRemappingInfo->m_preLutNumValMinus1[c],         failed, fic, "pre_lut_num_val_minus1[c]",        Int(0), Int(32) );
        if( seiColourRemappingInfo->m_preLutNumValMinus1[c]>0 )
        {
          seiColourRemappingInfo->m_preLut[c].resize(seiColourRemappingInfo->m_preLutNumValMinus1[c]+1);
          for( Int i=0 ; i<=seiColourRemappingInfo->m_preLutNumValMinus1[c] ; i++ )
          {
            readTokenValueAndValidate(seiColourRemappingInfo->m_preLut[c][i].codedValue,   failed, fic, "pre_lut_coded_value[c][i]",  Int(0), maximumInputValue    );
            readTokenValueAndValidate(seiColourRemappingInfo->m_preLut[c][i].targetValue,  failed, fic, "pre_lut_target_value[c][i]", Int(0), maximumRemappedValue );
          }
        }
      }
      readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapMatrixPresentFlag, failed, fic, "colour_remap_matrix_present_flag" );
      if( seiColourRemappingInfo->m_colourRemapMatrixPresentFlag )
      {
        readTokenValueAndValidate(seiColourRemappingInfo->m_log2MatrixDenom, failed, fic, "log2_matrix_denom", Int(0), Int(15) );
        for( Int c=0 ; c<3 ; c++ )
        {
          for( Int i=0 ; i<3 ; i++ )
          {
            readTokenValueAndValidate(seiColourRemappingInfo->m_colourRemapCoeffs[c][i], failed, fic, "colour_remap_coeffs[c][i]", -32768, 32767 );
          }
        }
      }
      for( Int c=0 ; c<3 ; c++ )
      {
        readTokenValueAndValidate(seiColourRemappingInfo->m_postLutNumValMinus1[c], failed, fic, "post_lut_num_val_minus1[c]", Int(0), Int(32) );
        if( seiColourRemappingInfo->m_postLutNumValMinus1[c]>0 )
        {
          seiColourRemappingInfo->m_postLut[c].resize(seiColourRemappingInfo->m_postLutNumValMinus1[c]+1);
          for( Int i=0 ; i<=seiColourRemappingInfo->m_postLutNumValMinus1[c] ; i++ )
          {
            readTokenValueAndValidate(seiColourRemappingInfo->m_postLut[c][i].codedValue,  failed, fic, "post_lut_coded_value[c][i]",  Int(0), maximumRemappedValue );
            readTokenValueAndValidate(seiColourRemappingInfo->m_postLut[c][i].targetValue, failed, fic, "post_lut_target_value[c][i]", Int(0), maximumRemappedValue );
          }
        }
      }
    }

    if( failed )
    {
      std::cerr << "Error while reading Colour Remapping Information SEI parameters file '" << colourRemapSEIFileWithPoc << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return true;
}

Void SEIEncoder::initSEIChromaResamplingFilterHint(SEIChromaResamplingFilterHint *seiChromaResamplingFilterHint, Int iHorFilterIndex, Int iVerFilterIndex)
{
  assert (m_isInitialized);
  assert (seiChromaResamplingFilterHint!=NULL);

  seiChromaResamplingFilterHint->m_verChromaFilterIdc = iVerFilterIndex;
  seiChromaResamplingFilterHint->m_horChromaFilterIdc = iHorFilterIndex;
  seiChromaResamplingFilterHint->m_verFilteringFieldProcessingFlag = 1;
  seiChromaResamplingFilterHint->m_targetFormatIdc = 3;
  seiChromaResamplingFilterHint->m_perfectReconstructionFlag = false;

  // this creates some example filter values, if explicit filter definition is selected
  if (seiChromaResamplingFilterHint->m_verChromaFilterIdc == 1)
  {
    const Int numVerticalFilters = 3;
    const Int verTapLengthMinus1[] = {5,3,3};

    seiChromaResamplingFilterHint->m_verFilterCoeff.resize(numVerticalFilters);
    for(Int i = 0; i < numVerticalFilters; i ++)
    {
      seiChromaResamplingFilterHint->m_verFilterCoeff[i].resize(verTapLengthMinus1[i]+1);
    }
    // Note: C++11 -> seiChromaResamplingFilterHint->m_verFilterCoeff[0] = {-3,13,31,23,3,-3};
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][0] = -3;
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][1] = 13;
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][2] = 31;
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][3] = 23;
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][4] = 3;
    seiChromaResamplingFilterHint->m_verFilterCoeff[0][5] = -3;

    seiChromaResamplingFilterHint->m_verFilterCoeff[1][0] = -1;
    seiChromaResamplingFilterHint->m_verFilterCoeff[1][1] = 25;
    seiChromaResamplingFilterHint->m_verFilterCoeff[1][2] = 247;
    seiChromaResamplingFilterHint->m_verFilterCoeff[1][3] = -15;

    seiChromaResamplingFilterHint->m_verFilterCoeff[2][0] = -20;
    seiChromaResamplingFilterHint->m_verFilterCoeff[2][1] = 186;
    seiChromaResamplingFilterHint->m_verFilterCoeff[2][2] = 100;
    seiChromaResamplingFilterHint->m_verFilterCoeff[2][3] = -10;
  }
  else
  {
    seiChromaResamplingFilterHint->m_verFilterCoeff.resize(0);
  }

  if (seiChromaResamplingFilterHint->m_horChromaFilterIdc == 1)
  {
    Int const numHorizontalFilters = 1;
    const Int horTapLengthMinus1[] = {3};

    seiChromaResamplingFilterHint->m_horFilterCoeff.resize(numHorizontalFilters);
    for(Int i = 0; i < numHorizontalFilters; i ++)
    {
      seiChromaResamplingFilterHint->m_horFilterCoeff[i].resize(horTapLengthMinus1[i]+1);
    }
    seiChromaResamplingFilterHint->m_horFilterCoeff[0][0] = 1;
    seiChromaResamplingFilterHint->m_horFilterCoeff[0][1] = 6;
    seiChromaResamplingFilterHint->m_horFilterCoeff[0][2] = 1;
  }
  else
  {
    seiChromaResamplingFilterHint->m_horFilterCoeff.resize(0);
  }
}

Void SEIEncoder::initSEITimeCode(SEITimeCode *seiTimeCode)
{
  assert (m_isInitialized);
  assert (seiTimeCode!=NULL);
  //  Set data as per command line options
  seiTimeCode->numClockTs = m_pcCfg->getNumberOfTimesets();
  for(Int i = 0; i < seiTimeCode->numClockTs; i++)
  {
    seiTimeCode->timeSetArray[i] = m_pcCfg->getTimeSet(i);
  }
}

Void SEIEncoder::initSEIAlternativeTransferCharacteristics(SEIAlternativeTransferCharacteristics *seiAltTransCharacteristics)
{
  assert (m_isInitialized);
  assert (seiAltTransCharacteristics!=NULL);
  //  Set SEI message parameters read from command line options
  seiAltTransCharacteristics->m_preferredTransferCharacteristics = m_pcCfg->getSEIPreferredTransferCharacteristics();
}

Void SEIEncoder::initSEIGreenMetadataInfo(SEIGreenMetadataInfo *seiGreenMetadataInfo, UInt u)
{
    assert (m_isInitialized);
    assert (seiGreenMetadataInfo!=NULL);

    seiGreenMetadataInfo->m_greenMetadataType = m_pcCfg->getSEIGreenMetadataType();
    seiGreenMetadataInfo->m_xsdMetricType = m_pcCfg->getSEIXSDMetricType();
    seiGreenMetadataInfo->m_xsdMetricValue = u;
}


//! \}
