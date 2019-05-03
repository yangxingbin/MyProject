#include "CensusMatch.h"


CensusMatch::CensusMatch()
{
}


CensusMatch::~CensusMatch()
{
}

void CensusMatch::processCensus5x5(uint8* leftImg, uint8* rightImg, float32* outputLeftDisImg, float32* outputRightDisImg,
	int width, int height, const int dispCount)
{
	const int maxDisp = dispCount - 1;

	// get disparity
	uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
	uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

	uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);

//#pragma omp parallel num_threads(2)
//	{
//#pragma omp sections nowait
//		{
//#pragma omp section
//			{
//				census5x5_SSE(leftImg, leftImgCensus, width, height);
//			}
//#pragma omp section
//			{
//				census5x5_SSE(rightImg, rightImgCensus, width, height);
//			}
//		}
//	}

	census5x5_SSE(leftImg, leftImgCensus, width, height);
	census5x5_SSE(rightImg, rightImgCensus, width, height);

	costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, 0, dsi);

	//matchWTAAndSubPixel_SSE(output, dsi, width, height, maxDisp, 0.95f);
	//matchWTAAndSubPixel_SSE(dispImgRight, dsi, width, height, maxDisp, 0.95f);   // invilid

	matchWTA_SSE_Left(outputLeftDisImg, dsi, width, height, maxDisp, 0.98);
	matchWTA_SSE_Right(outputRightDisImg, dsi, width, height, maxDisp, 0.98);

	subPixelRefine(outputLeftDisImg, dsi, width, height, maxDisp, 1);
}

/* fill disparity cube */
void CensusMatch::costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2,
	const sint32 height, const sint32 width, const sint32 dispCount, const uint16 invalidDispValue, uint16* dsi)
{
	// first 2 lines are empty
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < width; j++) {
			for (int d = 0; d <= dispCount - 1; d++) {
				*getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;
			}
		}
	}

	costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height - 2);

	/* last 2 lines are empty*/
	for (int i = height - 2; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int d = 0; d <= dispCount - 1; d++) {
				*getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;
			}
		}
	}
}

void CensusMatch::costMeasureCensus5x5Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2
	, const int width, const int dispCount, const uint16 invalidDispValue, uint16* dsi, const int lineStart, const int lineEnd)
{
	ALIGN16 const unsigned _LUT[] = { 0x02010100, 0x03020201, 0x03020201, 0x04030302 };
	const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
	const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

	for (sint32 i = lineStart; i < lineEnd; i++) {
		uint32* pBase = intermediate1 + i*width;
		uint32* pMatchRow = intermediate2 + i*width;
		for (uint32 j = 0; j < (uint32)width; j++) {
			uint32* pBaseJ = pBase + j;
			uint32* pMatchRowJmD = pMatchRow + j - dispCount + 1;

			int d = dispCount - 1;

			for (; d > (sint32)j && d >= 0; d--) {
				*getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;
				pMatchRowJmD++;
			}

			int dShift4m1 = ((d - 1) >> 2) * 4;
			int diff = d - dShift4m1;
			// rest
			if (diff != 0) {
				for (; diff >= 0 && d >= 0; d--, diff--) {
					uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
					*getDispAddr_xyd(dsi, width, dispCount, i, j, d) = cost;
					pMatchRowJmD++;
				}
			}

			// 4 costs at once
			__m128i lPoint4 = _mm_set1_epi32(*pBaseJ);
			d -= 3;

			uint16* baseAddr = getDispAddr_xyd(dsi, width, dispCount, i, j, 0);
			for (; d >= 0; d -= 4) {
				// flip the values
				__m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
				_mm_storel_pi((__m64*)(baseAddr + d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4), xmm7, xmm6)));
				pMatchRowJmD += 4;
			}

		}
	}
}

void CensusMatch::matchWTA_SSE_Left(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
	const uint32 factorUniq = (uint32)(1024 * uniqueness);
	const sint32 disp = maxDisp + 1;

	// find best by WTA
	float32* pDestDisp = dispImg;
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			// WTA on disparity values

			uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i, j, 0);
			uint16* pCostBase = pCost;
			uint32 minCost = *pCost;
			uint32 secMinCost = minCost;
			int secBestDisp = 0;

			const uint32 end = MIN(disp - 1, j);
			if (end == (uint32)disp - 1) {
				uint32 bestDisp = 0;

				for (uint32 loop = 0; loop < end; loop += 8) {
					// load costs
					const __m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					const __m128i b = _mm_minpos_epu16(costs);
					const int minValue = _mm_extract_epi16(b, 0);

					if ((uint32)minValue < minCost) {
						minCost = (uint32)minValue;
						bestDisp = _mm_extract_epi16(b, 1) + loop;
					}
					pCost += 8;
				}


				// get value of second minimum
				pCost = pCostBase;
				pCost[bestDisp] = 65535;

				__m128i secMinVector = _mm_set1_epi16(-1);
				const uint16* pCostEnd = pCost + disp;
				for (; pCost < pCostEnd; pCost += 8) {
					// load costs
					__m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					secMinVector = _mm_min_epu16(secMinVector, costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector), 0);


				pCostBase[bestDisp] = (uint16)minCost;

				// assign disparity
				if (1024 * minCost <= secMinCost*factorUniq) {
					*pDestDisp = (float)bestDisp;
				}
				else {
					bool check = false;
					if (bestDisp < (uint32)maxDisp - 1 && pCostBase[bestDisp + 1] == secMinCost) {
						check = true;
					}
					if (bestDisp > 0 && pCostBase[bestDisp - 1] == secMinCost) {
						check = true;
					}
					if (!check) {
						*pDestDisp = -10;
					}
					else {
						*pDestDisp = (float)bestDisp;
					}
				}

			}
			else {
				int bestDisp = 0;
				// for start
				for (uint32 k = 1; k <= end; k++) {
					pCost += 1;
					const uint16 cost = *pCost;
					if (cost < secMinCost) {
						if (cost < minCost) {
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						}
						else  {
							secMinCost = cost;
							secBestDisp = k;
						}
					}
				}
				// assign disparity
				if (1024 * minCost <= secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
					*pDestDisp = (float)bestDisp;
				}
				else {
					*pDestDisp = -10;
				}
			}
			pDestDisp++;
		}
	}
}

void CensusMatch::matchWTA_SSE_Right(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
	const uint32 factorUniq = (uint32)(1024 * uniqueness);

	const uint32 disp = maxDisp + 1;
	_ASSERT(disp <= 256);
	ALIGN32 uint16 store[256 + 32];
	store[15] = UINT16_MAX - 1;
	store[disp + 16] = UINT16_MAX - 1;

	// find best by WTA
	float32* pDestDisp = dispImg;
	for (uint32 i = 0; i < (uint32)height; i++) {
		for (uint32 j = 0; j < (uint32)width; j++) {
			// WTA on disparity values
			int bestDisp = 0;
			uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i, j, 0);
			sint32 minCost = *pCost;
			sint32 secMinCost = minCost;
			int secBestDisp = 0;
			const uint32 maxCurrDisp = MIN(disp - 1, width - 1 - j);

			if (maxCurrDisp == disp - 1) {

				// transfer to linear storage, slightly unrolled
				for (uint32 k = 0; k <= maxCurrDisp; k += 4) {
					store[k + 16] = *pCost;
					store[k + 16 + 1] = pCost[disp + 1];
					store[k + 16 + 2] = pCost[2 * disp + 2];
					store[k + 16 + 3] = pCost[3 * disp + 3];
					pCost += 4 * disp + 4;
				}
				// search in there
				uint16* pStore = &store[16];
				const uint16* pStoreEnd = pStore + disp;
				for (; pStore < pStoreEnd; pStore += 8) {
					// load costs
					const __m128i costs = _mm_load_si128((__m128i*)pStore);
					// get minimum for 8 values
					const __m128i b = _mm_minpos_epu16(costs);
					const int minValue = _mm_extract_epi16(b, 0);

					if (minValue < minCost) {
						minCost = minValue;
						bestDisp = _mm_extract_epi16(b, 1) + (int)(pStore - &store[16]);
					}

				}

				// get value of second minimum
				pStore = &store[16];
				store[16 + bestDisp] = 65535;
#ifndef USE_AVX2
				__m128i secMinVector = _mm_set1_epi16(-1);
				for (; pStore < pStoreEnd; pStore += 8) {
					// load costs
					__m128i costs = _mm_load_si128((__m128i*)pStore);
					// get minimum for 8 values
					secMinVector = _mm_min_epu16(secMinVector, costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector), 0);
#else
				__m256i secMinVector = _mm256_set1_epi16(-1);
				for (; pStore < pStoreEnd; pStore += 16) {
					// load costs
					__m256i costs = _mm256_load_si256((__m256i*)pStore);
					// get minimum for 8 values
					secMinVector = _mm256_min_epu16(secMinVector, costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
				int secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
				if (secMinCost2 < secMinCost)
					secMinCost = secMinCost2;
#endif

				// assign disparity
				if (1024U * minCost <= secMinCost*factorUniq) {
					*pDestDisp = (float)bestDisp;
				}
				else {
					bool check = (store[16 + bestDisp + 1] == secMinCost);
					check = check | (store[16 + bestDisp - 1] == secMinCost);
					if (!check) {
						*pDestDisp = -10;
					}
					else {
						*pDestDisp = (float)bestDisp;
					}
				}
				pDestDisp++;
			}
			else {
				// border case handling
				for (uint32 k = 1; k <= maxCurrDisp; k++) {
					pCost += disp + 1;
					const sint32 cost = (sint32)*pCost;
					if (cost < secMinCost) {
						if (cost < minCost) {
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						}
						else {
							secMinCost = cost;
							secBestDisp = k;
						}
					}
				}
				// assign disparity
				if (1024U * minCost <= factorUniq*secMinCost || abs(bestDisp - secBestDisp) < 2) {
					*pDestDisp = (float)bestDisp;
				}
				else {
					*pDestDisp = -10;
				}
				pDestDisp++;
			}
		}
	}
}

void CensusMatch::matchWTAAndSubPixel_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
	const uint32 factorUniq = (uint32)(1024 * uniqueness);
	const sint32 disp = maxDisp + 1;

	// find best by WTA
	float32* pDestDisp = dispImg;
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			// WTA on disparity values

			uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i, j, 0);
			uint16* pCostBase = pCost;
			uint32 minCost = *pCost;
			uint32 secMinCost = minCost;
			int secBestDisp = 0;

			const uint32 end = MIN(disp - 1, j);
			if (end == (uint32)disp - 1) {
				uint32 bestDisp = 0;

				for (uint32 loop = 0; loop < end; loop += 8) {
					// load costs
					const __m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					const __m128i b = _mm_minpos_epu16(costs);
					const int minValue = _mm_extract_epi16(b, 0);

					if ((uint32)minValue < minCost) {
						minCost = (uint32)minValue;
						bestDisp = _mm_extract_epi16(b, 1) + loop;
					}
					pCost += 8;
				}

				// get value of second minimum
				pCost = pCostBase;
				pCost[bestDisp] = 65535;

				__m128i secMinVector = _mm_set1_epi16(-1);
				const uint16* pCostEnd = pCost + disp;
				for (; pCost < pCostEnd; pCost += 8) {
					// load costs
					__m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					secMinVector = _mm_min_epu16(secMinVector, costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector), 0);
				pCostBase[bestDisp] = (uint16)minCost;

				// assign disparity
				if (1024 * minCost <= secMinCost*factorUniq) {
					*pDestDisp = (float)bestDisp;
				}
				else {
					bool check = false;
					if (bestDisp < (uint32)maxDisp - 1 && pCostBase[bestDisp + 1] == secMinCost) {
						check = true;
					}
					if (bestDisp > 0 && pCostBase[bestDisp - 1] == secMinCost) {
						check = true;
					}
					if (!check) {
						*pDestDisp = -10;
					}
					else {
						if (0 < bestDisp && bestDisp < (uint32)maxDisp - 1) {
							setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp - 1], minCost, pCostBase[bestDisp + 1]);
						}
						else {
							*pDestDisp = (float)bestDisp;
						}
					}
				}

			}
			else {
				int bestDisp = 0;
				// for start
				for (uint32 k = 1; k <= end; k++) {
					pCost += 1;
					const uint16 cost = *pCost;
					if (cost < secMinCost) {
						if (cost < minCost) {
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						}
						else  {
							secMinCost = cost;
							secBestDisp = k;
						}
					}
				}
				// assign disparity
				if (1024 * minCost <= secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
					if (0 < bestDisp && bestDisp < maxDisp - 1) {
						setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp - 1], minCost, pCostBase[bestDisp + 1]);
					}
					else {
						*pDestDisp = (float)bestDisp;
					}
				}
				else {
					*pDestDisp = -10;
				}
			}
			pDestDisp++;
		}
	}
}

void CensusMatch::doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
	float* dispRow = dispImg;
	float* dispCheckRow = dispCheckImg;
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 baseDisp = dispRow[j];
			if (baseDisp >= 0 && baseDisp <= j) {
				const float matchDisp = dispCheckRow[(int)(j - baseDisp)];

				sint32 diff = (sint32)(baseDisp - matchDisp);
				if (abs(diff) > lrThreshold) {
					dispRow[j] = -10;
				}
			}
			else {
				dispRow[j] = -10;
			}
		}
		dispRow += width;
		dispCheckRow += width;
	}
}

/*  do a sub pixel refinement by a parabola fit to the winning pixel and its neighbors */
void CensusMatch::subPixelRefine(float32* dispImg, uint16* dsiImg, const sint32 width, const sint32 height, const sint32 maxDisp, sint32 method)
{
	const sint32 disp_n = maxDisp + 1;

	/* equiangular */
	if (method == 0) {

		for (sint32 y = 0; y < height; y++)
		{
			uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
			float* disp = (float*)dispImg + y*width;

			for (sint32 x = 1; x < width - 1; x++, cost += disp_n)
			{
				if (disp[x] > 0.0) {

					// Get minimum
					int d_min = (int)disp[x];

					// Compute the equations of the parabolic fit
					uint16* costDmin = cost + d_min;
					sint32 c0 = costDmin[-1], c1 = *costDmin, c2 = costDmin[1];

					__m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(), c2 - c0);
					__m128 left = _mm_cvt_si2ss(_mm_setzero_ps(), c1 - c0);
					__m128 right = _mm_cvt_si2ss(_mm_setzero_ps(), c1 - c2);
					__m128 lowerMin = _mm_min_ss(left, right);
					__m128 result = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f), lowerMin)));

					__m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(), d_min);
					result = _mm_add_ss(baseDisp, result);
					_mm_store_ss(disp + x, result);

				}
				else {
					disp[x] = -10;
				}
			}
		}
		/* 1: parabolic Å×ÎïÏß*/
	}
	else if (method == 1){
		for (sint32 y = 0; y < height; y++)
		{
			uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
			float32* disp = dispImg + y*width;

			for (sint32 x = 1; x < width - 1; x++, cost += disp_n)
			{
				if (disp[x] > 0.0) {

					// Get minimum, but offset by 1 from ends
					int d_min = (int)disp[x];

					// Compute the equations of the parabolic fit
					sint32 c0 = cost[d_min - 1], c1 = cost[d_min], c2 = cost[d_min + 1];
					sint32 a = c0 + c0 - 4 * c1 + c2 + c2;
					sint32 b = (c0 - c2);

					// Solve for minimum, which is a correction term to d_min
					disp[x] = d_min + b / (float32)a;

				}
				else {
					disp[x] = -10;
				}
			}
		}
	}
}

// get window medain pixel
float CensusMatch::getMedainValue(float * windowPixel, int windowSize)
{
	// sort pixel
	int i, j;
	float temp;
	for (j = 0; j < windowSize * windowSize - 1; j++)
	{
		for (i = 0; i < windowSize * windowSize - 1 - j; i++)
		{
			if (windowPixel[i] > windowPixel[i + 1])
			{
				temp = windowPixel[i];
				windowPixel[i] = windowPixel[i + 1];
				windowPixel[i + 1] = temp;
			}
		}
	}

	return windowPixel[windowSize * windowSize / 2];
}

void CensusMatch::median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
{
	// check width restriction
	/*assert(width % 4 == 0); */

	float32* destStart = dest;
	//  lines
	float32* line1 = source;
	float32* line2 = source + width;
	float32* line3 = source + 2 * width;

	float32* end = source + width*height;

	dest += width;
	__m128 lastMedian = _mm_setzero_ps();

	do {
		// fill value
		const __m128 l1_reg = _mm_load_ps(line1);
		const __m128 l1_reg_next = _mm_load_ps(line1 + 4);
		__m128 v0 = l1_reg;
		__m128 v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next), _mm_castps_si128(l1_reg), 4));
		__m128 v2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next), _mm_castps_si128(l1_reg), 8));

		const __m128 l2_reg = _mm_load_ps(line2);
		const __m128 l2_reg_next = _mm_load_ps(line2 + 4);
		__m128 v3 = l2_reg;
		__m128 v4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next), _mm_castps_si128(l2_reg), 4));
		__m128 v5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next), _mm_castps_si128(l2_reg), 8));

		const __m128 l3_reg = _mm_load_ps(line3);
		const __m128 l3_reg_next = _mm_load_ps(line3 + 4);
		__m128 v6 = l3_reg;
		__m128 v7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next), _mm_castps_si128(l3_reg), 4));
		__m128 v8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next), _mm_castps_si128(l3_reg), 8));

		// find median through sorting network
		vecSortandSwap(v1, v2); vecSortandSwap(v4, v5); vecSortandSwap(v7, v8);
		vecSortandSwap(v0, v1); vecSortandSwap(v3, v4); vecSortandSwap(v6, v7);
		vecSortandSwap(v1, v2); vecSortandSwap(v4, v5); vecSortandSwap(v7, v8);
		vecSortandSwap(v0, v3); vecSortandSwap(v5, v8); vecSortandSwap(v4, v7);
		vecSortandSwap(v3, v6); vecSortandSwap(v1, v4); vecSortandSwap(v2, v5);
		vecSortandSwap(v4, v7); vecSortandSwap(v4, v2); vecSortandSwap(v6, v4);
		vecSortandSwap(v4, v2);

		// comply to alignment restrictions
		const __m128i c = _mm_alignr_epi8(_mm_castps_si128(v4), _mm_castps_si128(lastMedian), 12);
		_mm_store_si128((__m128i*)dest, c);
		lastMedian = v4;

		dest += 4; line1 += 4; line2 += 4; line3 += 4;

	} while (line3 + 4 + 4 <= end);

	memcpy(destStart, source, sizeof(float32)*(width + 1));
	memcpy(destStart + width*height - width - 1 - 3, source + width*height - width - 1 - 3, sizeof(float32)*(width + 1 + 3));
}

// void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
void CensusMatch::medianNxN(float32 * dispImgUnfiltered, float32 * &dispImg, uint32 width, uint32 height, uint32 windowSize)
{
	dispImg[0] = -10;
	float32 windowPix[9] = { 0.0 };
#pragma omp parallel for 
	for (int i = windowSize / 2; i < height - windowSize / 2; i++)
	{
		for (int j = windowSize / 2; j < width - windowSize / 2; j++)
		{
			uint32 pos = i * width + j;
			windowPix[0] = dispImgUnfiltered[pos - width - 1];  windowPix[1] = dispImgUnfiltered[pos - width]; windowPix[2] = dispImgUnfiltered[pos - width + 1];
			windowPix[3] = dispImgUnfiltered[pos - 1];  windowPix[4] = dispImgUnfiltered[pos]; windowPix[5] = dispImgUnfiltered[pos + 1];
			windowPix[6] = dispImgUnfiltered[pos + width - 1];  windowPix[7] = dispImgUnfiltered[pos + width]; windowPix[8] = dispImgUnfiltered[pos + width + 1];
			dispImg[pos] = getMedainValue(windowPix, windowSize);
		}
	}
}

void CensusMatch::census5x5_SSE(uint8* source, uint32* dest, uint32 width, uint32 height)
{
	uint32* dst = dest;
	const uint8* src = source;

	// input lines 0,1,2
	const uint8* i0 = src;

	// output at first result
	uint32* result = dst + 2 * width;
	const uint8* const end_input = src + width*height;

	/* expand mask */
	__m128i expandByte1_First4 = _mm_set_epi8(0x80u, 0x80u, 0x80u, 0x03u, 0x80u, 0x80u, 0x80u, 0x02u,
		0x80u, 0x80u, 0x80u, 0x01u, 0x80u, 0x80u, 0x80u, 0x00u);

	__m128i expandByte2_First4 = _mm_set_epi8(0x80u, 0x80u, 0x03u, 0x80u, 0x80u, 0x80u, 0x02u, 0x80u,
		0x80u, 0x80u, 0x01u, 0x80u, 0x80u, 0x80u, 0x00u, 0x80u);

	__m128i expandByte3_First4 = _mm_set_epi8(0x80u, 0x03u, 0x80u, 0x80u, 0x80u, 0x02u, 0x80u, 0x80u,
		0x80u, 0x01u, 0x80u, 0x80u, 0x80u, 0x00u, 0x80u, 0x80u);

	/* xor with 0x80, as it is a signed compare */
	__m128i l2_register = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 2 * width)), _mm_set1_epi8('\x80'));
	__m128i l3_register = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 3 * width)), _mm_set1_epi8('\x80'));
	__m128i l4_register = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 4 * width)), _mm_set1_epi8('\x80'));
	__m128i l1_register = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + width)), _mm_set1_epi8('\x80'));
	__m128i l0_register = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0)), _mm_set1_epi8('\x80'));

	i0 += 16;

	__m128i lastResult = _mm_setzero_si128();

	for (; i0 + 4 * width < end_input; i0 += 16) {

		/* parallel 16 pixel processing */
		const __m128i l0_register_next = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0)), _mm_set1_epi8('\x80'));
		const __m128i l1_register_next = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + width)), _mm_set1_epi8('\x80'));
		const __m128i l2_register_next = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 2 * width)), _mm_set1_epi8('\x80'));
		const __m128i l3_register_next = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 3 * width)), _mm_set1_epi8('\x80'));
		const __m128i l4_register_next = _mm_xor_si128(_mm_stream_load_si128((__m128i*)(i0 + 4 * width)), _mm_set1_epi8('\x80'));

		/*  0  1  2  3  4
		5  6  7  8  9
		10 11  c 12 13
		14 15 16 17 18
		19 20 21 22 23 */
		/* r/h is result, v is pixelvalue */

		/* pixel c */
		__m128i pixelcv = _mm_alignr_epi8(l2_register_next, l2_register, 2);

		/* pixel 0*/
		__m128i pixel0h = _mm_cmplt_epi8(l0_register, pixelcv);

		/* pixel 1*/
		__m128i pixel1v = _mm_alignr_epi8(l0_register_next, l0_register, 1);
		__m128i pixel1h = _mm_cmplt_epi8(pixel1v, pixelcv);

		/* pixel 2 */
		__m128i pixel2v = _mm_alignr_epi8(l0_register_next, l0_register, 2);
		__m128i pixel2h = _mm_cmplt_epi8(pixel2v, pixelcv);

		/* pixel 3 */
		__m128i pixel3v = _mm_alignr_epi8(l0_register_next, l0_register, 3);
		__m128i pixel3h = _mm_cmplt_epi8(pixel3v, pixelcv);

		/* pixel 4 */
		__m128i pixel4v = _mm_alignr_epi8(l0_register_next, l0_register, 4);
		__m128i pixel4h = _mm_cmplt_epi8(pixel4v, pixelcv);

		/** line  **/
		/* pixel 5 */
		__m128i pixel5h = _mm_cmplt_epi8(l1_register, pixelcv);

		/* pixel 6 */
		__m128i pixel6v = _mm_alignr_epi8(l1_register_next, l1_register, 1);
		__m128i pixel6h = _mm_cmplt_epi8(pixel6v, pixelcv);

		/* pixel 7 */
		__m128i pixel7v = _mm_alignr_epi8(l1_register_next, l1_register, 2);
		__m128i pixel7h = _mm_cmplt_epi8(pixel7v, pixelcv);

		/* pixel 8 */
		__m128i pixel8v = _mm_alignr_epi8(l1_register_next, l1_register, 3);
		__m128i pixel8h = _mm_cmplt_epi8(pixel8v, pixelcv);

		/* pixel 9 */
		__m128i pixel9v = _mm_alignr_epi8(l1_register_next, l1_register, 4);
		__m128i pixel9h = _mm_cmplt_epi8(pixel9v, pixelcv);

		/* create bitfield part 1*/
		__m128i resultByte1 = _mm_and_si128(_mm_set1_epi8(128u), pixel0h);
		resultByte1 = _mm_or_si128(resultByte1, _mm_and_si128(_mm_set1_epi8(64), pixel1h));
		resultByte1 = _mm_or_si128(resultByte1, _mm_and_si128(_mm_set1_epi8(32), pixel2h));
		resultByte1 = _mm_or_si128(resultByte1, _mm_and_si128(_mm_set1_epi8(16), pixel3h));
		__m128i resultByte1b = _mm_and_si128(_mm_set1_epi8(8), pixel4h);
		resultByte1b = _mm_or_si128(resultByte1b, _mm_and_si128(_mm_set1_epi8(4), pixel5h));
		resultByte1b = _mm_or_si128(resultByte1b, _mm_and_si128(_mm_set1_epi8(2), pixel6h));
		resultByte1b = _mm_or_si128(resultByte1b, _mm_and_si128(_mm_set1_epi8(1), pixel7h));
		resultByte1 = _mm_or_si128(resultByte1, resultByte1b);

		/** line **/
		/* pixel 10 */
		__m128i pixel10h = _mm_cmplt_epi8(l2_register, pixelcv);

		/* pixel 11 */
		__m128i pixel11v = _mm_alignr_epi8(l2_register_next, l2_register, 1);
		__m128i pixel11h = _mm_cmplt_epi8(pixel11v, pixelcv);

		/* pixel 12 */
		__m128i pixel12v = _mm_alignr_epi8(l2_register_next, l2_register, 3);
		__m128i pixel12h = _mm_cmplt_epi8(pixel12v, pixelcv);

		/* pixel 13 */
		__m128i pixel13v = _mm_alignr_epi8(l2_register_next, l2_register, 4);
		__m128i pixel13h = _mm_cmplt_epi8(pixel13v, pixelcv);

		/* line */
		/* pixel 14 */
		__m128i pixel14h = _mm_cmplt_epi8(l3_register, pixelcv);

		/* pixel 15 */
		__m128i pixel15v = _mm_alignr_epi8(l3_register_next, l3_register, 1);
		__m128i pixel15h = _mm_cmplt_epi8(pixel15v, pixelcv);

		/* pixel 16 */
		__m128i pixel16v = _mm_alignr_epi8(l3_register_next, l3_register, 2);
		__m128i pixel16h = _mm_cmplt_epi8(pixel16v, pixelcv);

		/* pixel 17 */
		__m128i pixel17v = _mm_alignr_epi8(l3_register_next, l3_register, 3);
		__m128i pixel17h = _mm_cmplt_epi8(pixel17v, pixelcv);

		/* pixel 18 */
		__m128i pixel18v = _mm_alignr_epi8(l3_register_next, l3_register, 4);
		__m128i pixel18h = _mm_cmplt_epi8(pixel18v, pixelcv);

		/* create bitfield part 2 */
		__m128i resultByte2 = _mm_and_si128(_mm_set1_epi8(128u), pixel8h);
		resultByte2 = _mm_or_si128(resultByte2, _mm_and_si128(_mm_set1_epi8(64), pixel9h));
		resultByte2 = _mm_or_si128(resultByte2, _mm_and_si128(_mm_set1_epi8(32), pixel10h));
		resultByte2 = _mm_or_si128(resultByte2, _mm_and_si128(_mm_set1_epi8(16), pixel11h));
		__m128i resultByte2b = _mm_and_si128(_mm_set1_epi8(8), pixel12h);
		resultByte2b = _mm_or_si128(resultByte2b, _mm_and_si128(_mm_set1_epi8(4), pixel13h));
		resultByte2b = _mm_or_si128(resultByte2b, _mm_and_si128(_mm_set1_epi8(2), pixel14h));
		resultByte2b = _mm_or_si128(resultByte2b, _mm_and_si128(_mm_set1_epi8(1), pixel15h));
		resultByte2 = _mm_or_si128(resultByte2, resultByte2b);

		/* line */
		/* pixel 19 */
		__m128i pixel19h = _mm_cmplt_epi8(l4_register, pixelcv);

		/* pixel 20 */
		__m128i pixel20v = _mm_alignr_epi8(l4_register_next, l4_register, 1);
		__m128i pixel20h = _mm_cmplt_epi8(pixel20v, pixelcv);

		/* pixel 21 */
		__m128i pixel21v = _mm_alignr_epi8(l4_register_next, l4_register, 2);
		__m128i pixel21h = _mm_cmplt_epi8(pixel21v, pixelcv);

		/* pixel 22 */
		__m128i pixel22v = _mm_alignr_epi8(l4_register_next, l4_register, 3);
		__m128i pixel22h = _mm_cmplt_epi8(pixel22v, pixelcv);

		/* pixel 23 */
		__m128i pixel23v = _mm_alignr_epi8(l4_register_next, l4_register, 4);
		__m128i pixel23h = _mm_cmplt_epi8(pixel23v, pixelcv);

		/* create bitfield part 3*/

		__m128i resultByte3 = _mm_and_si128(_mm_set1_epi8(128u), pixel16h);
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(64), pixel17h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(32), pixel18h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(16), pixel19h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(8), pixel20h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(4), pixel21h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(2), pixel22h));
		resultByte3 = _mm_or_si128(resultByte3, _mm_and_si128(_mm_set1_epi8(1), pixel23h));

		/* blend the first part together */
		__m128i resultPart1_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
		__m128i resultPart1_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
		__m128i resultPart1 = _mm_or_si128(resultPart1_1, resultPart1_2);
		__m128i resultPart1_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
		resultPart1 = _mm_or_si128(resultPart1, resultPart1_3);

		/* rotate result bytes */
		/* replace by _mm_alignr_epi8 */
		resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0, 3, 2, 1));

		/* blend the second part together */
		__m128i resultPart2_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
		__m128i resultPart2_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
		__m128i resultPart2 = _mm_or_si128(resultPart2_1, resultPart2_2);
		__m128i resultPart2_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
		resultPart2 = _mm_or_si128(resultPart2, resultPart2_3);

		/* rotate result bytes */
		resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0, 3, 2, 1));

		/* blend the third part together */
		__m128i resultPart3_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
		__m128i resultPart3_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
		__m128i resultPart3 = _mm_or_si128(resultPart3_1, resultPart3_2);
		__m128i resultPart3_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
		resultPart3 = _mm_or_si128(resultPart3, resultPart3_3);

		/* rotate result bytes */
		resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0, 3, 2, 1));
		resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0, 3, 2, 1));

		/* blend the fourth part together */
		__m128i resultPart4_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
		__m128i resultPart4_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
		__m128i resultPart4 = _mm_or_si128(resultPart4_1, resultPart4_2);
		__m128i resultPart4_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
		resultPart4 = _mm_or_si128(resultPart4, resultPart4_3);

		/* shift because of offsets */
		_mm_store_si128((__m128i*)result, _mm_alignr_epi8(resultPart1, lastResult, 8));
		_mm_store_si128((__m128i*)(result + 4), _mm_alignr_epi8(resultPart2, resultPart1, 8));
		_mm_store_si128((__m128i*)(result + 8), _mm_alignr_epi8(resultPart3, resultPart2, 8));
		_mm_store_si128((__m128i*)(result + 12), _mm_alignr_epi8(resultPart4, resultPart3, 8));

		result += 16;
		lastResult = resultPart4;

		/*load next */
		l0_register = l0_register_next;
		l1_register = l1_register_next;
		l2_register = l2_register_next;
		l3_register = l3_register_next;
		l4_register = l4_register_next;

	}
	/* last pixels */
	{
		int i = height - 3;
		for (sint32 j = width - 16 + 2; j < (sint32)width - 2; j++) {
			const int centerValue = *getPixel8(source, width, j, i);
			uint32 value = 0;
			for (sint32 x = -2; x <= 2; x++) {
				for (sint32 y = -2; y <= 2; y++) {
					if (x != 0 || y != 0) {
						value *= 2;
						if (centerValue >  *getPixel8(source, width, j + y, i + x)) {
							value += 1;
						}
					}
				}
			}
			*getPixel32(dest, width, j, i) = value;
		}
	}
}


