#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <limits.h>
#include <omp.h>

#define ALIGN16 __declspec(align(16))
#define HW_POPCNT
#define FORCEINLINE __forceinline

#ifdef HW_POPCNT
#define POPCOUNT32 _mm_popcnt_u32
#if defined(_M_X64) || defined(__amd64__) || defined(__amd64)
#define POPCOUNT64 (uint16)_mm_popcnt_u64
#else 
#define POPCOUNT64 popcount64LUT
#endif
#else
#define POPCOUNT32 popcount32
#define POPCOUNT64 popcount64LUT
#endif

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))

// Types
typedef float float32;
typedef double float64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int8_t sint8;
typedef int16_t sint16;
typedef int32_t sint32;
// Defines

#define ALIGN32 __declspec(align(32))

FORCEINLINE __m128 rcp_nz_ss(__m128 input) {
	__m128 mask = _mm_cmpeq_ss(_mm_set1_ps(0.0), input);
	__m128 recip = _mm_rcp_ss(input);
	return _mm_andnot_ps(mask, recip);
}

class CensusMatch
{
public:
	CensusMatch();
	~CensusMatch();

	void processCensus5x5(uint8* leftImg, uint8* rightImg, float32* outputLeftDisImg, float32* outputRightDisImg,
		int width, int height, const int dispCount);

	void matchWTAAndSubPixel_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);

	void matchWTA_SSE_Left(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);

	void matchWTA_SSE_Right(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);

	void medianNxN(float32 * dispImgUnfiltered, float32 * &dispImg, uint32 width, uint32 height, uint32 windowSize);

	void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height);
	
	void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold);

	/*  do a sub pixel refinement by a parabola fit to the winning pixel and its neighbors */
	void subPixelRefine(float32* dispImg, uint16* dsiImg, const sint32 width, const sint32 height, const sint32 maxDisp, sint32 method);

private:

	inline uint32* getPixel32(uint32* base, uint32 width, int j, int i)
	{
		return base + i*width + j;
	}

	inline uint8* getPixel8(uint8* base, uint32 width, int j, int i)
	{
		return base + i*width + j;
	}

	inline uint16* getDispAddr_xyd(uint16* dsi, sint32 width, sint32 disp, sint32 i, sint32 j, sint32 k)
	{
		return dsi + i*(disp*width) + j*disp + k;
	}

	// pop count for 4 32bit values
	inline FORCEINLINE __m128i popcount32_4(const __m128i& a, const __m128i& lut, const __m128i& mask)
	{
		__m128i b = _mm_srli_epi16(a, 4); // psrlw       $4, %%xmm1

		__m128i a2 = _mm_and_si128(a, mask); // pand    %%xmm6, %%xmm0  ; xmm0 - lower nibbles
		b = _mm_and_si128(b, mask); // pand    %%xmm6, %%xmm1  ; xmm1 - higher nibbles

		__m128i popA = _mm_shuffle_epi8(lut, a2); // pshufb  %%xmm0, %%xmm2  ; xmm2 = vector of popcount for lower nibbles
		__m128i popB = _mm_shuffle_epi8(lut, b); //  pshufb  %%xmm1, %%xmm3  ; xmm3 = vector of popcount for higher nibbles

		__m128i popByte = _mm_add_epi8(popA, popB); // paddb   %%xmm3, %%xmm2  ; xmm2 += xmm3 -- vector of popcount for bytes;

		// How to get to added quadwords?
		const __m128i ZERO = _mm_setzero_si128();

		// Version 1 - with horizontal adds

		__m128i upper = _mm_unpackhi_epi8(popByte, ZERO);
		__m128i lower = _mm_unpacklo_epi8(popByte, ZERO);
		__m128i popUInt16 = _mm_hadd_epi16(lower, upper); // uint16 pop count
		// the lower four 16 bit values contain the uint32 pop count
		__m128i popUInt32 = _mm_hadd_epi16(popUInt16, ZERO);

		return popUInt32;
	}

	inline FORCEINLINE void setSubpixelValue(float32* dest, uint32 bestDisp, const sint32& c0, const sint32& c1, const sint32& c2)
	{
		__m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(), c2 - c0);
		__m128 left = _mm_cvt_si2ss(_mm_setzero_ps(), c1 - c0);
		__m128 right = _mm_cvt_si2ss(_mm_setzero_ps(), c1 - c2);
		__m128 lowerMin = _mm_min_ss(left, right);
		__m128 d_offset = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f), lowerMin)));
		__m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(), bestDisp);
		__m128 result = _mm_add_ss(baseDisp, d_offset);
		_mm_store_ss(dest, result);
	}

	inline void vecSortandSwap(__m128& a, __m128& b)
	{
		__m128 temp = a;
		a = _mm_min_ps(a, b);
		b = _mm_max_ps(temp, b);
	}

	/* Optimized Versions */
	void census5x5_SSE(uint8* source, uint32* dest, uint32 width, uint32 height);

	/* fill disparity cube */
	void costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2,
		const sint32 height, const sint32 width, const sint32 dispCount, const uint16 invalidDispValue, uint16* dsi);

	void costMeasureCensus5x5Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,
		const int width, const int dispCount, const uint16 invalidDispValue, uint16* dsi, const int lineStart, const int lineEnd);

	float getMedainValue(float * windowPixel, int windowSize);

};

