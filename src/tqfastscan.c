#include "postgres.h"

#include <math.h>

#include "halfvec.h"
#include "tq.h"

/*
 * tqfastscan.c -- blocked fast-scan code layout + scoring kernels.
 *
 * Pure, index-structure-agnostic: a block is up to TQ_BLOCK_WIDTH vectors whose
 * 4-bit codes are stored coordinate-strided and nibble-packed (low nibbles =
 * lanes 0..15, high nibbles = lanes 16..31).  The kernel performs a per-query
 * 8-bit LUT table lookup per coordinate across all 32 lanes, accumulating in
 * uint16 lanes flushed to uint32 every TQ_LUT_FLUSH coords (255*128 < 65535).
 *
 * Only the code-plane is read here; side data (heaptid/norm/scale) lives
 * elsewhere.  This file is what an IVF+TurboQuant integration embeds verbatim.
 */

/* Unpack 4-bit code `idx` from row-major packed codes (2 codes per byte). */
static inline uint8
fs_unpack4(const char *codes, int idx)
{
	uint8		b = (uint8) codes[idx >> 1];

	return (idx & 1) ? (uint8) (b >> 4) : (uint8) (b & 0x0F);
}

void
TqScatterCodes(const TqModel *model, const char *codes, int slot, uint8 *codePlane)
{
	int			dc = model->dimCodes;
	int			i;
	int			lane = slot & 15;
	bool		high = slot >= 16;

	for (i = 0; i < dc; i++)
	{
		uint8		code = fs_unpack4(codes, i);
		uint8	   *cell = codePlane + (Size) i * 16 + lane;

		if (high)
			*cell = (uint8) ((*cell & 0x0F) | (code << 4));
		else
			*cell = (uint8) ((*cell & 0xF0) | (code & 0x0F));
	}
}

void
TqBlockAccumInit(TqBlockAccum *acc)
{
	memset(acc->acc16, 0, sizeof(acc->acc16));
	memset(acc->acc32, 0, sizeof(acc->acc32));
	acc->sinceFlush = 0;
}

/* Flush uint16 lane stage into uint32 and reset the stage. */
static inline void
fs_flush(TqBlockAccum *acc)
{
	int			j;

	for (j = 0; j < TQ_BLOCK_WIDTH; j++)
	{
		acc->acc32[j] += acc->acc16[j];
		acc->acc16[j] = 0;
	}
	acc->sinceFlush = 0;
}

void
TqBlockAccumFinish(TqBlockAccum *acc)
{
	fs_flush(acc);
}

void
TqScoreBlockRangeDefault(const uint8 *lut8, const uint8 *codeRun,
						 int c0, int c1, TqBlockAccum *acc)
{
	int			i;

	for (i = c0; i < c1; i++)
	{
		const uint8 *cell = codeRun + (Size) (i - c0) * 16;
		const uint8 *tbl = lut8 + (Size) i * 16;	/* nLevels == 16 */
		int			j;

		for (j = 0; j < 16; j++)
		{
			uint8		packed = cell[j];

			acc->acc16[j] += tbl[packed & 0x0F];
			acc->acc16[j + 16] += tbl[packed >> 4];
		}

		if (++acc->sinceFlush >= TQ_LUT_FLUSH)
			fs_flush(acc);
	}
}

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>

void
TqScoreBlockRangeNeon(const uint8 *lut8, const uint8 *codeRun,
					  int c0, int c1, TqBlockAccum *acc)
{
	uint8x16_t	mask = vdupq_n_u8(0x0F);
	int			i;

	for (i = c0; i < c1; i++)
	{
		const uint8 *cell = codeRun + (Size) (i - c0) * 16;
		uint8x16_t	tbl = vld1q_u8(lut8 + (Size) i * 16);
		uint8x16_t	codes = vld1q_u8(cell);
		uint8x16_t	lo = vandq_u8(codes, mask);
		uint8x16_t	hi = vshrq_n_u8(codes, 4); /* per-byte shift -> 0..15 */
		uint8x16_t	r0 = vqtbl1q_u8(tbl, lo);	/* lanes 0..15 (low nibble) */
		uint8x16_t	r1 = vqtbl1q_u8(tbl, hi);	/* lanes 16..31 (high nibble) */
		uint16x8_t	a0 = vld1q_u16(acc->acc16 + 0);
		uint16x8_t	a1 = vld1q_u16(acc->acc16 + 8);
		uint16x8_t	a2 = vld1q_u16(acc->acc16 + 16);
		uint16x8_t	a3 = vld1q_u16(acc->acc16 + 24);

		a0 = vaddw_u8(a0, vget_low_u8(r0));
		a1 = vaddw_u8(a1, vget_high_u8(r0));
		a2 = vaddw_u8(a2, vget_low_u8(r1));
		a3 = vaddw_u8(a3, vget_high_u8(r1));

		vst1q_u16(acc->acc16 + 0, a0);
		vst1q_u16(acc->acc16 + 8, a1);
		vst1q_u16(acc->acc16 + 16, a2);
		vst1q_u16(acc->acc16 + 24, a3);

		if (++acc->sinceFlush >= TQ_LUT_FLUSH)
			fs_flush(acc);
	}
}
#endif							/* __aarch64__ || __ARM_NEON */

#if defined(USE_DISPATCH)
#include <immintrin.h>
#if defined(USE__GET_CPUID)
#include <cpuid.h>
#else
#include <intrin.h>
#endif

#ifdef _MSC_VER
#define TARGET_TQ_AVX512
#define TARGET_TQ_XSAVE
#else
#define TARGET_TQ_AVX512 __attribute__((target("avx512f,avx512bw")))
#define TARGET_TQ_XSAVE  __attribute__((target("xsave")))
#endif

#define TQ_CPU_OSXSAVE   (1u << 27)	/* CPUID leaf 1 ECX bit 27 */
#define TQ_CPU_AVX512F   (1u << 16)	/* CPUID leaf 7,0 EBX bit 16 */
#define TQ_CPU_AVX512BW  (1u << 30)	/* CPUID leaf 7,0 EBX bit 30 */

/*
 * TqSupportsAvx512 -- runtime probe for AVX-512F + AVX-512BW, gated by
 * OSXSAVE and the OS XMM/YMM/ZMM-state xgetbv check.
 * Mirrors bitutils.c SupportsAvx512Popcount but checks AVX512BW (bit 30)
 * instead of AVX512VPOPCNTDQ (bit 14 ECX).
 * Non-static: referenced by TqInitDispatch in tqdistance.c (separate TU).
 */
TARGET_TQ_XSAVE bool
TqSupportsAvx512(void)
{
	unsigned int exx[4] = {0, 0, 0, 0};

#if defined(USE__GET_CPUID)
	__get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuid(exx, 1);
#endif

	/* Check OS supports XSAVE */
	if ((exx[2] & TQ_CPU_OSXSAVE) != TQ_CPU_OSXSAVE)
		return false;

	/* Check XMM, YMM, and ZMM registers are OS-enabled */
	if ((_xgetbv(0) & 0xe6) != 0xe6)
		return false;

#if defined(USE__GET_CPUID)
	__get_cpuid_count(7, 0, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuidex(exx, 7, 0);
#endif

	/* Check AVX512F (EBX bit 16) and AVX512BW (EBX bit 30) */
	return (exx[1] & TQ_CPU_AVX512F) != 0 &&
		(exx[1] & TQ_CPU_AVX512BW) != 0;
}

/*
 * TqScoreBlockRangeAvx512 -- 128-bit-shuffle fast-scan kernel, bit-identical
 * to TqScoreBlockRangeDefault.
 *
 * Lane-mapping rationale (independently verified):
 *   codes = 16 packed bytes; cell[j] holds lo-nibble for lane j (0..15) and
 *   hi-nibble for lane j+16 (16..31).
 *   lo[j] = cell[j] & 0x0F  -> tbl lookup -> r0 byte j = tbl[cell[j]&0x0F]
 *   hi[j] = (cell[j]>>4)&0x0F -> tbl lookup -> r1 byte j = tbl[cell[j]>>4]
 *   _mm_unpacklo_epi8(r0, z) -> u16 { r0[0]..r0[7] }  -> acc16[0..7]
 *   _mm_unpackhi_epi8(r0, z) -> u16 { r0[8]..r0[15] } -> acc16[8..15]
 *   _mm_unpacklo_epi8(r1, z) -> u16 { r1[0]..r1[7] }  -> acc16[16..23]
 *   _mm_unpackhi_epi8(r1, z) -> u16 { r1[8]..r1[15] } -> acc16[24..31]
 *   Scalar: acc16[j] += tbl[cell[j]&0x0F]; acc16[j+16] += tbl[cell[j]>>4]
 *   => r0 byte j -> acc16[j], r1 byte j -> acc16[j+16]. Match. ✓
 *
 * CRITICAL x86-vs-NEON difference: _mm_srli_epi16 is a 16-BIT shift; it
 * bleeds bits from the adjacent byte into the top of each byte.  The 0x0F
 * mask after the shift is mandatory to zero those bleed bits before the
 * table lookup.  NEON's vshrq_n_u8 is a true per-byte shift and needs no
 * post-mask.
 *
 * NOTE: amd64 consistency with TqScoreBlockRangeDefault is deferred to the
 * amd64 box; tqflat_test_score_block_consistency asserts ==Default there.
 * TODO(amd64 throughput): widen to _mm512_shuffle_epi8 (4 coords/iter).
 */
TARGET_TQ_AVX512 void
TqScoreBlockRangeAvx512(const uint8 *lut8, const uint8 *codeRun,
						int c0, int c1, TqBlockAccum *acc)
{
	const __m128i mask = _mm_set1_epi8(0x0F);
	int			i;

	for (i = c0; i < c1; i++)
	{
		const uint8 *cell = codeRun + (Size) (i - c0) * 16;
		__m128i		tbl = _mm_loadu_si128((const __m128i *) (lut8 + (Size) i * 16));
		__m128i		codes = _mm_loadu_si128((const __m128i *) cell);
		__m128i		lo = _mm_and_si128(codes, mask);

		/*
		 * _mm_srli_epi16 shifts 16-bit lanes right by 4: bits [7:4] of byte N
		 * get bits [11:8] of the 16-bit word, i.e. the low 4 bits of byte N+1
		 * bleed in.  The 0x0F mask discards those bleed bits, leaving only the
		 * original hi nibble of each byte.
		 */
		__m128i		hi = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
		__m128i		r0 = _mm_shuffle_epi8(tbl, lo);	/* result for lanes 0..15 */
		__m128i		r1 = _mm_shuffle_epi8(tbl, hi);	/* result for lanes 16..31 */
		__m128i		z = _mm_setzero_si128();
		__m128i		a0 = _mm_loadu_si128((const __m128i *) (acc->acc16 + 0));
		__m128i		a1 = _mm_loadu_si128((const __m128i *) (acc->acc16 + 8));
		__m128i		a2 = _mm_loadu_si128((const __m128i *) (acc->acc16 + 16));
		__m128i		a3 = _mm_loadu_si128((const __m128i *) (acc->acc16 + 24));

		/*
		 * Widen u8 -> u16 by interleaving with zero, then add.
		 * unpacklo(r0, z): bytes 0..7  of r0 -> u16 words 0..7  -> acc16[0..7]
		 * unpackhi(r0, z): bytes 8..15 of r0 -> u16 words 0..7  -> acc16[8..15]
		 * unpacklo(r1, z): bytes 0..7  of r1 -> u16 words 0..7  -> acc16[16..23]
		 * unpackhi(r1, z): bytes 8..15 of r1 -> u16 words 0..7  -> acc16[24..31]
		 */
		a0 = _mm_add_epi16(a0, _mm_unpacklo_epi8(r0, z));
		a1 = _mm_add_epi16(a1, _mm_unpackhi_epi8(r0, z));
		a2 = _mm_add_epi16(a2, _mm_unpacklo_epi8(r1, z));
		a3 = _mm_add_epi16(a3, _mm_unpackhi_epi8(r1, z));

		_mm_storeu_si128((__m128i *) (acc->acc16 + 0), a0);
		_mm_storeu_si128((__m128i *) (acc->acc16 + 8), a1);
		_mm_storeu_si128((__m128i *) (acc->acc16 + 16), a2);
		_mm_storeu_si128((__m128i *) (acc->acc16 + 24), a3);

		if (++acc->sinceFlush >= TQ_LUT_FLUSH)
			fs_flush(acc);
	}
}
#endif							/* USE_DISPATCH */

void		(*TqScoreBlockRange) (const uint8 *lut8, const uint8 *codeRun,
								  int c0, int c1, TqBlockAccum *acc);

void
TqBuildLut8(const TqModel *model, const float *lut,
			uint8 *lut8, float *lutBias, float *lutScale)
{
	int			n = model->dimCodes * model->nLevels;
	float		lo = lut[0];
	float		hi = lut[0];
	float		scale;
	int			i;

	for (i = 1; i < n; i++)
	{
		if (lut[i] < lo)
			lo = lut[i];
		if (lut[i] > hi)
			hi = lut[i];
	}

	if (hi - lo < 1e-30f)
		hi = lo + 1e-30f;		/* degenerate (constant LUT) guard */

	scale = (hi - lo) / 255.0f;
	for (i = 0; i < n; i++)
	{
		int			q = (int) lroundf((lut[i] - lo) / scale);

		lut8[i] = (uint8) (q < 0 ? 0 : (q > 255 ? 255 : q));
	}

	*lutBias = lo;
	*lutScale = scale;
}
