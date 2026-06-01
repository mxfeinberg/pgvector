/*
 * tq_simd_microbench.c -- throwaway diagnostic for the tqflat SIMD-kernel
 * layout decision (Option A blocked-on-disk vs Option B row-major +
 * transpose-on-scan).  NOT part of the build; arm64/NEON only.
 *
 * Measures the per-query *score* step (scan of all N stored vectors) three ways:
 *   scalar : faithful copy of TqScoreEntryDefault -- unpack 4-bit code, gather
 *            from a float LUT, accumulate.  (current production path)
 *   B      : codes stored row-major packed 4-bit (today's on-disk format);
 *            per 32-vector block, unpack+transpose into a scratch buffer, then
 *            run the NEON vqtbl1q_u8 fast-scan.  (Option B)
 *   A      : codes already transposed/blocked in memory; time ONLY the NEON
 *            fast-scan.  (Option A upper bound / pure kernel ceiling)
 *
 * scalar/B = the speedup Option B actually delivers.
 * B/A      = the extra Option A's on-disk format would buy over B.
 *
 * Caveats (acceptable for a go/no-go): SIMD path uses an 8-bit quantized LUT
 * (real fast-scan does too); scratch is 1 byte/code, not nibble-packed (so B's
 * transpose moves ~2x the bytes a nibble-packed kernel would -- conservative
 * against B); uint16 lane accumulation flushed to uint32 every 128 coords.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLOCK 32
#define NLEVELS 16			/* bits = 4 */

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* unpack 4-bit code i from a packed row (2 codes per byte). */
static inline uint8_t
unpack4(const uint8_t *codes, int i)
{
	uint8_t		b = codes[i >> 1];

	return (i & 1) ? (b >> 4) : (b & 0x0F);
}

int
main(int argc, char **argv)
{
	int			dc = argc > 1 ? atoi(argv[1]) : 2048;	/* padded dim */
	long		N = argc > 2 ? atol(argv[2]) : (1L << 20);
	int			reps = argc > 3 ? atoi(argv[3]) : 3;
	int			bytesPerVec = (dc + 1) / 2;	/* 4-bit packed */

	N = (N / BLOCK) * BLOCK;	/* whole blocks */

	printf("dc=%d  N=%ld  reps=%d  (4-bit codes, %d B/vec packed)\n",
		   dc, N, reps, bytesPerVec);

	/* ---- synthetic data ---------------------------------------------- */
	uint8_t    *codes = malloc((size_t) N * bytesPerVec);	/* row-major packed */
	float	   *lutf = malloc(sizeof(float) * dc * NLEVELS); /* scalar float LUT */
	uint8_t    *lut8 = malloc((size_t) dc * NLEVELS);		/* 8-bit fast-scan LUT */

	srandom(42);
	for (size_t i = 0; i < (size_t) N * bytesPerVec; i++)
		codes[i] = (uint8_t) random();
	float		lutmax = 0.0f;

	for (int i = 0; i < dc * NLEVELS; i++)
	{
		float		v = (float) ((random() / (double) RAND_MAX) * 2.0 - 1.0);

		lutf[i] = v;
		if (fabsf(v) > lutmax)
			lutmax = fabsf(v);
	}
	/* quantize LUT to [0,255] per fast-scan (shift+scale; offset dropped --
	 * we only need representative *work*, not the exact estimate here). */
	for (int i = 0; i < dc * NLEVELS; i++)
	{
		int			q = (int) lroundf((lutf[i] + lutmax) / (2 * lutmax) * 255.0f);

		lut8[i] = (uint8_t) (q < 0 ? 0 : q > 255 ? 255 : q);
	}

	volatile double sink = 0.0;

	/* ---- scalar (TqScoreEntryDefault) -------------------------------- */
	double		t_scalar = 1e30;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		double		acc = 0.0;

		for (long v = 0; v < N; v++)
		{
			const uint8_t *row = codes + (size_t) v * bytesPerVec;
			double		mse = 0.0;

			for (int i = 0; i < dc; i++)
				mse += lutf[i * NLEVELS + unpack4(row, i)];
			acc += mse;
		}
		double		t = now_ms() - t0;

		if (t < t_scalar)
			t_scalar = t;
		sink += acc;
	}

	/* ---- Option A: pre-transposed blocked codes (built once, untimed) -
	 * blk[block][coord][vec] = code, 1 byte each. */
	long		nblocks = N / BLOCK;
	uint8_t    *blocked = malloc((size_t) N * dc);	/* nblocks * dc * BLOCK */

	for (long b = 0; b < nblocks; b++)
	{
		uint8_t    *dst = blocked + (size_t) b * dc * BLOCK;

		for (int v = 0; v < BLOCK; v++)
		{
			const uint8_t *row = codes + (size_t) (b * BLOCK + v) * bytesPerVec;

			for (int i = 0; i < dc; i++)
				dst[(size_t) i * BLOCK + v] = unpack4(row, i);
		}
	}

	/* NEON fast-scan over one block's transposed codes (32 vectors). */
#define FLUSH 128
	/* scan: blk = dc*BLOCK bytes (coord-major), out[32] uint32 sums. */
	#define RUN_FASTSCAN(blk, out)                                            \
	do {                                                                      \
		uint16x8_t a16[4] = { vdupq_n_u16(0), vdupq_n_u16(0),                 \
							  vdupq_n_u16(0), vdupq_n_u16(0) };                 \
		uint32x4_t a32[8];                                                    \
		for (int _k = 0; _k < 8; _k++) a32[_k] = vdupq_n_u32(0);              \
		for (int _i = 0; _i < dc; _i++)                                       \
		{                                                                     \
			uint8x16_t t = vld1q_u8(lut8 + (size_t) _i * NLEVELS);           \
			const uint8_t *cp = (blk) + (size_t) _i * BLOCK;                  \
			uint8x16_t i0 = vld1q_u8(cp);                                     \
			uint8x16_t i1 = vld1q_u8(cp + 16);                               \
			uint8x16_t r0 = vqtbl1q_u8(t, i0);                              \
			uint8x16_t r1 = vqtbl1q_u8(t, i1);                              \
			a16[0] = vaddw_u8(a16[0], vget_low_u8(r0));                      \
			a16[1] = vaddw_u8(a16[1], vget_high_u8(r0));                     \
			a16[2] = vaddw_u8(a16[2], vget_low_u8(r1));                      \
			a16[3] = vaddw_u8(a16[3], vget_high_u8(r1));                     \
			if ((_i & (FLUSH - 1)) == (FLUSH - 1))                           \
			{                                                                 \
				a32[0] = vaddw_u16(a32[0], vget_low_u16(a16[0]));            \
				a32[1] = vaddw_u16(a32[1], vget_high_u16(a16[0]));          \
				a32[2] = vaddw_u16(a32[2], vget_low_u16(a16[1]));           \
				a32[3] = vaddw_u16(a32[3], vget_high_u16(a16[1]));         \
				a32[4] = vaddw_u16(a32[4], vget_low_u16(a16[2]));           \
				a32[5] = vaddw_u16(a32[5], vget_high_u16(a16[2]));         \
				a32[6] = vaddw_u16(a32[6], vget_low_u16(a16[3]));          \
				a32[7] = vaddw_u16(a32[7], vget_high_u16(a16[3]));        \
				for (int _k = 0; _k < 4; _k++) a16[_k] = vdupq_n_u16(0);     \
			}                                                                 \
		}                                                                     \
		a32[0] = vaddw_u16(a32[0], vget_low_u16(a16[0]));                    \
		a32[1] = vaddw_u16(a32[1], vget_high_u16(a16[0]));                  \
		a32[2] = vaddw_u16(a32[2], vget_low_u16(a16[1]));                   \
		a32[3] = vaddw_u16(a32[3], vget_high_u16(a16[1]));                 \
		a32[4] = vaddw_u16(a32[4], vget_low_u16(a16[2]));                   \
		a32[5] = vaddw_u16(a32[5], vget_high_u16(a16[2]));                 \
		a32[6] = vaddw_u16(a32[6], vget_low_u16(a16[3]));                  \
		a32[7] = vaddw_u16(a32[7], vget_high_u16(a16[3]));                \
		for (int _k = 0; _k < 8; _k++) vst1q_u32((out) + _k * 4, a32[_k]);   \
	} while (0)

	double		t_A = 1e30;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;
		uint32_t	out[BLOCK];

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blocked + (size_t) b * dc * BLOCK;

			RUN_FASTSCAN(blk, out);
			for (int v = 0; v < BLOCK; v++)
				acc += out[v];
		}
		double		t = now_ms() - t0;

		if (t < t_A)
			t_A = t;
		sink += (double) acc;
	}

	/* ---- Option B: row-major on disk; transpose a block into scratch,
	 * then fast-scan it. */
	double		t_B = 1e30;

	for (int r = 0; r < reps; r++)
	{
		uint8_t    *scratch = malloc((size_t) dc * BLOCK);
		double		t0 = now_ms();
		uint64_t	acc = 0;
		uint32_t	out[BLOCK];

		for (long b = 0; b < nblocks; b++)
		{
			/* transpose+unpack this block from the packed row-major codes */
			for (int v = 0; v < BLOCK; v++)
			{
				const uint8_t *row = codes + (size_t) (b * BLOCK + v) * bytesPerVec;

				for (int i = 0; i < dc; i++)
					scratch[(size_t) i * BLOCK + v] = unpack4(row, i);
			}
			RUN_FASTSCAN(scratch, out);
			for (int v = 0; v < BLOCK; v++)
				acc += out[v];
		}
		double		t = now_ms() - t0;

		if (t < t_B)
			t_B = t;
		sink += (double) acc;
		free(scratch);
	}

	printf("\n  scalar (current)        : %8.1f ms\n", t_scalar);
	printf("  B = row-major+transpose : %8.1f ms   (%.1fx vs scalar)\n",
		   t_B, t_scalar / t_B);
	printf("  A = blocked on-disk     : %8.1f ms   (%.1fx vs scalar)\n",
		   t_A, t_scalar / t_A);
	printf("\n  transpose overhead (B/A): %.2fx   -> A would buy %.0f%% over B\n",
		   t_B / t_A, (t_B / t_A - 1.0) * 100.0);
	printf("  [sink=%g]\n", (double) sink);
	return 0;
}
