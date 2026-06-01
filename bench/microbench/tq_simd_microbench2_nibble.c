/*
 * tq_simd_microbench2.c -- nibble-packed fast-scan variant.
 *
 * Confirms the *real* Option A: codes stored blocked AND 4-bit nibble-packed
 * (index size preserved, same bytes as today's row-major 4-bit), with the
 * nibble split done inside the NEON kernel (FAISS fast-scan trick).  arm64 only.
 *
 * Four paths, all scoring N stored vectors (one query's score step):
 *   scalar    : faithful TqScoreEntryDefault (unpack4 + float-LUT gather).
 *   B         : row-major 4-bit -> per-query transpose to scratch -> kernel.
 *   A-byte    : blocked, 1 byte/code (2x size) -> kernel (prev microbench's A).
 *   A-nibble  : blocked, 4-bit packed (SAME size as today) -> kernel splits
 *               nibbles.  <-- the shippable Option A.
 *
 * Blocked nibble layout: per 32-vector block, per coordinate, 16 bytes where
 * byte j = code(vec j) | (code(vec j+16) << 4)  -> low nibbles = vectors 0..15,
 * high nibbles = vectors 16..31.  dc*16 bytes/block = 0.5 byte/code = today's.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK 32
#define NLEVELS 16			/* bits = 4 */
#define FLUSH 128

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static inline uint8_t
unpack4(const uint8_t *codes, int i)
{
	uint8_t		b = codes[i >> 1];

	return (i & 1) ? (b >> 4) : (b & 0x0F);
}

/* accumulate two lookup result regs (vecs 0..15, 16..31) into 4 uint16 accs */
#define ACC4(r0, r1)                                          \
	do {                                                      \
		a16[0] = vaddw_u8(a16[0], vget_low_u8(r0));           \
		a16[1] = vaddw_u8(a16[1], vget_high_u8(r0));          \
		a16[2] = vaddw_u8(a16[2], vget_low_u8(r1));           \
		a16[3] = vaddw_u8(a16[3], vget_high_u8(r1));          \
	} while (0)

#define FLUSH16TO32()                                                 \
	do {                                                              \
		a32[0] = vaddw_u16(a32[0], vget_low_u16(a16[0]));            \
		a32[1] = vaddw_u16(a32[1], vget_high_u16(a16[0]));          \
		a32[2] = vaddw_u16(a32[2], vget_low_u16(a16[1]));           \
		a32[3] = vaddw_u16(a32[3], vget_high_u16(a16[1]));         \
		a32[4] = vaddw_u16(a32[4], vget_low_u16(a16[2]));           \
		a32[5] = vaddw_u16(a32[5], vget_high_u16(a16[2]));         \
		a32[6] = vaddw_u16(a32[6], vget_low_u16(a16[3]));          \
		a32[7] = vaddw_u16(a32[7], vget_high_u16(a16[3]));        \
	} while (0)

int
main(int argc, char **argv)
{
	int			dc = argc > 1 ? atoi(argv[1]) : 2048;
	long		N = argc > 2 ? atol(argv[2]) : (1L << 20);
	int			reps = argc > 3 ? atoi(argv[3]) : 5;
	int			bytesPerVec = (dc + 1) / 2;

	N = (N / BLOCK) * BLOCK;
	long		nblocks = N / BLOCK;
	printf("dc=%d  N=%ld  reps=%d\n", dc, N, reps);

	uint8_t    *codes = malloc((size_t) N * bytesPerVec);
	float	   *lutf = malloc(sizeof(float) * dc * NLEVELS);
	uint8_t    *lut8 = malloc((size_t) dc * NLEVELS);

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
	for (int i = 0; i < dc * NLEVELS; i++)
	{
		int			q = (int) lroundf((lutf[i] + lutmax) / (2 * lutmax) * 255.0f);

		lut8[i] = (uint8_t) (q < 0 ? 0 : q > 255 ? 255 : q);
	}

	volatile double sink = 0.0;

	/* ---- scalar ---- */
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

	/* ---- A-byte blocked (1 byte/code), built once ---- */
	uint8_t    *blockedB = malloc((size_t) N * dc);	/* dc*BLOCK per block */

	for (long b = 0; b < nblocks; b++)
	{
		uint8_t    *dst = blockedB + (size_t) b * dc * BLOCK;

		for (int v = 0; v < BLOCK; v++)
		{
			const uint8_t *row = codes + (size_t) (b * BLOCK + v) * bytesPerVec;

			for (int i = 0; i < dc; i++)
				dst[(size_t) i * BLOCK + v] = unpack4(row, i);
		}
	}

	/* ---- A-nibble blocked (4-bit packed), built once ---- */
	uint8_t    *blockedN = malloc((size_t) nblocks * dc * 16);	/* dc*16 per block */

	for (long b = 0; b < nblocks; b++)
	{
		uint8_t    *dst = blockedN + (size_t) b * dc * 16;

		for (int i = 0; i < dc; i++)
		{
			uint8_t    *cell = dst + (size_t) i * 16;

			for (int j = 0; j < 16; j++)
			{
				uint8_t		lo = unpack4(codes + (size_t) (b * BLOCK + j) * bytesPerVec, i);
				uint8_t		hi = unpack4(codes + (size_t) (b * BLOCK + j + 16) * bytesPerVec, i);

				cell[j] = lo | (hi << 4);
			}
		}
	}

	uint8x16_t	mask = vdupq_n_u8(0x0F);

	/* ---- A-byte kernel ---- */
	double		t_Abyte = 1e30;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;
		uint32_t	out[BLOCK];

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blockedB + (size_t) b * dc * BLOCK;
			uint16x8_t	a16[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
			uint32x4_t	a32[8];

			for (int k = 0; k < 8; k++)
				a32[k] = vdupq_n_u32(0);
			for (int i = 0; i < dc; i++)
			{
				uint8x16_t	t = vld1q_u8(lut8 + (size_t) i * NLEVELS);
				const uint8_t *cp = blk + (size_t) i * BLOCK;
				uint8x16_t	r0 = vqtbl1q_u8(t, vld1q_u8(cp));
				uint8x16_t	r1 = vqtbl1q_u8(t, vld1q_u8(cp + 16));

				ACC4(r0, r1);
				if ((i & (FLUSH - 1)) == (FLUSH - 1))
				{
					FLUSH16TO32();
					for (int k = 0; k < 4; k++)
						a16[k] = vdupq_n_u16(0);
				}
			}
			FLUSH16TO32();
			for (int k = 0; k < 8; k++)
				vst1q_u32(out + k * 4, a32[k]);
			for (int v = 0; v < BLOCK; v++)
				acc += out[v];
		}
		double		t = now_ms() - t0;

		if (t < t_Abyte)
			t_Abyte = t;
		sink += (double) acc;
	}

	/* ---- A-nibble kernel (the shippable Option A) ---- */
	double		t_Anib = 1e30;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;
		uint32_t	out[BLOCK];

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blockedN + (size_t) b * dc * 16;
			uint16x8_t	a16[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
			uint32x4_t	a32[8];

			for (int k = 0; k < 8; k++)
				a32[k] = vdupq_n_u32(0);
			for (int i = 0; i < dc; i++)
			{
				uint8x16_t	t = vld1q_u8(lut8 + (size_t) i * NLEVELS);
				uint8x16_t	c = vld1q_u8(blk + (size_t) i * 16);
				uint8x16_t	lo = vandq_u8(c, mask);
				uint8x16_t	hi = vandq_u8(vshrq_n_u8(c, 4), mask);
				uint8x16_t	r0 = vqtbl1q_u8(t, lo);
				uint8x16_t	r1 = vqtbl1q_u8(t, hi);

				ACC4(r0, r1);
				if ((i & (FLUSH - 1)) == (FLUSH - 1))
				{
					FLUSH16TO32();
					for (int k = 0; k < 4; k++)
						a16[k] = vdupq_n_u16(0);
				}
			}
			FLUSH16TO32();
			for (int k = 0; k < 8; k++)
				vst1q_u32(out + k * 4, a32[k]);
			for (int v = 0; v < BLOCK; v++)
				acc += out[v];
		}
		double		t = now_ms() - t0;

		if (t < t_Anib)
			t_Anib = t;
		sink += (double) acc;
	}

	double		szRow = (double) N * bytesPerVec / (1024 * 1024);
	double		szByte = (double) N * dc / (1024 * 1024);

	printf("\n  index code bytes: row-major/nibble = %.0f MB,  byte-per-code = %.0f MB\n\n",
		   szRow, szByte);
	printf("  scalar (current)         : %8.1f ms\n", t_scalar);
	printf("  B (transpose-on-scan)    : (see microbench 1; ~1.5-2.1x)\n");
	printf("  A-byte  (2x size)        : %8.1f ms   (%.1fx vs scalar)\n",
		   t_Abyte, t_scalar / t_Abyte);
	printf("  A-nibble (same size) **  : %8.1f ms   (%.1fx vs scalar)\n",
		   t_Anib, t_scalar / t_Anib);
	printf("\n  nibble-split cost (Anib/Abyte): %.2fx\n", t_Anib / t_Abyte);
	printf("  [sink=%g]\n", (double) sink);
	return 0;
}
