/*
 * tq_simd_microbench3_avx512.c -- x86-64 AVX-512 counterpart of microbench2.
 *
 * Same experiment as the NEON microbench2 (scalar vs blocked A-byte vs blocked
 * A-nibble fast-scan), but with the AVX-512F+BW kernel from
 * src/tqfastscan.c::TqScoreBlockRangeAvx512 (128-bit _mm_shuffle_epi8, the
 * shipped v1).  Lets the x86 raw score-step throughput be compared directly to
 * the arm64/NEON numbers.  x86-64 only.
 *
 * Build (on an AVX-512 host):
 *   cc -O3 -ffast-math -mavx512bw -o /tmp/mb3 tq_simd_microbench3_avx512.c -lm
 *   /tmp/mb3 <dc padded-dim> <N vectors> <reps>
 *
 * Blocked nibble layout (identical to microbench2 / the real index): per
 * 32-vector block, per coordinate, 16 bytes where byte j = code(vec j) |
 * (code(vec j+16) << 4) -> low nibbles = lanes 0..15, high nibbles = 16..31.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK 32
#define NLEVELS 16			/* bits = 4 */
#define FLUSH 128			/* coords between uint16 -> uint32 accumulator flush */

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

/*
 * Widen the two 16-byte shuffle results (r0 = lanes 0..15, r1 = lanes 16..31)
 * to u16 and add into the 32 uint16 accumulators, matching tqfastscan.c.
 */
#define ACC4(r0, r1)                                                          \
	do {                                                                      \
		__m128i z = _mm_setzero_si128();                                      \
		__m128i a0 = _mm_loadu_si128((const __m128i *) (acc16 + 0));          \
		__m128i a1 = _mm_loadu_si128((const __m128i *) (acc16 + 8));          \
		__m128i a2 = _mm_loadu_si128((const __m128i *) (acc16 + 16));         \
		__m128i a3 = _mm_loadu_si128((const __m128i *) (acc16 + 24));         \
		a0 = _mm_add_epi16(a0, _mm_unpacklo_epi8(r0, z));                     \
		a1 = _mm_add_epi16(a1, _mm_unpackhi_epi8(r0, z));                     \
		a2 = _mm_add_epi16(a2, _mm_unpacklo_epi8(r1, z));                     \
		a3 = _mm_add_epi16(a3, _mm_unpackhi_epi8(r1, z));                     \
		_mm_storeu_si128((__m128i *) (acc16 + 0), a0);                        \
		_mm_storeu_si128((__m128i *) (acc16 + 8), a1);                        \
		_mm_storeu_si128((__m128i *) (acc16 + 16), a2);                       \
		_mm_storeu_si128((__m128i *) (acc16 + 24), a3);                       \
	} while (0)

#define FLUSH16TO32()                                                         \
	do {                                                                      \
		for (int _k = 0; _k < BLOCK; _k++) {                                  \
			acc32[_k] += acc16[_k];                                           \
			acc16[_k] = 0;                                                    \
		}                                                                     \
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
	printf("dc=%d  N=%ld  reps=%d  (AVX-512F+BW, 128-bit shuffle)\n", dc, N, reps);

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

	/* ---- scalar (faithful TqScoreEntryDefault: unpack4 + float-LUT gather) ---- */
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

	const __m128i mask = _mm_set1_epi8(0x0F);

	/* ---- A-byte kernel ---- */
	double		t_Abyte = 1e30;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blockedB + (size_t) b * dc * BLOCK;
			uint16_t	acc16[BLOCK] = {0};
			uint32_t	acc32[BLOCK] = {0};

			for (int i = 0; i < dc; i++)
			{
				__m128i		tbl = _mm_loadu_si128((const __m128i *) (lut8 + (size_t) i * NLEVELS));
				const uint8_t *cp = blk + (size_t) i * BLOCK;
				__m128i		r0 = _mm_shuffle_epi8(tbl, _mm_loadu_si128((const __m128i *) cp));
				__m128i		r1 = _mm_shuffle_epi8(tbl, _mm_loadu_si128((const __m128i *) (cp + 16)));

				ACC4(r0, r1);
				if ((i & (FLUSH - 1)) == (FLUSH - 1))
					FLUSH16TO32();
			}
			FLUSH16TO32();
			for (int v = 0; v < BLOCK; v++)
				acc += acc32[v];
		}
		double		t = now_ms() - t0;

		if (t < t_Abyte)
			t_Abyte = t;
		sink += (double) acc;
	}

	/* ---- A-nibble kernel: ORIGINAL 128-bit shuffle (shipped v1) ---- */
	double		t_Anib = 1e30;
	uint64_t	chk_128 = 0;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blockedN + (size_t) b * dc * 16;
			uint16_t	acc16[BLOCK] = {0};
			uint32_t	acc32[BLOCK] = {0};

			for (int i = 0; i < dc; i++)
			{
				__m128i		tbl = _mm_loadu_si128((const __m128i *) (lut8 + (size_t) i * NLEVELS));
				__m128i		c = _mm_loadu_si128((const __m128i *) (blk + (size_t) i * 16));
				__m128i		lo = _mm_and_si128(c, mask);
				__m128i		hi = _mm_and_si128(_mm_srli_epi16(c, 4), mask);
				__m128i		r0 = _mm_shuffle_epi8(tbl, lo);
				__m128i		r1 = _mm_shuffle_epi8(tbl, hi);

				ACC4(r0, r1);
				if ((i & (FLUSH - 1)) == (FLUSH - 1))
					FLUSH16TO32();
			}
			FLUSH16TO32();
			for (int v = 0; v < BLOCK; v++)
				acc += acc32[v];
		}
		double		t = now_ms() - t0;

		if (t < t_Anib)
			t_Anib = t;
		chk_128 = acc;
		sink += (double) acc;
	}

	/*
	 * ---- A-nibble kernel: NEW 256-bit-shuffle / 512-bit-widen, register-
	 * resident accumulators (mirrors the rewritten TqScoreBlockRangeAvx512).
	 * One vpshufb (both nibbles) + one vpmovzxbw + one vpaddw per coordinate;
	 * acc16 held in one zmm, acc32 in two, touched only at block start/end.
	 */
	const __m256i mask256 = _mm256_set1_epi8(0x0F);
	double		t_512 = 1e30;
	uint64_t	chk_512 = 0;

	for (int r = 0; r < reps; r++)
	{
		double		t0 = now_ms();
		uint64_t	acc = 0;

		for (long b = 0; b < nblocks; b++)
		{
			const uint8_t *blk = blockedN + (size_t) b * dc * 16;
			__m512i		va16 = _mm512_setzero_si512();
			__m512i		va32lo = _mm512_setzero_si512();
			__m512i		va32hi = _mm512_setzero_si512();
			int			sinceFlush = 0;

			for (int i = 0; i < dc; i++)
			{
				__m128i		cell = _mm_loadu_si128((const __m128i *) (blk + (size_t) i * 16));
				__m128i		tbl128 = _mm_loadu_si128((const __m128i *) (lut8 + (size_t) i * NLEVELS));
				__m256i		bc = _mm256_broadcastsi128_si256(cell);
				__m256i		sh = _mm256_srli_epi16(bc, 4);
				__m256i		idx = _mm256_and_si256(_mm256_blend_epi32(bc, sh, 0xF0), mask256);
				__m256i		tbl = _mm256_broadcastsi128_si256(tbl128);
				__m256i		rr = _mm256_shuffle_epi8(tbl, idx);

				va16 = _mm512_add_epi16(va16, _mm512_cvtepu8_epi16(rr));
				if (++sinceFlush >= FLUSH)
				{
					va32lo = _mm512_add_epi32(va32lo,
											  _mm512_cvtepu16_epi32(_mm512_castsi512_si256(va16)));
					va32hi = _mm512_add_epi32(va32hi,
											  _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(va16, 1)));
					va16 = _mm512_setzero_si512();
					sinceFlush = 0;
				}
			}
			/* finish: fold acc16 -> acc32 */
			va32lo = _mm512_add_epi32(va32lo,
									  _mm512_cvtepu16_epi32(_mm512_castsi512_si256(va16)));
			va32hi = _mm512_add_epi32(va32hi,
									  _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(va16, 1)));

			uint32_t	acc32[BLOCK];

			_mm512_storeu_si512((void *) acc32, va32lo);
			_mm512_storeu_si512((void *) (acc32 + 16), va32hi);
			for (int v = 0; v < BLOCK; v++)
				acc += acc32[v];
		}
		double		t = now_ms() - t0;

		if (t < t_512)
			t_512 = t;
		chk_512 = acc;
		sink += (double) acc;
	}

	double		szRow = (double) N * bytesPerVec / (1024 * 1024);
	double		szByte = (double) N * dc / (1024 * 1024);

	printf("\n  index code bytes: row-major/nibble = %.0f MB,  byte-per-code = %.0f MB\n\n",
		   szRow, szByte);
	printf("  scalar (current)              : %8.1f ms\n", t_scalar);
	printf("  A-byte  (2x size)             : %8.1f ms   (%.1fx vs scalar)\n",
		   t_Abyte, t_scalar / t_Abyte);
	printf("  A-nibble 128-bit (shipped v1) : %8.1f ms   (%.1fx vs scalar)\n",
		   t_Anib, t_scalar / t_Anib);
	printf("  A-nibble 256/512 (new)     ** : %8.1f ms   (%.1fx vs scalar)\n",
		   t_512, t_scalar / t_512);
	printf("\n  new-vs-128bit speedup: %.2fx     checksums: %s (128=%llu 512=%llu)\n",
		   t_Anib / t_512,
		   chk_128 == chk_512 ? "MATCH" : "*** MISMATCH ***",
		   (unsigned long long) chk_128, (unsigned long long) chk_512);
	printf("  [sink=%g]\n", (double) sink);
	return chk_128 == chk_512 ? 0 : 1;
}
