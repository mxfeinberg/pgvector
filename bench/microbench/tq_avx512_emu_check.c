/*
 * Scalar emulation of TqScoreBlockRangeAvx512's exact intrinsic sequence,
 * checked bit-identical against TqScoreBlockRangeDefault. Validates the
 * lane-mapping / blend / shift / widen logic independent of the real
 * intrinsics (which are syntax/codegen-checked by cross-compile and
 * functionally checked on the x86 host by tqflat_test_score_block_consistency).
 *
 * Runs natively anywhere (no SIMD, no Postgres):
 *   cc -O2 -o /tmp/emu bench/microbench/tq_avx512_emu_check.c && /tmp/emu
 * Expected: "PASS  (0 mismatching trials / 2000)".
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BLOCK 32
#define FLUSH 128

/* ---- Default reference (mirror of TqScoreBlockRangeDefault) ---- */
static void
ref_default(const uint8_t *lut8, const uint8_t *codeRun, int c0, int c1,
			uint16_t acc16[BLOCK], uint32_t acc32[BLOCK], int *sinceFlush)
{
	for (int i = c0; i < c1; i++)
	{
		const uint8_t *cell = codeRun + (size_t) (i - c0) * 16;
		const uint8_t *tbl = lut8 + (size_t) i * 16;

		for (int j = 0; j < 16; j++)
		{
			uint8_t packed = cell[j];

			acc16[j] += tbl[packed & 0x0F];
			acc16[j + 16] += tbl[packed >> 4];
		}
		if (++(*sinceFlush) >= FLUSH)
		{
			for (int j = 0; j < BLOCK; j++)
			{
				acc32[j] += acc16[j];
				acc16[j] = 0;
			}
			*sinceFlush = 0;
		}
	}
}

/* ---- intrinsic emulations (Intel-documented semantics, byte/word exact) ---- */

/* _mm256_broadcastsi128_si256: out[0..15]=in, out[16..31]=in */
static void
emu_broadcast128(uint8_t out[32], const uint8_t in[16])
{
	memcpy(out, in, 16);
	memcpy(out + 16, in, 16);
}

/* _mm256_srli_epi16(x,4): 16 independent u16 lanes (little-endian byte pairs) */
static void
emu_srli_epi16_4(uint8_t out[32], const uint8_t in[32])
{
	for (int w = 0; w < 16; w++)
	{
		uint16_t v = (uint16_t) (in[2 * w] | (in[2 * w + 1] << 8));

		v >>= 4;
		out[2 * w] = (uint8_t) (v & 0xFF);
		out[2 * w + 1] = (uint8_t) (v >> 8);
	}
}

/* _mm256_blend_epi32(a,b,0xF0): dwords 4..7 from b, 0..3 from a */
static void
emu_blend_epi32_F0(uint8_t out[32], const uint8_t a[32], const uint8_t b[32])
{
	memcpy(out, a, 16);			/* dwords 0..3 -> bytes 0..15 */
	memcpy(out + 16, b + 16, 16);	/* dwords 4..7 -> bytes 16..31 */
}

static void
emu_and_0F(uint8_t out[32], const uint8_t in[32])
{
	for (int k = 0; k < 32; k++)
		out[k] = in[k] & 0x0F;
}

/* _mm256_shuffle_epi8: per 128-bit lane, out[j]=(idx[j]&0x80)?0:tbl[lane][idx[j]&0x0F] */
static void
emu_shuffle_epi8(uint8_t out[32], const uint8_t tbl[32], const uint8_t idx[32])
{
	for (int lane = 0; lane < 2; lane++)
	{
		const uint8_t *t = tbl + lane * 16;
		const uint8_t *x = idx + lane * 16;
		uint8_t    *o = out + lane * 16;

		for (int j = 0; j < 16; j++)
			o[j] = (x[j] & 0x80) ? 0 : t[x[j] & 0x0F];
	}
}

/* ---- new kernel, expressed purely via the emulated intrinsics ---- */
static void
emu_new(const uint8_t *lut8, const uint8_t *codeRun, int c0, int c1,
		uint16_t acc16[BLOCK], uint32_t acc32[BLOCK], int *sinceFlush)
{
	for (int i = c0; i < c1; i++)
	{
		const uint8_t *cell = codeRun + (size_t) (i - c0) * 16;
		const uint8_t *tbl16 = lut8 + (size_t) i * 16;
		uint8_t		bc[32],
					sh[32],
					mixed[32],
					idx[32],
					tbl[32],
					r[32];

		emu_broadcast128(bc, cell);
		emu_srli_epi16_4(sh, bc);
		emu_blend_epi32_F0(mixed, bc, sh);
		emu_and_0F(idx, mixed);
		emu_broadcast128(tbl, tbl16);
		emu_shuffle_epi8(r, tbl, idx);

		/* _mm512_cvtepu8_epi16(r) then _mm512_add_epi16 into acc16 */
		for (int k = 0; k < BLOCK; k++)
			acc16[k] = (uint16_t) (acc16[k] + r[k]);

		if (++(*sinceFlush) >= FLUSH)
		{
			/* cvtepu16_epi32 of low/high 256 + add into acc32 */
			for (int k = 0; k < BLOCK; k++)
			{
				acc32[k] += acc16[k];
				acc16[k] = 0;
			}
			*sinceFlush = 0;
		}
	}
}

int
main(void)
{
	srand(12345);
	int			fails = 0;

	for (int trial = 0; trial < 2000; trial++)
	{
		int			dc = 1 + (rand() % 600);	/* varies across the flush boundary */
		uint8_t    *code = malloc((size_t) dc * 16);
		uint8_t    *lut = malloc((size_t) dc * 16);

		for (size_t k = 0; k < (size_t) dc * 16; k++)
		{
			code[k] = (uint8_t) rand();
			lut[k] = (uint8_t) rand();
		}

		uint16_t	a16r[BLOCK] = {0},
					a16n[BLOCK] = {0};
		uint32_t	a32r[BLOCK] = {0},
					a32n[BLOCK] = {0};
		int			sfr = 0,
					sfn = 0;

		/* split into 1..3 ranges to exercise cross-call accumulator carry */
		int			cuts[4];

		cuts[0] = 0;
		cuts[1] = rand() % (dc + 1);
		cuts[2] = rand() % (dc + 1);
		cuts[3] = dc;
		for (int a = 0; a < 3; a++)		/* simple sort */
			for (int b = a + 1; b < 4; b++)
				if (cuts[b] < cuts[a])
				{
					int t = cuts[a];
					cuts[a] = cuts[b];
					cuts[b] = t;
				}

		for (int s = 0; s < 3; s++)
		{
			ref_default(lut, code + (size_t) cuts[s] * 16, cuts[s], cuts[s + 1], a16r, a32r, &sfr);
			emu_new(lut, code + (size_t) cuts[s] * 16, cuts[s], cuts[s + 1], a16n, a32n, &sfn);
		}
		/* finish: fold acc16 -> acc32 (TqBlockAccumFinish) */
		for (int k = 0; k < BLOCK; k++)
		{
			a32r[k] += a16r[k];
			a32n[k] += a16n[k];
		}

		for (int k = 0; k < BLOCK; k++)
			if (a32r[k] != a32n[k])
			{
				fails++;
				if (fails <= 5)
					printf("MISMATCH trial=%d dc=%d lane=%d ref=%u new=%u\n",
						   trial, dc, k, a32r[k], a32n[k]);
				break;
			}
		free(code);
		free(lut);
	}
	printf("%s  (%d mismatching trials / 2000)\n", fails ? "FAIL" : "PASS", fails);
	return fails ? 1 : 0;
}
