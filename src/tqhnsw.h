#ifndef TQHNSW_H
#define TQHNSW_H

#include "postgres.h"

#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/reloptions.h"
#include "fmgr.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "utils/hsearch.h"
#include "utils/relcache.h"
#include "utils/relptr.h"
#include "tq.h"						/* TqModel, TqEntry, TqMetric, TqTypeInfo, TqPackCode */
#include "vector.h"

/*
 * base==NULL -> absolute pointer; relptr branch dormant until parallel build (#3).
 * Two intentional divergences from hnsw.h's HnswPtr: (1) base is cast to (char *)
 * so relptr's byte-offset arithmetic is correct regardless of the caller's base
 * type; (2) there is no extra TqhnswNeighborsPtr indirection layer (see
 * TqhnswGetNeighbors) -- the relptr migration in #3 must reconcile that.
 */
#define TqhnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

typedef struct TqhnswElementData TqhnswElementData;
typedef struct TqhnswNeighborArray TqhnswNeighborArray;

TqhnswPtrDeclare(TqhnswElementData, TqhnswElementRelptr, TqhnswElementPtr);
TqhnswPtrDeclare(TqhnswNeighborArray, TqhnswNeighborArrayRelptr, TqhnswNeighborArrayPtr);

#define TqhnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access((char *)(base), (hp).relptr))
#define TqhnswPtrStore(base, hp, v) ((base) == NULL ? (void) ((hp).ptr = (v)) : (void) relptr_store((char *)(base), (hp).relptr, v))
#define TqhnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))

typedef struct TqhnswCandidate
{
	TqhnswElementPtr element;
	ItemPointerData tid;		/* disk-path: TID before element is materialized */
	double		distance;
}			TqhnswCandidate;

struct TqhnswNeighborArray
{
	int			count;
	TqhnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

#define TQHNSW_NEIGHBOR_ARRAY_SIZE(lm) \
	(offsetof(TqhnswNeighborArray, items) + sizeof(TqhnswCandidate) * (lm))

/* Limits / defaults */
#define TQHNSW_DEFAULT_M 16
#define TQHNSW_MIN_M 2
#define TQHNSW_MAX_M 100
#define TQHNSW_DEFAULT_EF_CONSTRUCTION 64
#define TQHNSW_MIN_EF_CONSTRUCTION 4
#define TQHNSW_MAX_EF_CONSTRUCTION 1000
#define TQHNSW_DEFAULT_EF_SEARCH 40
#define TQHNSW_MIN_EF_SEARCH 1
#define TQHNSW_MAX_EF_SEARCH 1000
#define TQHNSW_DEFAULT_RERANK 100
#define TQHNSW_MAX_RERANK 1000000

/*
 * Page-lock id used to serialize inserts against the (future) graph-mutating
 * vacuum, mirroring HNSW's HNSW_UPDATE_LOCK.  Inserts hold it ShareLock for the
 * duration (ExclusiveLock when they may change the entry point) so vacuum can
 * drain in-flight inserts before repairing the graph.
 */
#define TQHNSW_UPDATE_LOCK 0

/* Page layout */
#define TQHNSW_METAPAGE_BLKNO 0
#define TQHNSW_MAGIC_NUMBER 0x71685451	/* distinct from tqflat 0x71665451 / tqivf 0x71715451 */
#define TQHNSW_VERSION 1
#define TQHNSW_PAGE_ID 0xFF94			/* distinct from tqflat 0xFF92 / tqivf 0xFF93 */

/* Tuple type tags */
#define TQHNSW_ELEMENT_TUPLE_TYPE 1
#define TQHNSW_NEIGHBOR_TUPLE_TYPE 2

/* Support function numbers (opclass FUNCTION slots) */
#define TQHNSW_DISTANCE_PROC 1		/* exact distance, used by rerank */
#define TQHNSW_NORM_PROC 2			/* l2 norm (cosine/ip parity; mirrors hnsw) */
#define TQHNSW_TYPE_INFO_PROC 3		/* tqhnsw_*_support -> TqTypeInfo.metric */

/* Per-level fanout: level 0 gets 2*m, upper levels m (HNSW convention). */
#define TqhnswGetLayerM(m, lc) ((lc) == 0 ? (m) * 2 : (m))
/* Level multiplier (HNSW paper); same as HnswGetMl. */
#define TqhnswGetMl(m) (1 / log(m))
/* Max level bounded by neighbor-tuple page capacity (mirror HnswGetMaxLevel). */
#define TqhnswGetMaxLevel(m) \
	Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(TqhnswPageOpaqueData)) \
		  - offsetof(TqhnswNeighborTupleData, indextids) - sizeof(ItemIdData)) \
		 / (sizeof(ItemPointerData)) / (m)) - 2, 255)

/* GUCs */
extern int	tqhnsw_ef_search;
extern int	tqhnsw_rerank;
extern bool tqhnsw_force_scalar;

/* reloptions */
typedef struct TqhnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* graph connectivity */
	int			efConstruction; /* build-time candidate list size */
	bool		fastRotation;	/* structured randomized Hadamard rotation */
}			TqhnswOptions;

/* Standard page opaque (mirrors HnswPageOpaqueData / TqPageOpaqueData). */
typedef struct TqhnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* TQHNSW_PAGE_ID */
}			TqhnswPageOpaqueData;

typedef TqhnswPageOpaqueData * TqhnswPageOpaque;

#define TqhnswPageGetOpaque(page) ((TqhnswPageOpaque) PageGetSpecialPointer(page))

/*
 * Meta page. Fuses HNSW's graph header (m, efConstruction, entry point) with the
 * TQ model header (so the TqLoadModel logic is reused). Codebook (and, dense mode
 * only, rotation matrix) follow on side pages.
 */
typedef struct TqhnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint16		dimensions;
	uint16		bits;			/* always 4 */
	uint16		metric;			/* TqMetric */
	uint16		fastRotation;	/* bool */
	uint16		dimPadded;		/* next_pow2(dim) in fast mode, else dim */
	uint16		m;
	uint16		efConstruction;
	uint32		nLevels;		/* 1 << bits */
	uint32		nVectors;		/* live nodes written */
	uint64		rotSeed;
	BlockNumber codebookStart;	/* codebook side-page chain */
	BlockNumber rotationStart;	/* dense mode only (Invalid in fast mode) */
	BlockNumber entryBlkno;		/* graph entry point element tuple */
	OffsetNumber entryOffno;
	int16		entryLevel;		/* -1 when empty */
	BlockNumber insertPage;		/* reserved; unused in MVP */
}			TqhnswMetaPageData;

typedef TqhnswMetaPageData * TqhnswMetaPage;

#define TqhnswPageGetMeta(page) ((TqhnswMetaPage) PageGetContents(page))

/*
 * On-disk element tuple. Mirrors HnswElementTupleData but replaces the inline
 * Vector with a TQ tail (norm + scale + packed codes). neighbortid points at the
 * element's neighbor tuple. Codes length = TQ_CODES_BYTES(dimCodes, 4).
 */
typedef struct TqhnswElementTupleData
{
	uint8		type;			/* TQHNSW_ELEMENT_TUPLE_TYPE */
	uint8		level;
	uint8		deleted;		/* reserved; MVP never sets (heap visibility handles deletes) */
	uint8		version;
	ItemPointerData heaptid;	/* the indexed heap row (single; no dup-merge in MVP) */
	ItemPointerData neighbortid;
	float		norm;			/* stripped L2 length */
	float		scale;			/* renormalization correction */
	char		codes[FLEXIBLE_ARRAY_MEMBER];
}			TqhnswElementTupleData;

typedef TqhnswElementTupleData * TqhnswElementTuple;

#define TQHNSW_ELEMENT_TUPLE_SIZE(_codesBytes) \
	MAXALIGN(offsetof(TqhnswElementTupleData, codes) + (_codesBytes))

/*
 * On-disk neighbor tuple. Byte-identical scheme to HnswNeighborTupleData:
 * (level+2)*m item pointers, layer lc's slice starting at (level-lc)*m, level 0
 * doubled. Quantization does not touch graph edges, so this is reused verbatim.
 */
typedef struct TqhnswNeighborTupleData
{
	uint8		type;			/* TQHNSW_NEIGHBOR_TUPLE_TYPE */
	uint8		version;
	uint16		count;			/* (level+2)*m */
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			TqhnswNeighborTupleData;

typedef TqhnswNeighborTupleData * TqhnswNeighborTuple;

#define TQHNSW_NEIGHBOR_TUPLE_SIZE(level, m) \
	MAXALIGN(offsetof(TqhnswNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

/* ---- tqhnsw.c ---- */
extern void TqhnswInit(void);
extern TqModel *TqhnswLoadModel(Relation index, MemoryContext ctx);
extern TqModel *TqhnswGetCachedModel(Relation index);
extern void TqhnswGetMetaInfo(Relation index, int *dim, TqMetric *metric, int *m,
							  BlockNumber *entryBlkno, OffsetNumber *entryOffno,
							  int *entryLevel);

/*
 * In-memory graph node used during the serial build (absolute pointers; no
 * relptr, no locks).  rhat is the reconstructed rotated vector used for the
 * build-time distance (unit-normalized for cosine).  blkno/offno/neighborPage/
 * neighborOffno are assigned during the flush-to-disk pass.
 */
typedef struct TqhnswElementData TqhnswElement;	/* keep the existing name */

struct TqhnswElementData
{
	ItemPointerData heaptid;
	uint8		level;
	float	   *rhat;			/* reconstructed rotated vector, dimCodes floats */
	char	   *codes;			/* packed codes (flushed to disk) */
	float		norm;
	float		scale;
	uint32		visitedGeneration;	/* per-search visited stamp */
	TqhnswNeighborArray **neighbors; /* [level+1]; each sized TqhnswGetLayerM(m,lc) */
	/* assigned during flush */
	BlockNumber blkno;
	OffsetNumber offno;
	BlockNumber neighborPage;
	OffsetNumber neighborOffno;
	TqhnswElement *next;		/* singly-linked build list */
};

static inline TqhnswNeighborArray *
TqhnswGetNeighbors(char *base, TqhnswElement *element, int lc)
{
	(void) base;				/* in-memory build: neighbors are absolute */
	Assert(lc <= element->level);	/* neighbors[] is sized [0..level] */
	return element->neighbors[lc];
}

/* ---- tqhnswutils.c ---- */
/* Reconstruct the rotated full-magnitude vector from an entry's codes (no inverse
 * rotation: distances are computed in rotated space, which is orthonormal). */
extern void TqhnswReconstruct(const TqModel *model, const char *codes,
							  float norm, float scale, float *rhat /* dimCodes */);

/* Build-time distance on reconstructed rotated vectors (smaller = nearer). */
extern double TqhnswBuildDistance(const float *a, const float *b, int dc,
								  TqMetric metric);

/* Add element as a neighbor of target at layer lc, pruning target's neighbor
 * list back to lm via SelectNeighbors if it overflows (HnswUpdateConnection). */
extern void TqhnswUpdateConnection(char *base, TqhnswElement *target,
								   TqhnswElement *element, double distance,
								   int lm, int lc, int dc, TqMetric metric);

/* Insert element into the graph (paper Alg 1).  build path: base=NULL, index=NULL. */
extern void TqhnswInsertElement(char *base, Relation index, const TqModel *model,
								HTAB *cache, MemoryContext ctx,
								TqhnswElement *element, TqhnswElement *entryPoint,
								int m, int efConstruction, int dc, TqMetric metric);

/*
 * Per-insert element cache entry: TID -> materialized graph node. The key
 * (tid) MUST be the first field (dynahash stores the key at offset 0).
 */
typedef struct TqhnswElementCacheEntry
{
	ItemPointerData tid;		/* hash key */
	TqhnswElement *element;
} TqhnswElementCacheEntry;

/* Create the TID->element cache used during on-disk insert search. */
extern HTAB *TqhnswCreateElementCache(MemoryContext ctx);

/* Lazily materialize a disk element (codes->rhat) into ctx, cached by TID. */
extern TqhnswElement *TqhnswLoadElement(Relation index, const TqModel *model,
										TqMetric metric, ItemPointer tid,
										MemoryContext ctx, HTAB *cache);

/* Load element's layer-lc neighbor TIDs into a TqhnswNeighborArray.  The
 * items[].element pointers start NULL and are resolved lazily in
 * TqhnswSearchLayer via TqhnswLoadElement (not here). */
extern TqhnswNeighborArray *TqhnswLoadNeighbors(Relation index, const TqModel *model,
												TqMetric metric, TqhnswElement *element,
												int lc, int m, MemoryContext ctx);

/* ---- tqhnswbuild.c ---- */
extern int	TqhnswRandomLevel(double ml, int maxLevel);
extern IndexBuildResult *tqhnswbuild(Relation heap, Relation index, struct IndexInfo *indexInfo);
extern void tqhnswbuildempty(Relation index);
extern bool tqhnswinsert(Relation index, Datum *values, bool *isnull,
						 ItemPointer heap_tid, Relation heap,
						 IndexUniqueCheck checkUnique,
						 bool indexUnchanged, struct IndexInfo *indexInfo);

/* ---- tqhnswscan.c ---- */
extern IndexScanDesc tqhnswbeginscan(Relation index, int nkeys, int norderbys);
extern void tqhnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
extern bool tqhnswgettuple(IndexScanDesc scan, ScanDirection dir);
extern void tqhnswendscan(IndexScanDesc scan);

/* ---- tqhnswvacuum.c ---- */
extern IndexBulkDeleteResult *tqhnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
											   IndexBulkDeleteCallback callback, void *callback_state);
extern IndexBulkDeleteResult *tqhnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);

#endif							/* TQHNSW_H */
