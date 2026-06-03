#include "postgres.h"
#include "tqhnsw.h"

#include "storage/bufmgr.h"
#include "utils/rel.h"

IndexBulkDeleteResult *
tqhnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				 IndexBulkDeleteCallback callback, void *callback_state)
{
	/* MVP: deletes are filtered at rerank via heap visibility; the graph is not
	 * mutated here.  Report no removals.  (Full tombstone/repair is a follow-up.) */
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	return stats;
}

IndexBulkDeleteResult *
tqhnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (stats == NULL)
		return NULL;
	stats->num_pages = RelationGetNumberOfBlocks(info->index);
	return stats;
}
