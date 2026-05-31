-- tqflat build tests: M2.1 (meta/side pages + loader), M2.2 (heap scan + data pages),
-- M2.4 (QJL residual encode side: tq_prod meta/side pages + signs region)
-- Scan is not implemented yet (M3), so we exercise build + meta only.

CREATE TABLE tqbuild (id serial, v vector(8));
INSERT INTO tqbuild (v)
	SELECT ('[' || array_to_string(array(SELECT (i * 7 + g) % 13 - 6 FROM generate_series(1, 8) i), ',') || ']')::vector
	FROM generate_series(1, 500) g;

-- bits = 2 with tq_prod = true (exercises the QJL stage; tq_prod now defaults OFF)
CREATE INDEX tqbuild_idx2 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 2, tq_prod = true);
SELECT tqflat_test_meta('tqbuild_idx2'::regclass);
SELECT pg_relation_size('tqbuild_idx2') > 0 AS idx2_nonempty;

-- bits = 4 grows the meta bits field
CREATE INDEX tqbuild_idx4 ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 4);
SELECT tqflat_test_meta('tqbuild_idx4'::regclass);

-- default (no options): bits = 2, tq_prod = off (the new default), fast_rotation = on
CREATE INDEX tqbuild_idxdef ON tqbuild USING tqflat (v vector_l2_ops);
SELECT tqflat_test_meta('tqbuild_idxdef'::regclass);

-- M2.4: tq_prod = false disables the QJL stage (tqProd column = 0)
CREATE INDEX tqbuild_idxnoprod ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 2, tq_prod = false);
SELECT tqflat_test_meta('tqbuild_idxnoprod'::regclass);

-- M2.4: tq_prod = true index carries the per-entry signs region (dim/8 B each)
-- plus a QJL side page, so it is strictly larger than the otherwise-identical
-- tq_prod = false index.  Pinned to dense rotation: under fast_rotation the QJL
-- side page is absent and, at dim = 8 / bits = 2, the 1-byte signs region is
-- absorbed by MAXALIGN, so the two indexes are byte-identical in size.
CREATE INDEX tqbuild_idx2_dense ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 2, tq_prod = true, fast_rotation = false);
CREATE INDEX tqbuild_idxnoprod_dense ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 2, tq_prod = false, fast_rotation = false);
SELECT pg_relation_size('tqbuild_idx2_dense') > pg_relation_size('tqbuild_idxnoprod_dense') AS prod_is_larger;
DROP INDEX tqbuild_idx2_dense;
DROP INDEX tqbuild_idxnoprod_dense;

-- M2.4: QJL matrix round-trips through TqLoadModel.
-- Returns {tqProd, qjlScale, qjl[0], max|reload - recompute|}.
-- tq_prod = true: tqProd = 1, qjlScale > 0, reload matches recompute exactly (maxdiff = 0).
WITH q AS (SELECT tqflat_test_qjl('tqbuild_idx2'::regclass) AS r)
SELECT (r)[1] AS tqprod, (r)[2] > 0 AS scale_positive, (r)[4] AS reload_maxdiff FROM q;
-- tq_prod = false: no QJL matrix; all-zero descriptor.
SELECT tqflat_test_qjl('tqbuild_idxnoprod'::regclass);

-- bad bits must error
CREATE INDEX tqbuild_idxbad ON tqbuild USING tqflat (v vector_l2_ops) WITH (bits = 5);

-- M2.3: single-row insert (tqinsert) -- build then insert, nVectors must increase.
-- A single INSERT updates ALL indexes on the table simultaneously, so one batch of
-- 100 rows increments nVectors on every index from 500 to 600.
INSERT INTO tqbuild (v)
	SELECT ('[' || array_to_string(array(SELECT (i * 3 + g) % 7 - 3 FROM generate_series(1, 8) i), ',') || ']')::vector
	FROM generate_series(1, 100) g;

-- tq_prod = true (explicit): nVectors 500 -> 600
SELECT tqflat_test_meta('tqbuild_idx2'::regclass);    -- {8,2,0,1,600,0,8}
-- tq_prod = false: nVectors 500 -> 600
SELECT tqflat_test_meta('tqbuild_idxnoprod'::regclass);  -- {8,2,0,0,600,0,8}
-- bits = 4, tq_prod = off (default): nVectors 500 -> 600
SELECT tqflat_test_meta('tqbuild_idx4'::regclass);    -- {8,4,0,0,600,0,8}

-- index size must be non-zero after inserts
SELECT pg_relation_size('tqbuild_idx2') > 0 AS idx2_nonempty_after_insert;

DROP TABLE tqbuild;
