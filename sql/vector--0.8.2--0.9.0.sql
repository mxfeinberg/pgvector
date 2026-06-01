-- tqflat access method

CREATE FUNCTION tqhandler(internal) RETURNS index_am_handler
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD tqflat TYPE INDEX HANDLER tqhandler;

COMMENT ON ACCESS METHOD tqflat IS 'tqflat (TurboQuant flat) index access method';

CREATE FUNCTION tqflat_l2_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION tqflat_ip_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION tqflat_cosine_support(internal) RETURNS internal
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE OPERATOR CLASS vector_l2_ops
	FOR TYPE vector USING tqflat AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_l2_squared_distance(vector, vector),
	FUNCTION 5 tqflat_l2_support(internal);

CREATE OPERATOR CLASS vector_ip_ops
	FOR TYPE vector USING tqflat AS
	OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 5 tqflat_ip_support(internal);

CREATE OPERATOR CLASS vector_cosine_ops
	FOR TYPE vector USING tqflat AS
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 2 vector_norm(vector),
	FUNCTION 5 tqflat_cosine_support(internal);

-- tqflat internal test wrappers (prototype/test-only; drop before upstreaming)
CREATE FUNCTION tqflat_test_codebook(int, int) RETURNS float8[]
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_roundtrip(vector, int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_rotation_orthogonality(int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_meta(regclass) RETURNS int[]
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT;
CREATE FUNCTION tqflat_test_qjl(regclass) RETURNS float8[]
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT;
CREATE FUNCTION tqflat_test_ip_estimate(vector, vector, int, bool) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_codebook_mse(int, int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_rotation_coord_stats(int) RETURNS float8[]
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_pack_roundtrip(int, int) RETURNS int
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_qjl_estimate(vector, vector, int, int, bool) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_fwht_involution(int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_rht_norm(int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_rht_coord_stats(int) RETURNS float8[]
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_ip_accuracy(vector[], vector[], int, bool, bool) RETURNS float8[]
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_transpose_roundtrip(int, int) RETURNS int
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_lut8_recovery(int) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_score_block(int) RETURNS int
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_score_block_consistency(int) RETURNS int
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;
CREATE FUNCTION tqflat_test_active_kernel() RETURNS text
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE;
