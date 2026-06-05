use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node;
my @queries = ();
my @expected;
my $limit = 10;
my $dim = 32;

# Build a SQL fragment for an inline random vector of $dim dimensions.
# Each coordinate is random()-0.5.  This produces a distinct vector per row
# because random() is called $dim times per row in the outer SELECT.
my $array_sql = join(",", ("random() - 0.5") x $dim);

# ---------------------------------------------------------------------------
# test_recall -- run all queries against the tqivf index, compute recall@K.
#
# $probes   -- value to SET tqivf.probes = N before each query
# $rerank   -- value to SET tqivf.rerank = N before each query
# $min      -- minimum acceptable recall fraction
# $operator -- distance operator (<->, <#>, <=>)
# $label    -- descriptive label for the cmp_ok message
# ---------------------------------------------------------------------------
sub test_recall
{
	my ($probes, $rerank, $min, $operator, $label, $table) = @_;
	$table //= "tst";
	my $correct = 0;
	my $total = 0;

	# Verify the planner actually uses the index.
	my $explain = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET tqivf.probes = $probes;
		SET tqivf.rerank = $rerank;
		EXPLAIN ANALYZE SELECT i FROM $table ORDER BY v $operator '$queries[0]' LIMIT $limit;
	));
	like($explain, qr/Index Scan using idx on $table/, "index scan used for $label");

	for my $i (0 .. $#queries)
	{
		my $actual = $node->safe_psql("postgres", qq(
			SET enable_seqscan = off;
			SET tqivf.probes = $probes;
			SET tqivf.rerank = $rerank;
			SELECT i FROM $table ORDER BY v $operator '$queries[$i]' LIMIT $limit;
		));
		my @actual_ids = split("\n", $actual);
		my %actual_set = map { $_ => 1 } @actual_ids;

		my @expected_ids = split("\n", $expected[$i]);

		foreach (@expected_ids)
		{
			if (exists($actual_set{$_}))
			{
				$correct++;
			}
			$total++;
		}
	}

	cmp_ok($correct / $total, ">=", $min, $label);
}

# ---------------------------------------------------------------------------
# Initialize node
# ---------------------------------------------------------------------------
$node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Use a fixed Perl RNG seed so query vectors are reproducible.
srand(20240101);

# ---------------------------------------------------------------------------
# Create table and insert random vectors.
# PostgreSQL random() is unseeded here, so the row data varies across runs.
# That is intentional: exact ground truth is recomputed via seqscan each run
# (before the index is built), so recall assertions remain valid regardless of
# which data were generated.  Query-vector reproducibility is handled by the
# Perl srand(20240101) call above.
# ---------------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", "CREATE TABLE tst (i int4, v vector($dim));");
$node->safe_psql("postgres",
	"INSERT INTO tst SELECT i, ARRAY[$array_sql]::vector($dim) FROM generate_series(1, 3000) i;");

# ---------------------------------------------------------------------------
# Generate 20 query vectors from the fixed Perl RNG.
# ---------------------------------------------------------------------------
for (1 .. 20)
{
	my @coords = map { rand() - 0.5 } (1 .. $dim);
	push(@queries, "[" . join(",", @coords) . "]");
}

# lists = 55 ~ sqrt(3000); keeps CI cost low while giving meaningful probes knob.
my $lists = 55;

# ===========================================================================
# Test configuration A: L2, probes=lists (full coverage), rerank=200
#   → expect recall ≥ 0.95  (mirrors tqflat bits=4 rerank=200 bound)
# ===========================================================================
{
	my $opclass  = "vector_l2_ops";
	my $operator = "<->";

	# Exact ground truth (seqscan).
	@expected = ();
	foreach my $q (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$q' LIMIT $limit;
		));
		push(@expected, $res);
	}

	$node->safe_psql("postgres",
		"CREATE INDEX idx ON tst USING tqivf (v $opclass) WITH (lists = $lists);");

	# High probes + high rerank: near-exact recovery.
	test_recall($lists, 200, 0.95, $operator, "L2 probes=lists rerank=200");

	# Moderate probes: reduced recall but still reasonable.
	test_recall(10, 100, 0.70, $operator, "L2 probes=10 rerank=100");

	# Low probes: exercises the code path; loose lower bound.
	test_recall(1, 100, 0.20, $operator, "L2 probes=1 rerank=100 (low probes path)");

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# ===========================================================================
# Test configuration B: cosine (<=>), probes=lists, rerank=200
#   → expect recall ≥ 0.90  (mirrors tqflat bits=4 cosine bound)
# ===========================================================================
{
	my $opclass  = "vector_cosine_ops";
	my $operator = "<=>";

	# Exact ground truth.
	@expected = ();
	foreach my $q (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$q' LIMIT $limit;
		));
		push(@expected, $res);
	}

	$node->safe_psql("postgres",
		"CREATE INDEX idx ON tst USING tqivf (v $opclass) WITH (lists = $lists);");

	# High probes + high rerank.
	test_recall($lists, 200, 0.90, $operator, "cosine probes=lists rerank=200");

	# Low probes: just exercise the path.
	test_recall(1, 100, 0.20, $operator, "cosine probes=1 rerank=100 (low probes path)");

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# ===========================================================================
# Test configuration C: inner product (<#>), probes=lists, rerank=200
#   → expect recall ≥ 0.90  (mirrors tqflat bits=4 inner product bound)
# ===========================================================================
{
	my $opclass  = "vector_ip_ops";
	my $operator = "<#>";

	# Exact ground truth.
	@expected = ();
	foreach my $q (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$q' LIMIT $limit;
		));
		push(@expected, $res);
	}

	$node->safe_psql("postgres",
		"CREATE INDEX idx ON tst USING tqivf (v $opclass) WITH (lists = $lists);");

	# High probes + high rerank.
	test_recall($lists, 200, 0.90, $operator, "inner product probes=lists rerank=200");

	# Low probes: just exercise the path.
	test_recall(1, 100, 0.20, $operator, "inner product probes=1 rerank=100 (low probes path)");

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# ===========================================================================
# Test configuration D: L2, rerank=0 (unreranked / quantized-only path)
#   → loose bound (≥0.30), just proves the quantized path executes and returns
#     plausible results; mirrors tqflat's "rerank=0" coverage.
# ===========================================================================
{
	my $opclass  = "vector_l2_ops";
	my $operator = "<->";

	# Exact ground truth.
	@expected = ();
	foreach my $q (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$q' LIMIT $limit;
		));
		push(@expected, $res);
	}

	$node->safe_psql("postgres",
		"CREATE INDEX idx ON tst USING tqivf (v $opclass) WITH (lists = $lists);");

	# Full probes but no rerank: quantized scoring only.
	test_recall($lists, 0, 0.30, $operator, "L2 probes=lists rerank=0 (unreranked path)");

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# ===========================================================================
# Test configuration E: parallel build (L2) must engage workers and match the
# serial recall bound.  Verifies the parallel build path end-to-end.
# ===========================================================================
{
	my $opclass  = "vector_l2_ops";
	my $operator = "<->";

	# Exact ground truth (seqscan).
	@expected = ();
	foreach my $q (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$q' LIMIT $limit;
		));
		push(@expected, $res);
	}

	# Build in parallel; force workers and capture the DEBUG line.
	my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
		SET client_min_messages = DEBUG1;
		SET min_parallel_table_scan_size = 1;
		SET max_parallel_maintenance_workers = 2;
		CREATE INDEX idx ON tst USING tqivf (v $opclass) WITH (lists = $lists);
	));
	is($ret, 0, "parallel tqivf build succeeded: $stderr");
	like($stderr, qr/using \d+ parallel workers/, "parallel tqivf build engaged workers");

	# Same recall bound as the serial L2 config A.
	test_recall($lists, 200, 0.95, $operator, "parallel build L2 probes=lists rerank=200");

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# ===========================================================================
# Test configuration F: halfvec (tqivf #6 Plan 2).  Native halfvec k-means via
# the reused ivfflat halfvec typeinfo (opclass FUNCTION 5).  Same recall bounds
# as the vector L2/cosine configs; fp16 input adds only minor noise (input
# coords are random()-0.5, well within fp16 range).  Separate tst_hv table so
# the vector configs above are untouched.
#
# No tqivf-specific dim-cap test: HALFVEC_MAX_DIM == TQ_MAX_DIM == 16000, so
# the halfvec type itself rejects over-sized columns before tqivf ever sees them.
# ===========================================================================
{
	$node->safe_psql("postgres", "CREATE TABLE tst_hv (i int4, v halfvec($dim));");
	$node->safe_psql("postgres",
		"INSERT INTO tst_hv SELECT i, ARRAY[$array_sql]::vector($dim)::halfvec($dim) FROM generate_series(1, 3000) i;");

	# ---- halfvec L2: probes=lists rerank=200 → recall ≥ 0.95 ----
	{
		my $operator = "<->";

		@expected = ();
		foreach my $q (@queries)
		{
			my $res = $node->safe_psql("postgres", qq(
				SET enable_indexscan = off;
				SELECT i FROM tst_hv ORDER BY v $operator '$q' LIMIT $limit;
			));
			push(@expected, $res);
		}

		$node->safe_psql("postgres",
			"CREATE INDEX idx ON tst_hv USING tqivf (v halfvec_l2_ops) WITH (lists = $lists);");

		test_recall($lists, 200, 0.95, $operator, "halfvec L2 probes=lists rerank=200", "tst_hv");
		test_recall(1, 100, 0.20, $operator, "halfvec L2 probes=1 rerank=100 (low probes path)", "tst_hv");

		$node->safe_psql("postgres", "DROP INDEX idx;");
	}

	# ---- halfvec cosine: probes=lists rerank=200 → recall ≥ 0.90 ----
	{
		my $operator = "<=>";

		@expected = ();
		foreach my $q (@queries)
		{
			my $res = $node->safe_psql("postgres", qq(
				SET enable_indexscan = off;
				SELECT i FROM tst_hv ORDER BY v $operator '$q' LIMIT $limit;
			));
			push(@expected, $res);
		}

		$node->safe_psql("postgres",
			"CREATE INDEX idx ON tst_hv USING tqivf (v halfvec_cosine_ops) WITH (lists = $lists);");

		test_recall($lists, 200, 0.90, $operator, "halfvec cosine probes=lists rerank=200", "tst_hv");
		test_recall(1, 100, 0.20, $operator, "halfvec cosine probes=1 rerank=100 (low probes path)", "tst_hv");

		$node->safe_psql("postgres", "DROP INDEX idx;");
	}

	$node->safe_psql("postgres", "DROP TABLE tst_hv;");
}

done_testing();
