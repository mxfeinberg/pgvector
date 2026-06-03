#!/usr/bin/env bash
# Throwaway-cluster reproduction of the tqhnsw concurrent-insert crash, with
# core dump capture + gdb backtrace. restart_after_crash=off so the postmaster
# surfaces the crash; ulimit -c unlimited so backends dump cores (apport).
set -u
BIN=/usr/lib/postgresql/18/bin
DD=/tmp/tqcrash_dd
SOCK=/tmp/tqcrash_sock
LOG=/tmp/tqcrash_server.log
DIM=16

ulimit -c unlimited
rm -rf "$DD" "$SOCK" "$LOG"
mkdir -p "$SOCK"
"$BIN/initdb" -D "$DD" -U max --no-sync >/dev/null 2>&1

"$BIN/pg_ctl" -D "$DD" -l "$LOG" \
  -o "-c restart_after_crash=off -c unix_socket_directories=$SOCK -c listen_addresses='' -c maintenance_work_mem=512MB" \
  start >/dev/null 2>&1
sleep 2

P() { "$BIN/psql" -h "$SOCK" -d postgres -U max -v ON_ERROR_STOP=1 "$@"; }
P -c "CREATE EXTENSION vector;" >/dev/null 2>&1
P -c "CREATE TABLE tst (i serial, v vector($DIM));" >/dev/null 2>&1
ARR=$(python3 -c "print(','.join(['random()']*$DIM))")
P -c "INSERT INTO tst (v) SELECT ARRAY[$ARR]::vector($DIM) FROM generate_series(1,200);" >/dev/null 2>&1
P -c "CREATE INDEX idx ON tst USING tqhnsw (v vector_l2_ops) WITH (m=16, ef_construction=64);" >/dev/null 2>&1

cat > /tmp/tqcrash_ins.sql <<SQL
INSERT INTO tst (v) VALUES (ARRAY[$ARR]::vector($DIM));
SQL
echo "=== pgbench (10 clients) ==="
"$BIN/pgbench" -h "$SOCK" -d postgres -U max --no-vacuum --client=10 --transactions=200 \
  -f /tmp/tqcrash_ins.sql 2>&1 | grep -i "processed\|aborted" | head -3
"$BIN/pg_ctl" -D "$DD" stop -m immediate >/dev/null 2>&1 || true
sleep 2

echo "=== newest /var/crash postgres entry ==="
CR=$(ls -t /var/crash/*postgres*.crash 2>/dev/null | head -1)
echo "crash file: ${CR:-NONE}"
if [ -n "${CR:-}" ]; then
  rm -rf /tmp/tqunpack; mkdir -p /tmp/tqunpack
  apport-unpack "$CR" /tmp/tqunpack >/dev/null 2>&1 && echo "unpacked"
  CORE=/tmp/tqunpack/CoreDump
  EXE=$(cat /tmp/tqunpack/ExecutablePath 2>/dev/null)
  echo "exe: $EXE  core: $(ls -la $CORE 2>/dev/null)"
  echo "=== GDB BACKTRACE ==="
  gdb -q -batch -ex "set pagination off" -ex "bt" "$EXE" "$CORE" 2>&1 | grep -iE "tqhnsw|hnsw|#[0-9]+ " | head -40
fi
echo "=== server log crash line ==="
grep -Ei "terminated by signal|segmentation" "$LOG" | head -3
