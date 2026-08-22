#!/bin/sh
# One entry point for the whole fuzzing campaign: minify the corpus, run every target,
# report to stdout, keep the corpus for next time.
#
#   ./run.sh                 # all targets, runs until Ctrl-C
#   ./run.sh -t 600          # ten minutes, then stop on its own
#   ./run.sh fz_serverpkt    # just one target
#   ./run.sh -t 300 -j 2 fz_gamemsg fz_console
#
# Every run builds first: build.sh configures build-fuzz if it is not there yet, ninja brings
# the server objects up to date and each harness is relinked, about two seconds a target. So
# no campaign can run against a binary older than the tree it is meant to test.
#
# State lives in one directory (default ~/ddnet-fuzz, override with FUZZ_RUN) and is REUSED
# on the next run:
#   corpus/<target>     the accumulated corpus, minified before every run
#   artifacts/<target>  what libFuzzer saved: crash-, leak-, timeout-, oom-, slow-unit-
#   log/<target>.log    the full libFuzzer output for that target
#   repro/<target>/<a>  a stored full run: the ordered inputs that reproduce artifact <a>
#
# Corpus minification matters more than it sounds: a libFuzzer corpus grows without bound
# (measured here: 230 MB for fz_gamemsg after a few hours), and every entry is re-executed
# at startup, so an unminified corpus makes each run slower than the last. `-merge=1` keeps
# only the inputs needed to preserve coverage.
#
# The summary at the end prints every artifact in full: the reproduce command, the bytes (a
# printable preview plus base64, so it can be recreated anywhere) and what its replay says.
# It treats the artifact kinds differently, because they do not mean the same thing - a
# slow-unit- is not a crash and exits 0 on replay by definition. When an artifact that SHOULD
# fail does not, the summary replays the whole corpus and then the artifact in one process,
# halves the prefix down while it still reproduces, and stores what survives under
# repro/<target>/ as a runnable sequence. That is the reproducer: a single artifact file is
# not one when the bug needed a packet that came before it.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${DDNET_FUZZ_SRC:-$(cd "$HERE/../.." && pwd)}
BUILD=${DDNET_FUZZ_BUILD:-$SRC/build-fuzz}
FZ=$BUILD/fuzzers
RUN=${FUZZ_RUN:-$HOME/ddnet-fuzz}

DURATION=0 # 0 = run until interrupted; -t sets a limit
WORKERS=0  # 0 = auto (share the machine out across the chosen targets)

while getopts "t:j:h" opt; do
	case "$opt" in
	t) DURATION=$OPTARG ;;
	j) WORKERS=$OPTARG ;;
	h)
		awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"
		exit 0
		;;
	*)
		echo "try -h" >&2
		exit 2
		;;
	esac
done
shift $((OPTIND - 1))

# Weighted by how much each target still has to explore. The two whole-server targets reach
# an order of magnitude more code than the byte-level parsers, so they get most of the box.
ALL="fz_serverpkt fz_gamemsg fz_tiles fz_console fz_netserver fz_snap_delta fz_unpack_msg fz_nameban fz_chunk fz_savetee fz_sevensix fz_fuzzystr fz_unpack_packet"
DEFAULT="fz_serverpkt fz_gamemsg fz_tiles fz_console fz_netserver fz_snap_delta fz_unpack_msg fz_nameban fz_chunk fz_savetee fz_sevensix"
TARGETS=${*:-$DEFAULT}

weight_of() {
	case "$1" in
	fz_serverpkt) echo 5 ;;
	fz_gamemsg) echo 4 ;;
	fz_tiles) echo 3 ;;
	fz_console) echo 2 ;;
	*) echo 1 ;;
	esac
}

echo "building harnesses..."
# shellcheck disable=SC2086 # $TARGETS is a list of names
DDNET_FUZZ_SRC=$SRC DDNET_FUZZ_BUILD=$BUILD sh "$HERE/build.sh" $TARGETS

# fz_gamemsg and fz_serverpkt construct a real server: CServer::LoadMap needs data/maps and,
# for the 0.7 half to be covered at all, data/maps7/<map>.map. Without the latter LoadMap
# sets sv_sixup=0 and the whole translation layer goes dark - the fixture warns, but make it
# not happen in the first place.
if [ ! -f "$SRC/data/maps7/coverage.map" ] && [ -f "$SRC/data/maps7/Tutorial.map" ]; then
	cp "$SRC/data/maps7/Tutorial.map" "$SRC/data/maps7/coverage.map"
	echo "note: created data/maps7/coverage.map so the 0.7 paths are covered"
fi

mkdir -p "$RUN/corpus" "$RUN/artifacts" "$RUN/log" "$RUN/work" "$RUN/repro"

# ---------------------------------------------------------------- minify + seed
echo
echo "minifying corpora (keeping only what preserves coverage)"
printf '%-18s %10s %10s\n' target before after
for t in $TARGETS; do
	C=$RUN/corpus/$t
	mkdir -p "$C"
	BEFORE=$(find "$C" -type f | wc -l | tr -d ' ')
	TMP=$RUN/corpus/.min_$t
	rm -rf "$TMP"
	mkdir -p "$TMP"
	# Merge the existing corpus AND the checked-in seeds. Seeds are re-added every run
	# because a merge can otherwise drop one: a seed that adds no NEW edge over the corpus
	# is redundant by libFuzzer's measure, yet it is the thing that documents a reachable
	# shape (a valid save string, the client version, an extended-message uuid).
	SEEDS=$HERE/seeds/$t
	[ -d "$SEEDS" ] || SEEDS=
	# Same sanitizer options and the same artifact directory as the run below. A merge
	# executes every input in the corpus, so it finds crashes, and without these it found
	# them in a way nothing could act on: it wrote crash- files into the SOURCE TREE (its
	# cwd) where the summary never looks, and it left UBSan printing and carrying on. One
	# such crash sat unnoticed in the tree for two days.
	mkdir -p "$RUN/artifacts/$t"
	# shellcheck disable=SC2086 # $SEEDS is one path or nothing at all
	(cd "$SRC" && UBSAN_OPTIONS=halt_on_error=1:abort_on_error=1:print_stacktrace=1 \
		ASAN_OPTIONS=abort_on_error=1 \
		"$FZ/$t" -merge=1 "$TMP" "$C" $SEEDS \
		-artifact_prefix="$RUN/artifacts/$t/") > "$RUN/log/$t.merge.log" 2>&1 || true
	if [ -d "$SEEDS" ]; then
		cp "$SEEDS"/* "$TMP/" 2> /dev/null || true
	fi
	rm -rf "$C"
	mv "$TMP" "$C"
	AFTER=$(find "$C" -type f | wc -l | tr -d ' ')
	printf '%-18s %10s %10s\n' "$t" "$BEFORE" "$AFTER"
done

# ---------------------------------------------------------------- run
NCPU=$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)
# The weights say how to divide the machine, not how many workers to start. libFuzzer times
# a unit by wall clock, so a campaign that runs more workers than there are cores reports its
# own scheduling as slow-unit- artifacts: at 23 workers on 16 cores, inputs measured at 5 ms
# per execution were saved as having taken over 10 s, and every one of them cost a triage.
# One worker per target is the floor, so more targets than cores still oversubscribes.
WEIGHT_SUM=0
for t in $TARGETS; do WEIGHT_SUM=$((WEIGHT_SUM + $(weight_of "$t"))); done
echo
if [ "$DURATION" -eq 0 ]; then
	echo "running until Ctrl-C: $(echo "$TARGETS" | wc -w | tr -d ' ') targets on $NCPU cores"
else
	echo "running $DURATION s: $(echo "$TARGETS" | wc -w | tr -d ' ') targets on $NCPU cores"
fi
echo

# Threshold for saving a slow-unit- artifact. libFuzzer times a unit by wall clock, so its
# 10 s default saves scheduling stalls rather than slow code. A genuine hang still reaches
# -timeout, so this only has to sit above the stalls.
SLOW_UNIT_S=120

PIDS=
for t in $TARGETS; do
	W=$WORKERS
	if [ "$W" -eq 0 ]; then
		W=$(($(weight_of "$t") * NCPU / WEIGHT_SUM))
		[ "$W" -lt 1 ] && W=1
	fi
	D=$HERE/dicts/$t.dict
	DICT=""
	[ -f "$D" ] && DICT="-dict=$D"
	mkdir -p "$RUN/artifacts/$t" "$RUN/work/$t"
	# Each target gets its own working directory containing a data symlink: the server
	# targets resolve $DATADIR from the cwd, and libFuzzer writes per-job logs into it.
	ln -sfn "$SRC/data" "$RUN/work/$t/data"
	# log_path sends each sanitizer report to a file of its own instead of stderr. Without it
	# a -fork child's report dies with the temp directory libFuzzer deletes, and all that
	# reaches log/<target>.log is the one line the parent echoes: no stack, and every
	# round_to_int in the tree reports as the same line of math.h.
	rm -f "$RUN/log/$t.report".*
	# shellcheck disable=SC2086 # $DICT is one option or nothing at all
	(
		cd "$RUN/work/$t"
		UBSAN_OPTIONS="halt_on_error=1:abort_on_error=1:print_stacktrace=1:log_path=$RUN/log/$t.report" \
			ASAN_OPTIONS="abort_on_error=1:log_path=$RUN/log/$t.report" \
			"$FZ/$t" "$RUN/corpus/$t" \
			-fork="$W" -ignore_crashes=1 -report_slow_units="$SLOW_UNIT_S" \
			-max_total_time="$DURATION" -print_final_stats=1 \
			-artifact_prefix="$RUN/artifacts/$t/" $DICT
	) > "$RUN/log/$t.log" 2>&1 &
	PIDS="$PIDS $!"
done

cleanup() {
	echo
	echo "stopping..."
	kill "$SLEEP_PID" 2> /dev/null || true
	for p in $PIDS; do kill "$p" 2> /dev/null || true; done
	# -fork spawns children of its own, so killing the parent is not enough. Match on the
	# exact process name, truncated to the 15 characters the kernel keeps - matching on the
	# full command line instead would also match the ssh/shell invocation that started this
	# script and kill the session.
	for _ in 1 2 3; do
		LEFT=0
		for t in $TARGETS; do
			NAME=$(echo "$t" | cut -c1-15)
			pkill -x "$NAME" 2> /dev/null || true
			# pgrep -c prints 0 AND exits non-zero when nothing matches, so `|| echo 0`
			# would make this "0\n0" and the arithmetic below would abort the script -
			# taking the summary with it.
			N=$(pgrep -xc "$NAME" 2> /dev/null) || N=0
			LEFT=$((LEFT + ${N:-0}))
		done
		[ "$LEFT" -eq 0 ] && break
		sleep 1
	done
}
trap 'cleanup; summary; exit 130' INT TERM

status_line() {
	# libFuzzer's fork-mode parent prints a running summary; take the last one.
	grep -oE '#[0-9]+: cov: [0-9]+ ft: [0-9]+ corp: [0-9]+ exec/s: [0-9]+ oom/timeout/crash: [0-9]+/[0-9]+/[0-9]+' \
		"$RUN/log/$1.log" 2> /dev/null | tail -1
}

# Everything one crashing input needs to be worked on somewhere else: the command, the input
# itself, and what its replay actually says. The summary used to print only a path, so every
# artifact cost a login to the machine that found it before anyone knew whether it was even
# a finding - and most of the time it was not, which is the case this is worth the most in.
REPLAY_LINES=25
# How many artifacts to replay per target. A campaign that finds one reproducible bug saves a
# NEW artifact for every recurrence - `-ignore_crashes=1` is what keeps it fuzzing - so this
# is routinely thousands: one round here ended with 3232 for fz_snap_delta, all of them the
# same over-read. Replaying every one is a serial hour-long stall with nothing to show for it.
# The count of what was skipped is printed, because a silent cap reads as "that was all".
REPLAY_MAX=10
REPLAY_TIMEOUT=""
SEQ_TIMEOUT=""
if command -v timeout > /dev/null 2>&1; then
	# An artifact that hangs would otherwise hang the summary with it. A sequence replay
	# gets its own, much longer limit: it runs the whole corpus, which took 42 s for
	# fz_serverpkt's 7719 inputs.
	REPLAY_TIMEOUT="timeout 60"
	SEQ_TIMEOUT="timeout 900"
fi
# Upper bound on the repetitions of the slow-unit probe below. The probe scales itself to
# stay near SLOW_UNIT_BUDGET_MS whatever one execution costs, so this only caps a cheap input.
SLOW_UNIT_RUNS=5000
SLOW_UNIT_BUDGET_MS=20000
# How many artifacts per target get the full-run search. It replays the entire corpus once
# and then once per bisection step, so it is minutes per artifact rather than seconds.
FULLRUN_MAX=2
FULLRUN_LEFT=$FULLRUN_MAX
SEQ_LEN=1
# A prefix that never got below this many inputs was not isolated to one packet.
FULLRUN_ISOLATED=64

# What the campaign itself recorded, which for an artifact that no longer reproduces is the
# only trace of what fired. The sanitizer report, stack included, is in log/<target>.report.*
# because the run set log_path; log/<target>.log holds libFuzzer's own lines.
report_campaign_lines() {
	t=$1
	L=$({
		cat "$RUN/log/$t.report".* 2> /dev/null
		grep -hE 'runtime error:|ERROR: (AddressSanitizer|LeakSanitizer|MemorySanitizer|libFuzzer)|Assertion' \
			"$RUN/log/$t.log" 2> /dev/null
	} | tail -40) || L=""
	if [ -n "$L" ]; then
		echo
		echo "  $t, what the campaign recorded (last 40 lines):"
		echo "$L" | sed 's/^/      /'
	fi
}

# libFuzzer names an artifact after what made it save the file, and the kinds do not mean the
# same thing. crash-/leak-/oom-/timeout- are failures and are expected to fail again on
# replay. slow-unit- is NOT a failure: it is saved when a single execution exceeded
# -report_slow_units, so it exits 0 on replay BY DEFINITION, and reporting "replays
# cleanly, so it needed earlier state" for one of those is simply wrong.
artifact_kind() {
	case ${1##*/} in
	crash-*) echo crash ;;
	leak-*) echo leak ;;
	oom-*) echo oom ;;
	timeout-*) echo timeout ;;
	slow-unit-*) echo slow ;;
	*) echo other ;;
	esac
}

# Run a list of inputs through ONE process, in the order given. libFuzzer executes several
# FILE arguments back to back without resetting anything in between, and the server fixtures
# deliberately keep their state across inputs, so this is the only way to reproduce something
# that needed an earlier packet. The arguments have to be files: hand it a DIRECTORY and it
# stops replaying and starts a fuzzing run instead.
# Sets RC, LAST_MS (what the last input in the list cost), SEQ_LEN and writes $LOG.
run_inputs() {
	RI_T=$1
	RI_LIST=$2
	RI_TMO=$3
	LOG=$RUN/log/$RI_T.replay.log
	OLDIFS=$IFS
	IFS='
'
	# shellcheck disable=SC2046 # splitting on the newline IFS above is the point
	set -- $(cat "$RI_LIST")
	IFS=$OLDIFS
	SEQ_LEN=$#
	# The same options the campaign ran under. Without them UBSan only prints and carries
	# on, so an artifact that aborted during the run replays as a success.
	(cd "$SRC" && UBSAN_OPTIONS=halt_on_error=1:abort_on_error=1:print_stacktrace=1 \
		ASAN_OPTIONS=abort_on_error=1 $RI_TMO "$FZ/$RI_T" "$@") > "$LOG" 2>&1 && RC=0 || RC=$?
	LAST_MS=$(grep -oE 'in [0-9]+ ms$' "$LOG" | tail -1 | tr -dc '0-9')
	[ -n "$LAST_MS" ] || LAST_MS=0
}

# Did that replay show the thing the artifact was saved for? For a crash kind, a non-zero
# exit; for a slow unit, the last input in the sequence taking longer than libFuzzer's own
# threshold again.
reproduces() {
	case $1 in
	slow) [ "$LAST_MS" -ge $((SLOW_UNIT_S * 1000)) ] ;;
	oom) [ "$RC" -ne 0 ] ;;
	*)
		# The out-of-memory check is not pedantry: a whole corpus replayed in one
		# process accumulates RSS, and tripping -rss_limit_mb that way would read as a
		# reproduced crash. One input that trips it on its own IS the finding though,
		# so only a sequence gets discounted.
		[ "$RC" -ne 0 ] && { [ "$SEQ_LEN" -le 1 ] ||
			! grep -q 'ERROR: libFuzzer: out-of-memory' "$LOG"; }
		;;
	esac
}

# Total milliseconds for $3 executions of one input in one process. libFuzzer's "Executed
# <path> in N ms" covers the whole -runs= loop, so this is the sum and not the average.
# -1 means there was no line to read, which is the replay time limit having killed it: an
# input can be expensive enough that even the short probe does not finish, and reading that
# as 0 ms would turn the slowest artifacts into the ones reported as flat.
time_runs() {
	(cd "$SRC" && $REPLAY_TIMEOUT "$FZ/$1" -runs="$3" "$2") > "$RUN/log/$1.replay.log" 2>&1 || true
	TR_MS=$(grep -oE 'in [0-9]+ ms$' "$RUN/log/$1.replay.log" | tail -1 | tr -dc '0-9')
	[ -n "$TR_MS" ] || TR_MS=-1
}

# A slow unit is a measurement, so the numbers are the report. Two repetition counts rather
# than one, because a single number cannot tell "this input is expensive" from "this input
# makes the NEXT one expensive" - and the second is the bug worth having. The fixtures keep
# server state across inputs, so what hides behind a slow unit is state that GROWS every
# input until some per-input scan turns quadratic, and that shows as a per-execution cost
# that climbs with the repetitions. A flat or falling one means the input neither is slow nor
# makes anything slower: libFuzzer times a unit by wall clock, and on a box running every
# target at once, a unit descheduled for long enough crosses the threshold on its own.
#
# Do not go looking for the campaign's own "Slow unit:" line either. In -fork mode the child
# prints it into a temp directory libFuzzer deletes when it exits, and the parent does not
# echo it, so log/<target>.log has no trace of what was measured.
report_slow_unit() {
	t=$1
	f=$2
	N1=20
	time_runs "$t" "$f" "$N1"
	MS1=$TR_MS
	echo "      replay: NOT a crash - libFuzzer saved this because one execution took over"
	echo "        ${SLOW_UNIT_S} s (-report_slow_units), so exit 0 is the expected outcome."
	if [ "$MS1" -lt 0 ]; then
		echo "        $N1 executions did not finish inside the replay time limit, which is"
		echo "        already the answer: this input is expensive on its own, so measure it"
		echo "        by hand with -runs= and no limit before deciding whether it grows."
		try_full_run "$t" "$f" slow
		return 0
	fi
	PER1=$((MS1 * 1000 / N1))
	# Scale the long probe to the input so an expensive one does not run for an hour.
	EACH=$((PER1 / 1000))
	[ "$EACH" -lt 1 ] && EACH=1
	N2=$((SLOW_UNIT_BUDGET_MS / EACH))
	[ "$N2" -gt "$SLOW_UNIT_RUNS" ] && N2=$SLOW_UNIT_RUNS
	[ "$N2" -lt "$N1" ] && N2=$N1
	time_runs "$t" "$f" "$N2"
	MS2=$TR_MS
	if [ "$MS2" -lt 0 ]; then
		PER2=-1
		echo "        cost per execution, in one process: ${PER1} us over $N1 runs;"
		echo "        the $N2-run probe hit the replay time limit, so there is no second"
		echo "        number and the growth question needs a hand-run probe."
	else
		PER2=$((MS2 * 1000 / N2))
		echo "        cost per execution, in one process: ${PER1} us over $N1 runs, ${PER2} us over $N2 runs"
	fi
	# Doubling alone is not enough to call it: below a millisecond an execution is not
	# heading anywhere near the seconds this artifact claims, and integer division makes the
	# short probe read 0 for a cheap input, which would turn every one of them into growth.
	if [ "$PER2" -ge $((PER1 * 2)) ] && [ "$PER2" -ge 1000 ]; then
		echo "        The per-execution cost CLIMBS with the repetitions, so something in"
		echo "        the fixture grows with every input and this is a real finding: either"
		echo "        server state that is never bounded, or state the harness should reset"
		echo "        between inputs. Find what grows before filing it against the server."
	elif [ "$PER2" -ge 0 ]; then
		echo "        Flat, so the input is not slow and does not make anything slower."
		echo "        libFuzzer times a unit by wall clock and this campaign runs every"
		echo "        target at once, so the ${SLOW_UNIT_S} s was scheduling, not work. The full-run"
		echo "        search below is the check that settles it."
	fi
	try_full_run "$t" "$f" slow
}

# "Replays cleanly" is where triage used to stop, and it did not have to. Whatever state the
# artifact needed came from inputs the fuzzer ran before it in the same process, and the
# corpus is precisely the set of inputs from that run that were worth keeping. Replaying the
# whole corpus and then the artifact in ONE process is a real reproduction attempt, and when
# it works the sequence halves down to something short enough to check in.
#
# It fails when the input that built the state was never saved - libFuzzer only keeps what
# added coverage - and that is an answer too, so it gets printed rather than hidden.
full_run_search() {
	t=$1
	f=$2
	KIND=$3
	W=$RUN/work/$t
	ALL=$W/seq.all
	TRY=$W/seq.try
	KEEP=$W/seq.keep
	find "$RUN/corpus/$t" -type f | sort > "$ALL"
	TOTAL=$(wc -l < "$ALL" | tr -d ' ')
	# Every path has to fit on one command line, and blowing ARG_MAX would look like a
	# failed reproduction rather than the truncation it is. Half of it is margin enough.
	BUDGET=$(($(getconf ARG_MAX 2> /dev/null || echo 262144) / 2))
	awk -v max="$BUDGET" '{ n += length($0) + 1; if(n > max) exit; print }' "$ALL" > "$KEEP"
	USED=$(wc -l < "$KEEP" | tr -d ' ')
	echo "$f" >> "$KEEP"
	echo "      full run: replaying the $USED-input corpus, then the artifact, in one process"
	[ "$USED" -lt "$TOTAL" ] &&
		echo "        (ARG_MAX cap: $((TOTAL - USED)) of $TOTAL corpus inputs left out)"
	run_inputs "$t" "$KEEP" "$SEQ_TIMEOUT"
	if ! reproduces "$KIND"; then
		echo "        does NOT reproduce either (exit $RC, last input ${LAST_MS} ms)."
		echo "        No sequence in the corpus reproduces this artifact, so there is"
		echo "        nothing to store. Either the input that built the state was never"
		echo "        saved - the corpus only keeps what added coverage - or, for a slow"
		echo "        unit, there was no state to build and the timing was the machine."
		return 1
	fi
	# Halve the prefix while a half still reproduces. The state usually comes from ONE
	# earlier input, which this finds in log2(corpus) replays. When neither half works the
	# cause is spread across both, so stop and keep the longer sequence that does
	# reproduce rather than reporting nothing.
	echo "        reproduces. Shrinking the prefix:"
	while :; do
		N=$(($(wc -l < "$KEEP" | tr -d ' ') - 1))
		[ "$N" -le 1 ] && break
		H=$((N / 2))
		HIT=0
		for HALF in 1 2; do
			if [ "$HALF" -eq 1 ]; then
				head -n "$H" "$KEEP" > "$TRY"
			else
				head -n "$N" "$KEEP" | tail -n $((N - H)) > "$TRY"
			fi
			echo "$f" >> "$TRY"
			run_inputs "$t" "$TRY" "$SEQ_TIMEOUT"
			if reproduces "$KIND"; then
				cp "$TRY" "$KEEP"
				HIT=1
				break
			fi
		done
		[ "$HIT" -eq 1 ] || break
		echo "          $(($(wc -l < "$KEEP" | tr -d ' ') - 1)) inputs + the artifact"
	done
	if [ "$(($(wc -l < "$KEEP" | tr -d ' ') - 1))" -gt "$FULLRUN_ISOLATED" ]; then
		echo "        Neither half alone reproduces it, so no single earlier packet is the"
		echo "        cause - this is something that accumulates over many inputs (memory,"
		echo "        a list that is never bounded, a counter). The sequence below is still"
		echo "        a reproducer, just not a minimal one."
	fi

	D=$RUN/repro/$t/${f##*/}
	rm -rf "$D"
	mkdir -p "$D"
	I=0
	while IFS= read -r p; do
		cp "$p" "$(printf '%s/%05d.bin' "$D" "$I")"
		I=$((I + 1))
	done < "$KEEP"
	{
		echo "#!/bin/sh"
		echo "# Reproduces ${f##*/} ($t). The inputs run through one process in this"
		echo "# order; the artifact is the last one and the rest are what built the state"
		echo "# it needed. Override the tree with DDNET_FUZZ_SRC and the binary with"
		echo "# DDNET_FUZZ_BIN."
		echo "set -eu"
		echo "D=\$(cd \"\$(dirname \"\$0\")\" && pwd)"
		echo "cd \"\${DDNET_FUZZ_SRC:-$SRC}\""
		echo "UBSAN_OPTIONS=halt_on_error=1:abort_on_error=1:print_stacktrace=1 \\"
		echo "ASAN_OPTIONS=abort_on_error=1 \\"
		echo "exec \"\${DDNET_FUZZ_BIN:-$FZ/$t}\" \"\$D\"/*.bin"
	} > "$D/replay.sh"
	chmod +x "$D/replay.sh"
	# Replay the STORED copy, not the list it was built from. A reproducer nobody has run
	# from where it now lives is a reproducer nobody has checked.
	(sh "$D/replay.sh") > "$D/replay.log" 2>&1 && RC=0 || RC=$?
	LAST_MS=$(grep -oE 'in [0-9]+ ms$' "$D/replay.log" | tail -1 | tr -dc '0-9')
	[ -n "$LAST_MS" ] || LAST_MS=0
	LOG=$D/replay.log
	echo "        stored: $D"
	echo "          $((I - 1)) input(s) then the artifact, plus replay.sh and its output"
	if reproduces "$KIND"; then
		echo "          verified from there: sh $D/replay.sh (exit $RC, last input ${LAST_MS} ms)"
	else
		echo "          WARNING: the stored copy did NOT reproduce (exit $RC, ${LAST_MS} ms)."
		echo "          Something outside these inputs is part of it - check the fixture"
		echo "          for state that survives a process, sqlite or a written file."
		return 1
	fi
	# Short enough to travel in the summary text, which is the whole point of the base64
	# above. Longer than this and the directory is the only sane way to move it.
	if [ "$I" -le 4 ]; then
		J=0
		while IFS= read -r p; do
			J=$((J + 1))
			[ "$J" -ge "$I" ] && break
			echo "          prefix input $J of $((I - 1)), $(wc -c < "$p" | tr -d ' ') bytes:"
			echo "            base64 -d > $(printf '%05d.bin' $((J - 1))) <<'EOF'"
			base64 < "$p" | sed 's/^/            /'
			echo "            EOF"
		done < "$KEEP"
	else
		echo "          tar cz -C $D . | base64   # to move it off this machine"
	fi
}

# The search costs a full-corpus replay per bisection step, so it is rationed. A campaign
# that finds one state-dependent bug saves an artifact per recurrence, and searching for all
# of them would be the same search over and over.
try_full_run() {
	if [ "$FULLRUN_LEFT" -le 0 ]; then
		echo "        full-run search skipped: already spent on $FULLRUN_MAX artifact(s) of"
		echo "        this target. Run it by hand if this one differs from those."
		return 0
	fi
	FULLRUN_LEFT=$((FULLRUN_LEFT - 1))
	full_run_search "$1" "$2" "$3" || true
}

# The summary replays and times every artifact, which takes minutes, so it only runs once
# the campaign stops. A run left going overnight would hide its first finding until then.
# Announce each new one as it lands instead, with the reproduce line and nothing that costs
# the fuzzers any CPU. Artifacts already present when the run started are not new.
announce_new_artifacts() {
	for t in $TARGETS; do
		for f in "$RUN/artifacts/$t"/*; do
			[ -f "$f" ] || continue
			grep -qxF "$f" "$ANNOUNCED" 2> /dev/null && continue
			printf '%s\n' "$f" >> "$ANNOUNCED"
			[ "${1:-}" = quiet ] && continue
			echo
			echo "  NEW $(artifact_kind "$f") in $t: $f"
			echo "      reproduce: (cd $SRC && $FZ/$t $f)"
			echo "      replayed and judged in the summary when this run ends"
		done
	done
}

report_artifact() {
	t=$1
	f=$2
	KIND=$(artifact_kind "$f")
	echo "  $f"
	echo "      kind: $KIND"
	echo "      reproduce: (cd $SRC && $FZ/$t $f)"
	echo "      input: $(wc -c < "$f" | tr -d ' ') bytes, first 512 with non-printable as '.'"
	# So an rcon line or a name is readable at a glance. The first 512 bytes only: past that
	# it is packed message bodies, and the base64 below is the authoritative copy anyway.
	dd if="$f" bs=1 count=512 2> /dev/null | tr -c '[:print:]' '.' | fold -w 96 | sed 's/^/        /'
	# `tr` turned every newline into a '.' too, so the preview never ends in one and the
	# next line would otherwise be printed onto the end of it.
	echo
	echo "      recreate:"
	echo "        base64 -d > $(basename "$f") <<'EOF'"
	base64 < "$f" | sed 's/^/        /'
	echo "        EOF"
	if [ "$KIND" = slow ]; then
		report_slow_unit "$t" "$f"
		return 0
	fi
	printf '%s\n' "$f" > "$RUN/work/$t/seq.one"
	run_inputs "$t" "$RUN/work/$t/seq.one" "$REPLAY_TIMEOUT"
	echo "      replay: exit $RC"
	if reproduces "$KIND"; then
		grep -vE '^(INFO:|Running:|Executed |artifact_prefix|\*\*\*|$)' "$LOG" |
			tail -"$REPLAY_LINES" | sed 's/^/        /'
		return 0
	fi
	echo "        Replays CLEANLY, so this file on its own is not the reproducer: it needed"
	echo "        state an earlier input left behind in the same process."
	try_full_run "$t" "$f" "$KIND"
}

summary() {
	echo
	echo "================================ summary ================================"
	printf '%-18s %9s %8s %9s %9s  %s\n' target cov ft corpus artifacts ""
	TOTAL_ART=0
	for t in $TARGETS; do
		L=$(status_line "$t")
		COV=$(echo "$L" | grep -oE 'cov: [0-9]+' | awk '{print $2}')
		FT=$(echo "$L" | grep -oE 'ft: [0-9]+' | awk '{print $2}')
		CORP=$(find "$RUN/corpus/$t" -type f 2> /dev/null | wc -l | tr -d ' ')
		A=$(find "$RUN/artifacts/$t" -type f 2> /dev/null | wc -l | tr -d ' ')
		TOTAL_ART=$((TOTAL_ART + A))
		NOTE=""
		[ -z "$COV" ] && NOTE="  <-- produced no status line; check $RUN/log/$t.log"
		[ "$A" -gt 0 ] && NOTE="  <-- ARTIFACTS"
		printf '%-18s %9s %8s %9s %9s%s\n' "$t" "${COV:--}" "${FT:--}" "$CORP" "$A" "$NOTE"
	done
	echo
	if [ "$TOTAL_ART" -gt 0 ]; then
		echo "artifacts:"
		for t in $TARGETS; do
			A=$(find "$RUN/artifacts/$t" -type f 2> /dev/null | wc -l | tr -d ' ')
			[ "$A" -gt 0 ] || continue
			report_campaign_lines "$t"
			FULLRUN_LEFT=$FULLRUN_MAX
			N=0
			for f in "$RUN/artifacts/$t"/*; do
				[ -f "$f" ] || continue
				N=$((N + 1))
				[ "$N" -gt "$REPLAY_MAX" ] && continue
				echo
				report_artifact "$t" "$f"
			done
			if [ "$A" -gt "$REPLAY_MAX" ]; then
				echo
				echo "  $t: $((A - REPLAY_MAX)) further artifacts not replayed, in $RUN/artifacts/$t."
				echo "  A count this high is almost always ONE bug saved once per recurrence."
				echo "  Bucket them before triaging:"
				echo "    for f in $RUN/artifacts/$t/*; do (cd $SRC && $FZ/$t \$f 2>&1 |"
				echo "      grep -m1 -E '^    #[0-9]+ '); done | sort | uniq -c | sort -rn"
			fi
		done
		echo
		echo "NOTE: the replay above is what decides whether an artifact is a finding, and"
		echo "      what counts as a replay depends on the kind. crash-/leak-/oom-/timeout-"
		echo "      must fail again; slow-unit- exits 0 whatever happens and is judged on"
		echo "      its per-execution cost instead. An artifact that should fail and does"
		echo "      not needed state from an earlier input, so it names the last packet and"
		echo "      not the cause - the full-run search then tries to rebuild the sequence"
		echo "      from the corpus and stores it under $RUN/repro. When it cannot, the"
		echo "      input that built the state was never saved and there is no reproducer"
		echo "      to check in. That has meant a defect in the harness as often as one in"
		echo "      the server."
	else
		echo "no artifacts."
	fi
	echo "corpus kept in $RUN/corpus - it is minified at the start of the next run."
}

# Live status while it runs, so there is something to watch. With no time limit this loops
# until Ctrl-C (the trap above stops the children and still prints the summary) or until
# every target has exited on its own.
ANNOUNCED=$RUN/work/.announced
: > "$ANNOUNCED"
announce_new_artifacts quiet
START=$(date +%s)
END=$((START + DURATION))
while :; do
	# `wait` is interruptible by a trapped signal; a bare `sleep` is not - the trap would
	# only run after it returned, so Ctrl-C looked like it did nothing for up to 30 s.
	sleep 30 &
	SLEEP_PID=$!
	wait "$SLEEP_PID" 2> /dev/null || true
	NOW=$(date +%s)
	[ "$DURATION" -gt 0 ] && [ "$NOW" -ge "$END" ] && break
	# shellcheck disable=SC2009 # pgrep -c prints 0 and exits non-zero, which set -e takes badly
	ALIVE=$(ps -eo comm | grep -c '^fz_' || true)
	if [ "$ALIVE" -eq 0 ]; then
		echo "all targets exited"
		break
	fi
	if [ "$DURATION" -gt 0 ]; then
		printf '[%5ss left, %2s procs] ' "$((END - NOW))" "$ALIVE"
	else
		printf '[%5ss elapsed, %2s procs] ' "$((NOW - START))" "$ALIVE"
	fi
	for t in $TARGETS; do
		L=$(status_line "$t")
		printf '%s=%s ' "$(echo "$t" | sed 's/^fz_//')" "$(echo "$L" | grep -oE 'cov: [0-9]+' | awk '{print $2}')"
	done
	echo
	announce_new_artifacts
done

# shellcheck disable=SC2086 # $PIDS is a list of pids
wait $PIDS 2> /dev/null || true
trap - INT TERM
summary
