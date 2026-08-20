#!/bin/sh
# Build the libFuzzer targets against an instrumented DDNet object set.
#
# Each target is linked by reusing an existing executable's link line with our object swapped
# in, which is the same trick tools/build_fakeclient.sh uses and needs no CMakeLists changes.
# Two base link lines are used:
#
#   twping        - enough for engine-only targets (packet, snapshot, message parsers)
#   DDNet-Server  - needed by anything touching game-server code. Those targets replace
#                   main.cpp, so they must stub the symbols it defined (IsInterrupted).
#
# The build directory ($SRC/build-fuzz, override with DDNET_FUZZ_BUILD) is configured here on
# first use rather than by hand, because the instrumentation the objects carry has to match
# what the harnesses below are compiled with.
#
# The harness must be compiled with the SAME semantic flags as the tree, not just the same
# defines. DDNet applies -fno-exceptions to every target (CMakeLists.txt adds it to
# OUR_FLAGS, applied PRIVATE to all of them), so a harness built with exceptions emits a
# different body for every libstdc++ inline that contains a try/catch - std::vector's
# range-init among them - and the linker picks one arbitrarily. That is a silent ODR
# violation on exactly the types that cross the harness/library boundary, and ASan's ODR
# detector only covers globals so it never fires.
#
# (An earlier comment here claimed the opposite and blamed -fsanitize=function for a
# type-mismatch report. That report had a different cause: the harness was missing
# _GLIBCXX_DEBUG, which changes std::vector's ABI. Extracting the project's defines fixed
# it; dropping -fno-exceptions was never the right response.)
#
# On macOS, Apple clang ships no libFuzzer runtime (libclang_rt.fuzzer_osx.a is absent), so
# set DDNET_FUZZ_CXX=/opt/homebrew/opt/llvm/bin/clang++.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${DDNET_FUZZ_SRC:-$(cd "$HERE/../.." && pwd)}
BUILD=${DDNET_FUZZ_BUILD:-$SRC/build-fuzz}
CXX=${DDNET_FUZZ_CXX:-clang++}
# The matching C compiler, not the default cc: the C sources are instrumented too, and gcc
# has no -fsanitize=fuzzer-no-link to configure with.
CC=${CXX%++}
OUT=${FUZZ_OUT:-$BUILD/fuzzers}
mkdir -p "$OUT"

if [ ! -f "$BUILD/build.ninja" ]; then
	FLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g -O1"
	cmake -S "$SRC" -B "$BUILD" -GNinja \
		-DCLIENT=OFF -DSERVER=ON -DTOOLS=ON -DDOWNLOAD_GTEST=OFF \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
		-DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
fi

cd "$BUILD"

TARGETS=${*:-"fz_unpack_packet fz_snap_delta fz_unpack_msg fz_savetee fz_console fz_sevensix fz_chunk fz_fuzzystr fz_nameban fz_gamemsg fz_netserver fz_serverpkt fz_tiles"}
for t in $TARGETS; do
	echo "=== $t"
	# Targets that reach into game-server code need the server object set.
	case "$t" in
	fz_savetee | fz_console | fz_sevensix | fz_fuzzystr | fz_nameban | fz_gamemsg | fz_netserver | fz_serverpkt | fz_tiles)
		BASE=DDNet-Server
		OBJ='CMakeFiles/game-server.dir/src/engine/server/main.cpp.o'
		;;
	*)
		BASE=twping
		OBJ='CMakeFiles/twping.dir/src/tools/twping.cpp.o'
		;;
	esac
	ninja "$BASE"
	ninja -t commands "$BASE" | tail -1 > "$OUT/.linkline.$BASE"

	# Mirror the project's own preprocessor defines instead of hardcoding a guess. Missing
	# _GLIBCXX_DEBUG (which a Debug build sets) silently changes the ABI of every std::
	# container: std::vector becomes std::__debug::vector, a different type with a different
	# layout. A harness built without it hands the library a differently-shaped object, which
	# surfaces as bogus ASan overflow reports inside vector::begin() and as
	# -fsanitize=function type mismatches at the boundary - both of which look like target
	# defects and are not.
	# Match a C++ compile specifically: the first " -c " line is a C dependency (zlib and
	# friends) that carries none of the project's defines.
	DEFINES=$(ninja -t commands "$BASE" | grep -m1 -e '-std=c++' | tr ' ' '\n' | grep '^-D' | tr '\n' ' ')
	# Fail, do not warn: the fallback would silently rebuild the very ABI mismatch this
	# extraction exists to prevent, and under nohup nobody reads a warning.
	if [ -z "$DEFINES" ]; then
		echo "ERROR: no -D flags extracted from $BASE; refusing to build with a guessed ABI" >&2
		exit 1
	fi

	# shellcheck disable=SC2086 # $DEFINES is a list of -D flags
	$CXX -c "$HERE/$t.cpp" -o "$OUT/$t.o" \
		-std=c++20 -g -O1 -fno-omit-frame-pointer \
		-fsanitize=fuzzer-no-link,address,undefined \
		-fno-sanitize-recover=undefined \
		-fno-exceptions -fsigned-char \
		$DEFINES \
		-I"$BUILD/src" -I"$SRC/src" -I"$SRC/src/rust-bridge" \
		-DFZ_UBSAN_SUPP="\"$HERE/ubsan.supp\"" \
		-Wall -Wno-unused-parameter -Wno-format
	# swap the base executable's own object for ours, retarget the output, link the runtime in
	LINK=$(sed -e "s|$OBJ|$OUT/$t.o|" \
		-e "s|-o $BASE|-o $OUT/$t|" \
		-e 's|-fsanitize=fuzzer-no-link|-fsanitize=fuzzer|g' "$OUT/.linkline.$BASE")
	case "$LINK" in
	*-fsanitize=fuzzer*) ;;
	*) LINK="$LINK -fsanitize=fuzzer" ;;
	esac
	# Verify both rewrites landed. sed is a no-op on a miss, and both misses fail SILENTLY in
	# the worst possible way: if the object swap misses, main.cpp still supplies main(), the
	# link succeeds, and the "fuzzer" is a renamed DDNet server that the campaign will happily
	# run and count as healthy. If the -o rewrite misses, the output overwrites the base
	# executable instead.
	case "$LINK" in
	*"$OUT/$t.o"*) ;;
	*)
		echo "ERROR: object swap failed for $t (pattern '$OBJ' not in link line)" >&2
		exit 1
		;;
	esac
	case "$LINK" in
	*"-o $OUT/$t "* | *"-o $OUT/$t") ;;
	*)
		echo "ERROR: output rewrite failed for $t" >&2
		exit 1
		;;
	esac
	eval "$LINK"
	echo "built: $OUT/$t"
done
