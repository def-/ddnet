#!/bin/sh
# Builds the converter once and installs the same binary everywhere it runs:
# the rank pipeline on this host and the moderators' page on the web host.
# The web host is Debian 13 and cannot run a binary built here, so the build
# happens in a Debian 13 container and its binary runs on both (glibc is
# forward compatible, the other way around it is not).
#
# Usage: deploy-tools.sh   (on the archive host, from any directory)
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=build-trixie
LOCAL=$REPO/build-tools
REMOTE=/home/teeworlds/bin/teehistorian2demo

docker run --rm -v "$REPO:$REPO" -v "$HOME/git/ddnet/.git:$HOME/git/ddnet/.git" -w "$REPO" debian:trixie sh -c '
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq >/dev/null
	apt-get install -y -qq build-essential cmake ninja-build python3 rustc cargo git \
		libcurl4-openssl-dev libfreetype-dev libpng-dev libsqlite3-dev libssl-dev zlib1g-dev >/dev/null 2>&1
	git config --global --add safe.directory "*"
	[ -f '"$BUILD"'/CMakeCache.txt ] || cmake -S . -B '"$BUILD"' -GNinja -DCMAKE_BUILD_TYPE=Release \
		-DCLIENT=OFF -DSERVER=OFF -DTOOLS=ON -DVULKAN=OFF -DVIDEORECORDER=OFF -DANTIBOT=OFF -DMYSQL=OFF -DWEBSOCKETS=OFF -DUPNP=OFF
	cmake --build '"$BUILD"' --target teehistorian2demo demo_scramble demo_splice
'

# Installed by rename: a conversion that is running keeps the file it opened,
# and the pipeline reads the mtime to decide which demos are out of date
for tool in teehistorian2demo demo_scramble demo_splice; do
	cp "$REPO/$BUILD/$tool" "$LOCAL/$tool.next"
	chmod 755 "$LOCAL/$tool.next"
	mv -f "$LOCAL/$tool.next" "$LOCAL/$tool"
done
"$LOCAL/teehistorian2demo" 2>&1 | head -1

scp -q "$REPO/$BUILD/teehistorian2demo" "ddnet:$REMOTE.next"
ssh ddnet "mv -f $REMOTE.next $REMOTE && $REMOTE 2>&1 | head -1"
