#!/bin/sh
# Builds the converter the rank pipeline uses for the moderators' page as
# well, so both always run the same code: ddnet (Debian 13) cannot run a
# binary built on this host, so it is built in a Debian 13 container from
# this checkout and put into place on the web host by a rename, which a
# conversion that is running there does not mind.
#
# Usage: deploy-moderator-tool.sh   (on the archive host, from any directory)
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=build-trixie
TARGET=ddnet:/home/teeworlds/bin/teehistorian2demo

docker run --rm -v "$REPO:$REPO" -v "$HOME/git/ddnet/.git:$HOME/git/ddnet/.git" -w "$REPO" debian:trixie sh -c '
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq >/dev/null
	apt-get install -y -qq build-essential cmake ninja-build python3 rustc cargo git \
		libcurl4-openssl-dev libfreetype-dev libpng-dev libsqlite3-dev libssl-dev zlib1g-dev >/dev/null 2>&1
	git config --global --add safe.directory "*"
	[ -f '"$BUILD"'/CMakeCache.txt ] || cmake -S . -B '"$BUILD"' -GNinja -DCMAKE_BUILD_TYPE=Release \
		-DCLIENT=OFF -DSERVER=OFF -DTOOLS=ON -DVULKAN=OFF -DVIDEORECORDER=OFF -DANTIBOT=OFF -DMYSQL=OFF -DWEBSOCKETS=OFF -DUPNP=OFF
	cmake --build '"$BUILD"' --target teehistorian2demo
'
scp -q "$REPO/$BUILD/teehistorian2demo" "$TARGET.next"
ssh ddnet "mv -f ${TARGET#*:}.next ${TARGET#*:} && ${TARGET#*:} 2>&1 | head -1"
