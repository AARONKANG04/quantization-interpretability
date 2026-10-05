#!/usr/bin/env bash
# Kill processes whose command line matches $1 (and their children), excluding this script and its caller.
pat="$1"
for p in $(pgrep -f "$pat"); do
  [ "$p" = "$$" ] && continue; [ "$p" = "$PPID" ] && continue
  echo "killing $p: $(ps -o cmd= -p "$p" | cut -c1-90)"
  pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null
done
sleep 3
for p in $(pgrep -f "$pat"); do [ "$p" = "$$" ] || [ "$p" = "$PPID" ] || kill -9 "$p" 2>/dev/null; done
