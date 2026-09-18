#!/usr/bin/env bash
# Pre-download Fuel models used by sim/worlds/nnc_yard.sdf.
# Missing models are logged and skipped; the world still has primitive geometry.
set -u
URLS=(
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pickup"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Construction Cone"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing person"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Hatchback"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Casual female"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/SUV"
)
ok=0
fail=0
for url in "${URLS[@]}"; do
  echo "gz fuel download -u $url"
  if timeout 180 gz fuel download -u "$url"; then
    echo "ok: $url"
    ok=$((ok + 1))
  else
    echo "skip: could not download $url" >&2
    fail=$((fail + 1))
  fi
done
echo "fuel download ok=$ok fail=$fail"
exit 0
