#!/usr/bin/env bash
# Pre-download Fuel models used by sim/worlds/nnc_yard.sdf. Missing models are
# logged and skipped; the world still has primitive boxes/cylinder plus PX4
# arucotag/helipad.
set -u
URLS=(
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pickup"
  "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Construction Cone"
)
ok=0
fail=0
for url in "${URLS[@]}"; do
  echo "gz fuel download -u $url"
  if timeout 45 gz fuel download -u "$url"; then
    ok=$((ok + 1))
  else
    echo "skip: could not download $url" >&2
    fail=$((fail + 1))
  fi
done
echo "fuel download ok=$ok fail=$fail"
exit 0
