#!/usr/bin/env bash
# Format (blank disks only) and mount the persistent data disk at /data. Idempotent, needs root.
# usage: sudo scripts/remote/mount_data.sh [device-name=data] [owner=$SUDO_USER]
set -euo pipefail
name="${1:-data}"; owner="${2:-${SUDO_USER:-$USER}}"
dev="/dev/disk/by-id/google-$name"
if [ ! -e "$dev" ]; then
  # fall back to the only non-boot, non-local persistent disk
  cand=$(ls /dev/disk/by-id/ | grep '^google-' | grep -v -e 'persistent-disk-0' -e 'local-ssd' -e '-part' || true)
  [ "$(echo "$cand" | grep -c .)" = 1 ] || { echo "cannot identify the data disk; candidates:"; echo "$cand"; exit 1; }
  dev="/dev/disk/by-id/$cand"
fi
echo "data disk: $dev -> $(readlink -f "$dev")"
blkid "$dev" >/dev/null 2>&1 || mkfs.ext4 -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
mkdir -p /data
uuid=$(blkid -s UUID -o value "$dev")
grep -q "$uuid" /etc/fstab || echo "UUID=$uuid /data ext4 discard,defaults,nofail 0 2" >> /etc/fstab
mountpoint -q /data || mount /data
chown "$owner:$owner" /data
df -h /data | tail -1
