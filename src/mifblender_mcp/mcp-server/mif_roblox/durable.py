"""Durable writes for the files MifBridge keeps between runs (records, locks' queues, manifests).

Found 2026-11-22: a power loss at 17:25 left the machine's coverage record AND its last-good copy empty at once. Both were
written to a temp file and renamed over the old one, and the rename reached the disk while the temp file's contents had
not: NTFS journals the rename (metadata), not the data. So every temp file is flushed and fsynced before it replaces
anything; then a power loss leaves the old file or the new one, never an empty one."""
import os


def sync(fh):
    """Flush fh and force its bytes to the disk. Call it inside the with block, before the replace."""
    fh.flush()
    os.fsync(fh.fileno())
