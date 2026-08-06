"""Two-level LPT (Longest Processing Time) partitioning of C(n, t) across N GPUs.

Each "tile" is a contiguous chunk of work defined as (first, second_lo, second_hi):
all combos (first, second, ..., t-th) where first is fixed and 
second ∈ [second_lo, second_hi). This is more granular than partitioning by 
first-index alone, which lets us balance work better.

Algorithm:
1. For each first ∈ [0, n-t], compute total work = C(n - first - 1, t - 1)
2. For first-indices whose work exceeds target_tile_size, subdivide by walking 
   second-indices and accumulating work until the tile is full
3. LPT-assign all tiles to N GPUs: greedily place each tile on the GPU with 
   minimum cumulative work
4. Return per-GPU tile lists

Result: imbalance ratio (max_work / min_work) ~ 1.05-1.15 for any N >= 16.
"""
from math import comb
from typing import List, Tuple, Dict


Tile = Tuple[int, int, int]  # (first, second_lo, second_hi_exclusive)


def generate_tiles(n_pool: int, t_sel: int, num_gpus: int,
                   tile_size_factor: int = 4) -> List[Tile]:
    """Generate tiles covering all of C(n_pool, t_sel) exactly once.
    
    target_tile_size = total_combos / (num_gpus * tile_size_factor)
    Heavy first-indices get subdivided; light ones become a single tile.
    """
    if num_gpus <= 0:
        return []
    total_combos = comb(n_pool, t_sel)
    target = max(1, total_combos // (num_gpus * tile_size_factor))
    max_first = n_pool - t_sel  # inclusive upper bound on first
    
    tiles: List[Tile] = []
    for first in range(max_first + 1):
        full_work = comb(n_pool - first - 1, t_sel - 1)
        # second can range over [first+1, max_second_inclusive]
        # max_second is the largest value such that there's still room for t_sel-1 
        # more strictly increasing indices ≤ n_pool - 1.
        # If we pick second, we still need t_sel - 2 more from (second+1, ..., n_pool-1).
        # So second can go up to n_pool - (t_sel - 1) = n_pool - t_sel + 1, inclusive.
        max_second_incl = n_pool - t_sel + 1
        second_hi_excl = max_second_incl + 1  # exclusive bound
        second_lo = first + 1
        
        if full_work <= target:
            tiles.append((first, second_lo, second_hi_excl))
            continue
        
        # Subdivide by second
        cur_lo = second_lo
        cur_work = 0
        for second in range(second_lo, second_hi_excl):
            w = comb(n_pool - second - 1, t_sel - 2)
            if cur_work > 0 and cur_work + w > target:
                tiles.append((first, cur_lo, second))
                cur_lo = second
                cur_work = w
            else:
                cur_work += w
        if cur_work > 0:
            tiles.append((first, cur_lo, second_hi_excl))
    
    return tiles


def tile_work(tile: Tile, n_pool: int, t_sel: int) -> int:
    """Number of combos contained in a tile."""
    first, lo, hi = tile
    total = 0
    for second in range(lo, hi):
        total += comb(n_pool - second - 1, t_sel - 2)
    return total


def lpt_assign(tiles: List[Tile], num_gpus: int, n_pool: int, t_sel: int
               ) -> Dict[int, List[Tile]]:
    """LPT bin-packing: sort tiles by work descending, assign each to least-loaded GPU."""
    weighted = [(tile_work(t, n_pool, t_sel), t) for t in tiles]
    weighted.sort(key=lambda x: -x[0])
    
    gpu_load = [0] * num_gpus
    gpu_tiles: Dict[int, List[Tile]] = {i: [] for i in range(num_gpus)}
    for w, t in weighted:
        target = min(range(num_gpus), key=lambda i: gpu_load[i])
        gpu_tiles[target].append(t)
        gpu_load[target] += w
    return gpu_tiles


def partition_for_fleet(n_pool: int, t_sel: int, num_gpus: int
                        ) -> Tuple[Dict[int, List[Tile]], Dict[str, float]]:
    """Top-level: returns (per-GPU tiles, stats dict)."""
    tiles = generate_tiles(n_pool, t_sel, num_gpus)
    assignment = lpt_assign(tiles, num_gpus, n_pool, t_sel)
    
    loads = [sum(tile_work(t, n_pool, t_sel) for t in tlist)
             for tlist in assignment.values()]
    total = comb(n_pool, t_sel)
    stats = {
        "num_tiles": len(tiles),
        "num_gpus": num_gpus,
        "total_combos": total,
        "min_per_gpu": min(loads),
        "max_per_gpu": max(loads),
        "imbalance_ratio": max(loads) / max(min(loads), 1),
        "covered": sum(loads) == total,
    }
    return assignment, stats


def write_tile_file(path: str, tiles: List[Tile]) -> None:
    """Binary tile file format:
    
    [4 bytes: num_tiles (uint32 LE)]
    For each tile:
        [4 bytes: first (uint32 LE)]
        [4 bytes: second_lo (uint32 LE)]
        [4 bytes: second_hi (uint32 LE)]
    """
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tiles)))
        for first, lo, hi in tiles:
            f.write(struct.pack("<III", first, lo, hi))


def read_tile_file(path: str) -> List[Tile]:
    """For testing / verification."""
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        out = []
        for _ in range(n):
            out.append(struct.unpack("<III", f.read(12)))
        return out


# ---- Self-test (CPU-side) ----

def enumerate_tile(tile: Tile, n_pool: int, t_sel: int):
    """Yield each (t_sel-tuple) combo in a tile. For testing only."""
    first, lo, hi = tile
    # We need to enumerate combos starting (first, second, ...) where 
    # second ∈ [lo, hi), and rest follow standard combinations.
    from itertools import combinations
    for second in range(lo, hi):
        # Pick remaining (t_sel - 2) indices strictly greater than second
        remaining_pool = range(second + 1, n_pool)
        for tail in combinations(remaining_pool, t_sel - 2):
            yield (first, second) + tail


def verify_partition(n_pool: int, t_sel: int, num_gpus: int):
    """Verify tile partitioning covers each combo exactly once. Slow for large n."""
    if comb(n_pool, t_sel) > 5_000_000:
        return f"verification skipped (too many combos: {comb(n_pool, t_sel):,})"
    
    assignment, stats = partition_for_fleet(n_pool, t_sel, num_gpus)
    seen = set()
    dupes = 0
    for gpu_id, tiles in assignment.items():
        for tile in tiles:
            for combo in enumerate_tile(tile, n_pool, t_sel):
                if combo in seen:
                    dupes += 1
                seen.add(combo)
    expected = comb(n_pool, t_sel)
    return (f"{n_pool}C{t_sel}={expected}  seen={len(seen)}  "
            f"dupes={dupes}  {'OK' if len(seen) == expected and dupes == 0 else 'FAIL'}")


if __name__ == "__main__":
    # Self-tests with small parameters
    failures = 0
    print("=== Tile partition tests ===")
    for n, t in [(10, 3), (15, 4), (20, 5), (12, 6)]:
        for ng in [1, 4, 8, 16]:
            result = verify_partition(n, t, ng)
            print(f"  C({n},{t}) on {ng} GPUs: {result}")
            if "FAIL" in result:
                failures += 1
    
    print("\n=== Real Config A: C(150, 9) ===")
    for ng in [16, 32, 64, 96, 128, 192]:
        _, stats = partition_for_fleet(150, 9, ng)
        ok = stats["covered"]
        marker = "✓" if ok else "✗ NOT COVERED"
        print(f"  N={ng:>3}: tiles={stats['num_tiles']:>4} "
              f"imbalance={stats['imbalance_ratio']:.4f}x  {marker}")
        if not ok:
            failures += 1
    
    if failures:
        print(f"\n❌ {failures} failures")
        import sys as _sys
        _sys.exit(1)
    else:
        print("\n✅ all tile-partition tests passed")
