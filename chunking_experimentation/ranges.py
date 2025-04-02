# This script is adapted from / inspired by the https://github.com/brandonstarxel/chunking_evaluation


def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges):
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect_ranges(r1, r2):
    start = max(r1[0], r2[0])
    end = min(r1[1], r2[1])
    return (start, end) if start < end else None
