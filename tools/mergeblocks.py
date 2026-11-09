#!/usr/bin/env python3
"""mergeblocks.py OUT P1 P1.left P2 P2.left ... Pn [Pn.left]: the boundary blocks of a multi-pass search.

Pass 1 covers every class it can; each later pass is run (--only) on the classes the previous one left
(its --left file) and writes complete blocks for them.  OUT gets each pass's blocks for the classes not
in its .left file, and the last pass's blocks.  Checks that every pass after the first has exactly the
previous pass's left classes and that the last pass leaves none.  Files ending in .zst are read and
written through zstd."""
import sys, subprocess

def lines(f):
    if not f.endswith('.zst'): return open(f)
    return subprocess.Popen(['zstd', '-dc', '-q', f], stdout=subprocess.PIPE, text=True).stdout

def blocks(f):
    cur = []
    for l in lines(f):
        l = l.rstrip('\n')
        if not l:
            if cur: yield cur; cur = []
        else: cur.append(l)
    if cur: yield cur

if len(sys.argv) < 3:
    print(__doc__.strip()); sys.exit(1)
print('mergeblocks.py ' + ' '.join(sys.argv[1:]))
out, args = sys.argv[1], sys.argv[2:]
passes = [(args[i], args[i + 1] if i + 1 < len(args) else None) for i in range(0, len(args), 2)]
zp = subprocess.Popen(['zstd', '-q', '-10', '-f', '-o', out], stdin=subprocess.PIPE, text=True) if out.endswith('.zst') else None
prev, n = None, 0
with (zp.stdin if zp else open(out, 'w')) as o:
    for txt, left in passes:
        L = set(l.strip() for l in open(left) if l.strip()) if left else set()
        firsts, kept = set(), 0
        for b in blocks(txt):
            firsts.add(b[0])
            if b[0] in L: continue
            o.write('\n'.join(b) + '\n\n'); kept += 1
        if prev is not None and firsts != prev: sys.exit('%s: its classes are not the previous pass\'s left classes' % txt)
        if not L <= firsts: sys.exit('%s: its .left file names classes it has no block for' % txt)
        print('%s: %d blocks, %d kept, %d left' % (txt, len(firsts), kept, len(L)))
        n += kept; prev = L
if zp and zp.wait() != 0: sys.exit('zstd failed')
if prev: sys.exit('the last pass left %d classes' % len(prev))
print('%s: %d blocks' % (out, n))
