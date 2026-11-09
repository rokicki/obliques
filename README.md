# Upper bounds for solving the n x n x n cube, orbit by orbit

**Result.** Every position of the n x n x n Rubik's cube can be solved in at most

| | even n | odd n |
|---|---|---|
| **certified (P = 80)** | 10 n^2 + 43 n - 115 | 10 n^2 + 42.5 n - 197.5 |
| **with boundary passes (P = 71)** | (71 n^2 + 398 n - 992) / 8 | (71 n^2 + 412 n - 1715) / 8 |

moves in the single-slice turn metric (SSTM: a quarter or half turn of any one layer counts 1). For small n:

| n | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 16 | 20 | 32 | 64 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P = 80 | 217 | 265 | 503 | 590 | 869 | 995 | 1,315 | 1,480 | 1,841 | 3,133 | 4,745 | 11,501 | 43,597 | 104,185 |
| P = 71 | 217 | 265 | 494 | 581 | 842 | 968 | 1,261 | 1,426 | 1,751 | 2,944 | 4,421 | 10,556 | 39,412 | 93,601 |

To our knowledge these are the first explicit upper bounds for every n from 6 on; see
[Comparison](#comparison-with-published-bounds).

The bound for P = 80 rests only on the certificate in this repository: a list of solving sequences that
anyone can check with the programs in `tools/`, and that `solver/certsolve` uses to solve any cube within
the bound. The bound for P = 71 adds a computational check of the boundaries between stages, described
below.

## How the cube is solved

Every piece of the n x n x n cube belongs to an **orbit**: a set of 24 pieces (8 for the corners) that
moves only among its own positions. The centre orbits are named by their offsets (x, y) from a face's
corner:

- **oblique** centres (x != y, neither in the middle row): these come in **chiral pairs** (p, q) and (q, p);
- **diagonal** (X-) centres (x = y);
- **mid** (T-) centres (odd n only; one coordinate in the middle row);
- **wings** (edge pieces off the middle);
- corners, and on odd n the 12 **midges** (middle edges).

With k = (n - 2)/2 for even n and k = (n - 3)/2 for odd n, there are k(k - 1)/2 oblique pairs, k diagonal
orbits, k mid orbits (odd n) and k wing orbits. The solve goes one orbit at a time, in this order. Each
stage keeps everything the earlier stages solved, and centre pieces of the same colour are
interchangeable.

| phase | pieces | stages (worst case of each) | per orbit |
|---|---|---|---|
| 1 | every oblique pair: both orbits' U pieces onto U (other obliques keep U and D) | 1ud: 17 | |
| 2 | every oblique pair: both orbits' D pieces onto D | 2ud: 18 | |
| 3 | each oblique pair: F and B pieces onto the F/B axis, L and R pieces onto the L/R axis | 3ax: 23 | |
| 4 | each oblique pair: F vs B and L vs R | 4op: 22 | **P = 80** per pair |
| 5 | each mid orbit (odd n) | U and D 18, then the side ring 21 | **M = 39** |
| 6 | each diagonal orbit | D 10, U 16, the side ring 18 | **D = 44** |
| 7 | corners (even n) / corners and midges (odd n) | one 3x3x3 solve with face turns | **11** / **20** |
| 8 | each wing orbit (keeps every centre, corner and midge, and the other wing orbits) | 4 wings 32, 4 wings 32, 5 wings 40, the last 11 wings 58 | **W = 162** |

The first four phases are the obliques, which make up almost all of the cube for large n. Their algorithms
are written once for the pair (2, 3) / (3, 2) of a 6x6x6 and work for every pair (p, q) of every n
(layers 2 and 3 become layers p + 1 and q + 1). Adding up:

- even n: P k(k - 1)/2 + (D + W) k + 11
- odd n: P k(k - 1)/2 + (D + M + W) k + 20

That gives the polynomials above; centres alone take P k(k - 1)/2 + D k (+ M k), about P n^2 / 8.

### Where 80 comes from

Each oblique stage is a finite puzzle: all positions of the pair's 48 pieces that can occur at that
stage. For each one, the certificate gives a solving sequence for every position, one sequence per
symmetry class. The longest are 17, 18, 23 and 22 moves, and 17 + 18 + 23 + 22 = **80**.

| stage | positions (classes) | longest |
|---|---|---|
| 1ud | 14,119,851 classes | 17 |
| 2ud | 2,936,935 classes | 18 |
| 3ax | 165,636,900 positions (10,366,284 classes) | 23 |
| 4op | 24,010,000 positions (1,508,387 classes) | 22 |

The checkers in `tools/` verify, from the sequences alone, that every position of every stage is covered,
that each sequence solves its position and keeps every other orbit as required (tested on the 6x6x6 to
10x10x10 and for every pair), and the stated lengths. Two of them, `udindcheck.py` and `axindcheck.py`,
share no code with the others: they use only the [twsearch](https://github.com/cubing/twsearch) puzzle
definitions in `tools/tws/`.

### Where 71 comes from: boundary passes

The plain sum assumes the worst line of one stage is followed by the worst line of the next. It rarely
is. A **boundary pass** bounds two consecutive stages together. Take each position of the first stage
whose line L0 is long enough to matter, and each way the rest of the cube can be coloured when L0 ends.
For every coloured position that L0 would carry to a next-stage position too deep, it finds a different
sequence for the first stage (an alternative of up to a few extra moves), and checks that the total stays
within the budget X.

- **UD 33**: 1ud + 2ud <= 33 (plain 35). Search: `gpuboundx`; check: `gpubound`, 5.24e12 colourings, 0 not
  solved.
- **AX 38**: 3ax + 4op <= 38 (plain 45). Search: `gpuaxbx`, 7,263,089 blocks; check: `gpuaxbcheck`,
  147,437,136,202,996 positions, 0 not solved; a CPU re-check of 1,453 sampled blocks agreed. (AX 39, the
  previous level: 5,071,318 blocks, 41,739,412,487,793 positions, 0 not solved.)

That gives P <= 33 + 38 = **71**. The GPU checkers (Metal, Apple silicon) were validated against the
independent CPU checkers `boundcheck` and `axbcheck`, with identical counts on the earlier levels UD 34,
AX 42 and an AX 41 sample. The boundary blocks are large (UD 33: 243 MB, AX 38: 7.9 GB, zstd-compressed)
and are not in this repository; the tools here regenerate and re-check them (see
[A boundary run](#a-boundary-run)). Pass `--bound` to `certsolve` and it chooses alternatives the same way
while solving.

## The certificate (`certificate/`)

Everything is plain text: one solving sequence per line, in standard notation (`2R'` = the second layer
from R, counter-clockwise). The files list only sequences, never positions; a line's position is the
sequence run backwards from solved. Each directory has a `MANIFEST` stating the stages, their sizes and
worst cases, and the rule for rebuilding the full tables.

| directory | what | size |
|---|---|---|
| `oblique/` | the four oblique pair stages (P = 80), small form: lines not one stage move away from a shorter class (3.9M of 28.9M lines); the full tables are rebuilt by a breadth-first pass | 206 MB |
| `mid/`, `diagonal/` | the mid and diagonal stages, concise form: a few kept lines; short ones serve as moves of a Dijkstra search that rebuilds every position within the stage worst | 240 KB, 72 KB |
| `wing/` | the wing algorithms (5,332, under symmetry); a Dijkstra search over each wing stage rebuilds it within the worst | 144 KB |

Corners (11, even n) and corners plus midges (20, odd n) need no certificate: they are the 3x3x3 corner
group's diameter (a full breadth-first search) and God's number for the 3x3x3.

## Building

```
make            # bin/certsolve and the CPU tools (C++20)
make gpu        # the Metal tools (macOS, Apple silicon)
make clean
```

## The solver (`solver/`)

`certsolve N` solves an N x N x N cube from the certificate alone, and checks every stage and the final
cube against the bound. Its command line follows the gullicube and brobdicube solvers'.

```
bin/certsolve 10 -M 5 --seed 1           # 10x10x10: five random positions, aggregate stats
bin/certsolve 10 --bound -M 5            # with boundary alternatives (budgets UD 33 + AX 39; --ax 38 for P = 71)
bin/certsolve 9 -R -o --verify           # a 450-move scramble; write the solution, check it independently
bin/certsolve 12 -i -o < scramble.txt    # scramble moves from stdin (-f: a Kociemba facelet string)
bin/certsolve --expand full              # write the four oblique stages in full (1.5 GB) to full/stages
```

With `-o`, every line of moves starts with a space and holds one set solved at once: an oblique pair's
first or second half, a mid, diagonal or wing orbit, or the corners. In order, the lines are the whole
solution; `-d` labels each with its stage lengths. `--verify` replays the scramble and the solution on
an independent sticker simulator (`solver/sim`). On odd n a scramble may turn the face centres; certsolve then
names each colour by the face its centre is on, so solved means every face one colour.

The rebuilt tables are cached in `~/.cache/certsolve` (`--cache DIR`, `--nopersist`; about 2 GB). The first
run builds them: about 10 minutes on a 16-core laptop (Apple M3 Max), plus about 18 minutes for the mid tables
the first time an odd n is solved, using about 5 GB of memory. Later runs start in seconds. Run
`bin/certsolve` with no arguments for every option. `--bound` also reads five algorithm pools (`pools/*.cls`,
7.7 MB). The cube model is brobdicube's (`solver/brob`).

## Checking the certificate

The checkers read the oblique stages in full, so expand them first:

```
bin/certsolve --expand full
bin/certcheck full/stages/none..aUbU --keep ud     # 1ud: PASS, at most 17 moves
bin/certcheck full/stages/aUbU..aDbD --keep ud     # 2ud: 18
bin/axcheck 3ax full/stages/3ax                    # 23
bin/axcheck 4op full/stages/4op                    # 22
pypy3 tools/udindcheck.py 1ud full/stages/none..aUbU   # the independent checks (also 2ud; axindcheck.py 3ax, 4op)
```

On the laptop above, expanding takes about 25 s after the tables are cached and the four checks about 90 s.

## A boundary run

```
make all gpu
./boundary-demo.sh work                    # UD 34, AX 42 and AX 41
./boundary-demo.sh work --ax 40 -t 16      # any levels: --ud X, --ax X, repeatable
```

For each level the script runs the GPU search (a broad pass, then passes with more slack on the classes
left), merges the blocks (`tools/mergeblocks.py`) and runs the GPU checker on them. Each level ends with
one line, `PASS: ...` or `FAIL: ...`; the blocks are deleted after a PASS. The default run takes about 15 minutes on the laptop above
(UD 34: 5,272 classes; AX 42: 245,171; AX 41: 1,073,807) and needs about 5 GB of disk. The levels behind P = 71 work the
same way but take much longer (AX 38: about 3 days of search and 10.6 hours of checking).

## Tools (`tools/`)

| tool | purpose |
|---|---|
| `certcheck`, `axcheck` | check the oblique stages (1ud/2ud; 3ax/4op): coverage, lengths, side effects on 6x6x6 to 10x10x10 |
| `udindcheck.py`, `axindcheck.py` | independent checks of the same, from twsearch puzzle files only (use pypy) |
| `pairdepth` | the pair sequences work for every pair (p, q) |
| `diagcheck`, `midcheck`, `wingcheck` | check diagonal, mid and wing stage lines on every size and depth (their coverage check needs the full forms, which are not included; certsolve rebuilds them in memory) |
| `boundx`, `boundcheck` | UD boundary search and check (CPU) |
| `axbx`, `axbcheck` | 3ax/4op boundary search and check (CPU) |
| `gpuboundx`, `gpubound`, `gpuaxbx`, `gpuaxbcheck` | the same on the GPU |
| `axbound`, `opbound` | build and lower the 3ax and 4op tables |
| `certextend`, `certimprove`, `certmerge` | grow, shorten and merge the 1ud/2ud stages |
| `mergeblocks.py` | merge the blocks of a multi-pass boundary search |
| `smallcert.py`, `colconcise` | write the small oblique and concise mid/diagonal forms |
| `cornerbfs` | the corner group's distances (11) |
| `rcube_worst.py` | worst case of the RCube solver's centre phase, for comparison |

## Comparison with published bounds

| n | best published upper bound | ours, P = 80 | ours, P = 71 |
|---|---|---|---|
| 4 | 53 SSTM (Shuang Chen, 2015) [1][2] | 217 | 217 |
| 5 | 130 OBTM, so <= 260 SSTM (Speedsolving forum, 2018) [3] | 265 | 265 |
| 6 and up | none found; lower bounds only (6x6x6: 72, 7x7x7: 95) [1][4] | 503, 590, ... | 494, 581, ... |
| large n | Theta(n^2 / log n), constants not given [5] | ~10 n^2 | ~8.9 n^2 |

For the 4x4x4 and 5x5x5, whole-cube reduction searches do far better than orbit-by-orbit solving.

For the centres alone, earlier explicit bounds are 36 n^2 + O(n) (a 2018 reddit comment [6]) and our own
36 n^2 - 144 n + 144 [7]; this work gives 10 n^2 - 38 n + 36 (P = 80) and about 8.9 n^2 (P = 71). Against
RCube's row-by-row centre phase (worst case by `tools/rcube_worst.py`), our centre bound is about 0.3 of
RCube's for n <= 20, 0.5 at n = 64 and 0.84 (P = 71) at n = 1000; both leading terms are 10 n^2 at P = 80.

## References

1. Speedsolving wiki, God's Algorithm: https://www.speedsolving.com/wiki/index.php?title=God%27s_Algorithm
2. W. Randelshofer, Virtual Cubes, history of 4x4x4 upper bounds:
   https://www.randelshofer.ch/rubik/virtual_cubes/revenge/instructions/4x_instructions_big.html
3. Speedsolving forum, "5x5x5 OBTM upper bound": https://www.speedsolving.com/threads/5x5x5-obtm-upper-bound.61270/
4. T. Rokicki, counting lower bounds for n x n x n (Domain of the Cube forum, 2011).
5. E. D. Demaine, M. L. Demaine, S. Eisenstat, A. Lubiw, A. Winslow, "Algorithms for Solving Rubik's Cubes",
   ESA 2011, https://arxiv.org/abs/1106.5736
6. r/Cubers, "I found a general formula for an upper bound of ..." (2018), comment giving 36 n^2 + O(n):
   https://old.reddit.com/r/Cubers/comments/8chfuu/i_found_a_general_formula_for_an_upper_bound_of/
7. Our n x n x n paper (to appear).
