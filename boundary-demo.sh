#!/bin/bash
# boundary-demo.sh WORKDIR [--ud X]... [--ax X]... [-t T]: boundary passes end to end, from the certificate
# alone, on the GPU (make all gpu first).  Default: UD 34, AX 42 and AX 41.
#
#  1. bin/certsolve --expand writes the four oblique stages in full (one line per class) to WORKDIR/full.
#  2. Per level, the search (gpuboundx for UD = 1ud + 2ud, gpuaxbx for AX = 3ax + 4op) runs a broad pass,
#     then passes with more slack and deeper alternatives on the classes still left; tools/mergeblocks.py
#     merges the passes' blocks.
#  3. The checker (gpubound, gpuaxbcheck) verifies the merged blocks against the full stages: every
#     position at the start of the first stage is solved through both stages in at most X moves.
# Each level ends with one assertion line, PASS or FAIL; the blocks are deleted after a PASS.
set -u
usage() { echo "usage: boundary-demo.sh WORKDIR [--ud X]... [--ax X]... [-t T]   (default: --ud 34 --ax 42 --ax 41)" >&2; exit 2; }
[ $# -ge 1 ] || usage
echo "# boundary-demo.sh $*"
W=$1; shift
UD=(); AX=(); T=8
while [ $# -gt 0 ]; do
  case $1 in
    --ud) [ $# -ge 2 ] || usage; UD+=("$2"); shift 2 ;;
    --ax) [ $# -ge 2 ] || usage; AX+=("$2"); shift 2 ;;
    -t) [ $# -ge 2 ] || usage; T=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done
[ ${#UD[@]} -eq 0 ] && [ ${#AX[@]} -eq 0 ] && UD=(34) && AX=(42 41)
R=$(cd "$(dirname "$0")" && pwd); B=$R/bin
for p in certsolve gpuboundx gpubound gpuaxbx gpuaxbcheck; do
  [ -x "$B/$p" ] || { echo "bin/$p missing: run make all gpu" >&2; exit 1; }
done
POOLS=""; for f in pair3 p8s10 p9 c12x eg8; do POOLS="$POOLS --base $R/pools/$f.cls"; done
mkdir -p "$W" || exit 1
W=$(cd "$W" && pwd); F=$W/full/stages
stamp() { date '+%H:%M:%S'; }

if [ ! -f "$W/full/done" ]; then
  echo "== $(stamp) expanding the oblique stages into $W/full"
  "$B/certsolve" --expand "$W/full" -t "$T" > "$W/expand.out" 2> "$W/expand.err" || { tail -3 "$W/expand.err"; exit 1; }
  grep "classes written" "$W/expand.err" | cut -d';' -f1
  touch "$W/full/done"
fi

# run one level: kind (ud|ax), X; the pass options in turn until no class is left
level() {
  local kind=$1 X=$2 A Bs search check name opts i L M n
  if [ "$kind" = ud ]; then
    name="UD $X"; A=$F/none..aUbU; Bs=$F/aUbU..aDbD; search=gpuboundx; check="gpubound"
    opts=("" "--deep 1 --deepprop 6" "--prop 10 --slack 6 --deep 2 --deepprop 6")
  else
    name="AX $X"; A=$F/3ax; Bs=$F/4op; search=gpuaxbx; check="gpuaxbcheck"
    opts=("--prop 8 --deep 2" "--prop 10 --slack 4 --deep 4 --deepprop 8" "--prop 10 --slack 4 --deep 4"
          "--prop 10 --slack 5 --deep 5" "--prop 11 --slack 6 --deep 6" "--prop 11 --slack 7 --deep 7 --deepprop 9")
  fi
  local D=$W/$kind$X; mkdir -p "$D"; rm -f "$D"/p*.txt "$D"/p*.left
  L=""; M=""; i=0
  for o in "${opts[@]}"; do
    i=$((i + 1))
    echo "== $(stamp) $name pass $i ${o:-(defaults)}${L:+ on $(wc -l < "$L" | tr -d ' ') classes}"
    "$B/$search" "$A" "$Bs" "$X" $o --threads "$T" ${L:+--only "$L"} --out "$D/p$i.txt" --left "$D/p$i.left" $POOLS \
      > "$D/p$i.out" 2> "$D/p$i.err" || { echo "$name: pass $i failed: $(tail -1 "$D/p$i.err")"; return 1; }
    tail -1 "$D/p$i.out"
    M="$M $D/p$i.txt $D/p$i.left"; L=$D/p$i.left
    [ -s "$L" ] || break
  done
  if [ -s "$L" ]; then echo "FAIL: $name: $(wc -l < "$L" | tr -d ' ') classes not covered after every pass"; return 1; fi
  python3 "$R/tools/mergeblocks.py" "$D/blocks.txt" $M > "$D/merge.out" 2>&1 || { echo "FAIL: $name: merge: $(tail -1 "$D/merge.out")"; return 1; }
  echo "== $(stamp) $name check ($(tail -1 "$D/merge.out" | cut -d' ' -f2-))"
  local keep=""; [ "$kind" = ud ] && keep="--keep ud"
  "$B/$check" "$A" "$Bs" "$D/blocks.txt" "$X" $keep --threads "$T" > "$D/check.out" 2> "$D/check.err"
  if grep -q "^PASS" "$D/check.out"; then
    echo "PASS: $name: $(grep '^PASS' "$D/check.out" | sed 's/^PASS: //')"
    rm -f "$D/blocks.txt" "$D"/p*.txt
  else
    echo "FAIL: $name: $(grep -v '^#' "$D/check.out" | tail -1)"; return 1
  fi
}

ok=0
for x in "${UD[@]+"${UD[@]}"}"; do level ud "$x" || ok=1; done
for x in "${AX[@]+"${AX[@]}"}"; do level ax "$x" || ok=1; done
echo "== $(stamp) $([ $ok -eq 0 ] && echo 'all levels PASS' || echo 'some level FAILED')"
exit $ok
