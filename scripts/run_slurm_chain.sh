#!/bin/bash
### Run AthenaK as a chain of SLURM jobs ("segments").
###
### Each segment runs AthenaK with -t set to the time left in the job minus MARGIN.
### When that runs out AthenaK stops cleanly, writes all outputs and a restart file,
### and this script resubmits itself. The next segment restarts from the newest
### restart file in rst/.
###
### The outputs written at the wall-clock stop are off schedule. The next segment
### undoes them with command-line overrides read from the restart file:
###   output#/last_time   = stored - dt  for every output with dt > 0
###   output#/file_number = stored - 1   for outputs that number their files
###                                      (all but rst, hst, log and trk)
### The next scheduled output therefore falls on the original time grid and
### overwrites the off-schedule file. The restart file keeps its number, so the
### file a segment restarts from is never overwritten. The extra hst row stays.
###
### Usage: edit the settings below, then run "sbatch run_slurm_chain.sh" from the
### run directory. Without rst/ there, the run starts from INPUT. Otherwise it
### continues from the newest restart file only if that file was written at one of
### the chain's own wall-clock stops (recorded in .chain_restart), so the same
### command also continues a chain that reached MAX_SEGMENTS. Any other newest
### restart file (after a crash, a killed job or a finished run) is refused; check
### it, then run "sbatch run_slurm_chain.sh resume" to restart from it without
### schedule fixes. Set the segment length only with the --time line. Each segment
### resubmits the script as first submitted; to change it mid-run, scancel the
### pending segment, edit, and sbatch again.
### The only SLURM command used is sbatch, for the resubmission.
###
### AthenaK prints its run summary (zone-cycles/cpu_second) only after it has written
### every output. Some MPI libraries then hang in MPI_Finalize: if AthenaK has not exited
### EXIT_GRACE seconds after the summary, the script stops it and goes on as if it had.
### SLURM mails on FAIL and on TIME_LIMIT. A segment reaches its time limit only if
### MARGIN is too short; check its newest restart file as above before continuing.

#SBATCH --job-name=Higher-Subsonic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=0-01:00:00        # length of ONE segment ([D-]HH:MM:SS)
#SBATCH --output=athena-%j.log   # this script's and AthenaK's messages; the script reads it
#SBATCH --dependency=singleton   # one job with this name at a time: give each run its own name
#SBATCH --mail-type=FAIL,TIME_LIMIT   # a segment failed or ran out of time
#SBATCH --mail-user=yoonsoo@princeton.edu

ATHENA=/scratch/gpfs/FPRETORI/yk8311/Magnus-GPU/athena
INPUT=magnus.input
LAUNCH=(srun -n 8)                              # MPI launcher
ATHENA_ARGS=(--kokkos-map-device-id-by=mpi_rank) # extra arguments for every segment
REREAD_INPUT=1     # 1: restarts also read INPUT (-r ... -i ...), so edits to it apply
MARGIN=300         # seconds kept for the last cycle, all outputs, and EXIT_GRACE + 35
EXIT_GRACE=60      # seconds AthenaK may take to exit after printing its run summary
MAX_SEGMENTS=50    # stop resubmitting after this many segments

module purge
module load nvhpc/24.11 openmpi/cuda-12.6/nvhpc-24.11/4.1.6 cudatoolkit/11.8

#----------------------------------------------------------------------------------------
# Print "block name value" for every parameter of an input file, or of the parameter
# header at the start of a restart file

list_params() {
  awk '/^<par_end>/ { exit }
       /^[[:space:]]*#/ { next }
       match($0, /^[[:space:]]*<[^>]*>/) {
         blk = substr($0, RSTART, RLENGTH); gsub(/[[:space:]<>]/, "", blk); next
       }
       index($0, "=") {
         i = index($0, "="); k = substr($0, 1, i - 1); v = substr($0, i + 1)
         sub(/#.*/, "", v); gsub(/[[:space:]]/, "", k); gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
         print blk, k, v
       }' "$1"
}

#----------------------------------------------------------------------------------------
# Print the newest restart file, in either the single-file or the per-rank layout

latest_restart() {
  ls rst/"$BASENAME".*.rst rst/rank_00000000/"$BASENAME".*.rst 2>/dev/null |
    awk -F. '{ print $(NF-1), $0 }' | sort -n | tail -n 1 | cut -d' ' -f2-
}

#----------------------------------------------------------------------------------------
# Print the overrides that undo the off-schedule outputs written at a wall-clock stop

schedule_fixes() {
  list_params "$1" | awk '
    $1 ~ /^output/ { p[$1, $2] = $3; blk[$1] = 1 }   # as AthenaK, any block named output*
    END {
      for (b in blk) {
        cyc = ((b, "dcycle") in p)
        dt = cyc ? 0 : p[b, "dt"] + 0
        if (!cyc && dt <= 0) continue                 # AthenaK skips this output
        type = p[b, "file_type"]
        if (dt > 0 && p[b, "last_time"] + 0 >= 0)
          printf "%s/last_time=%.15g\n", b, p[b, "last_time"] - dt
        if (type != "rst" && type != "hst" && type != "log" && type != "trk" &&
            p[b, "file_number"] + 0 > 0)
          printf "%s/file_number=%d\n", b, p[b, "file_number"] - 1
      }
    }'
}

#----------------------------------------------------------------------------------------
# Print the seconds left before SLURM ends this job: from SLURM_JOB_END_TIME where
# SLURM sets it, else from this script's "#SBATCH --time=[D-]HH:MM:SS" line minus the
# time the script has been running

time_left() {
  if [[ -n $SLURM_JOB_END_TIME ]]; then
    echo $((SLURM_JOB_END_TIME - $(date +%s)))
    return
  fi
  local t d=0 h m s
  t=$(sed -n 's/^#SBATCH[[:space:]]*--time=\([0-9:-]*\).*/\1/p' "$0" | head -n 1)
  [[ $t =~ ^([0-9]+-)?[0-9]+:[0-9]+:[0-9]+$ ]] || return 1
  if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
  IFS=: read -r h m s <<< "$t"
  echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s - SECONDS ))
}

#----------------------------------------------------------------------------------------

if [[ -z $SLURM_JOB_ID ]]; then
  echo "Submit this script with sbatch"
  exit 1
fi
cd "$SLURM_SUBMIT_DIR" || exit 1
RESUME=0
if [[ $1 == resume ]]; then RESUME=1; SEGMENT=1; else SEGMENT=${1:-1}; fi
STATE=.chain_restart          # names the restart file written at the last wall-clock stop
LOG=athena-$SLURM_JOB_ID.log  # this job's output file, as named by --output
if [[ ! -f $LOG ]]; then
  echo "Keep #SBATCH --output=athena-%j.log: the script reads AthenaK's messages from it"
  exit 1
fi
echo "$(date): segment $SEGMENT, job $SLURM_JOB_ID"

BASENAME=$(list_params "$INPUT" | awk '$1 == "job" && $2 == "basename" { print $3; exit }')
if [[ -z $BASENAME ]]; then
  echo "No <job>/basename found in $INPUT"
  exit 1
fi

# Restart from the newest restart file, adding the schedule fixes when that file was
# written at a wall-clock stop. Any other restart file needs "resume". Start from
# INPUT only when rst/ does not exist yet.
rst=$(latest_restart)
if [[ -n $rst ]]; then
  args=(-r "$rst")
  (( REREAD_INPUT )) && args+=(-i "$INPUT")
  if [[ -f $STATE && $(< "$STATE") == "$rst" ]]; then
    while read -r fix; do args+=("$fix"); done < <(schedule_fixes "$rst")
  elif (( ! RESUME )); then
    echo "The newest restart file, $rst, was not written at a wall-clock stop of"
    echo "this chain: the run may have finished, crashed, or been killed while writing"
    echo "it. Check it (delete it if it is incomplete), then submit this script with"
    echo "the argument \"resume\" to restart from the newest restart file."
    exit 1
  fi
elif [[ -d rst ]]; then
  echo "rst/ exists but holds no $BASENAME restart files; remove rst/ to start over"
  exit 1
else
  rm -f "$STATE"
  args=(-i "$INPUT")
fi

if ! left=$(time_left); then
  echo "Could not read the time limit: set it as #SBATCH --time=[D-]HH:MM:SS"
  exit 1
fi
run=$((left - MARGIN))
if (( run <= 0 )); then
  echo "Only ${left}s left in the job, which is not more than MARGIN=${MARGIN}s"
  exit 1
fi
walltime=$(printf '%d:%02d:%02d' $((run / 3600)) $((run % 3600 / 60)) $((run % 60)))

echo "${LAUNCH[*]} $ATHENA ${ATHENA_ARGS[*]} ${args[*]} -t $walltime"
# AthenaK prints to LOG too. Search only what it prints from here on, in case LOG
# still holds an earlier attempt of a requeued job.
start=$(( $(wc -c < "$LOG") + 1 ))
"${LAUNCH[@]}" "$ATHENA" "${ATHENA_ARGS[@]}" "${args[@]}" -t "$walltime" &
pid=$!
# Wait for AthenaK to exit. Once it has printed its run summary all outputs are written,
# so if it then hangs on exit, stop it after EXIT_GRACE seconds and carry on.
summary_at=
stopped=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 5
  if [[ -z $summary_at ]]; then
    tail -c +"$start" "$LOG" | grep -q '^zone-cycles/cpu_second' && summary_at=$SECONDS
  elif (( SECONDS - summary_at >= EXIT_GRACE )); then
    echo "$(date): AthenaK did not exit ${EXIT_GRACE}s after its run summary; stopping it"
    kill -TERM "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6; do sleep 5; kill -0 "$pid" 2>/dev/null || break; done
    kill -KILL "$pid" 2>/dev/null
    stopped=1
    break
  fi
done
wait "$pid"
status=$?
(( stopped )) && status=0

if (( status != 0 )); then
  echo "AthenaK exited with status $status; not resubmitting"
  exit "$status"
fi
if ! tail -c +"$start" "$LOG" | grep -q 'Terminating on wall clock limit'; then
  echo "Run finished; not resubmitting"
  rm -f "$STATE"
  exit 0
fi

latest_restart > "$STATE"
echo "$(date): stopped on the wall clock and wrote $(< "$STATE")"
# AthenaK also reports a wall-clock stop when the loop never ran: setup took longer
# than -t, or ncycle was already past nlim. Resubmitting would repeat that forever.
if tail -c +"$start" "$LOG" | grep -q '^MeshBlock-cycles = 0$'; then
  echo "AthenaK stopped without taking a step; not resubmitting"
  exit 1
fi
if (( SEGMENT >= MAX_SEGMENTS )); then
  echo "Reached MAX_SEGMENTS=$MAX_SEGMENTS; not resubmitting"
  exit 0
fi

# $0 is SLURM's copy of the script as it was submitted
if ! sbatch "$0" $((SEGMENT + 1)); then
  echo "Resubmission failed; submit this script again from $PWD to continue"
  exit 1
fi
