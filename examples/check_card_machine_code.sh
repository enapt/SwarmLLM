#!/usr/bin/env bash
# Fails unless every vendored CUDA kernel library this build made carries machine code for
# each generation in SWARMLLM_CUDA_MACHINE_CODE_ALSO, beside the floor's machine code and
# PTX (#326). Run by the Linux CUDA release and cache-warm builds right after `cargo build`:
# a library whose build script stopped passing the list would otherwise ship without the
# RTX 50-series code, and those cards would compile its PTX for minutes with nothing failing.
#
# Usage (from the workspace root, after a release build):
#   SWARMLLM_CUDA_MACHINE_CODE_ALSO=120 CUDA_COMPUTE_CAP=80 examples/check_card_machine_code.sh
set -euo pipefail

floor=${CUDA_COMPUTE_CAP:?CUDA_COMPUTE_CAP is the floor every library must carry}
also=${SWARMLLM_CUDA_MACHINE_CODE_ALSO:-}
cuobjdump=${CUDA_PATH:+$CUDA_PATH/bin/}cuobjdump

# The newest copy of each library: a restored cache can hold an older build directory.
# A directory that does not exist is no copy (find's failure would end the script under pipefail).
newest() { { find "$@" -name "$name" -printf '%T@ %p\n' 2>/dev/null || true; } | sort -n | tail -1 | cut -d' ' -f2-; }
libs=()
for name in libmoe.a libggml_mmq.a; do
  lib=$(newest target/release/build)
  [ -n "$lib" ] || { echo "no $name under target/release/build — run after the release build"; exit 1; }
  libs+=("$lib")
done
name=libflashattention.a
lib=$(newest "${CANDLE_FLASH_ATTN_BUILD_DIR:-.flash-attn-build}" target/release/build)
[ -n "$lib" ] || { echo "no $name — run after the release build"; exit 1; }
libs+=("$lib")

failed=0
for lib in "${libs[@]}"; do
  elf=$("$cuobjdump" --list-elf "$lib" | grep -o 'sm_[0-9]*[a-z]*' | sort -u | tr '\n' ' ')
  ptx=$("$cuobjdump" --list-ptx "$lib" | grep -o 'sm_[0-9]*[a-z]*' | sort -u | tr '\n' ' ')
  echo "$(basename "$lib"): machine code ${elf:-none}; PTX ${ptx:-none}"
  case " $ptx " in *" sm_$floor "*) ;; *) echo "  MISSING the floor's PTX (sm_$floor)"; failed=1 ;; esac
  case "$(basename "$lib")" in
    libggml_mmq.a)
      # Only the floor: with 12.x among MMQ's generations its host side configures a 12.x card
      # for Blackwell while a 12.1 card runs the floor's PTX (candle-kernels/build.rs).
      case " $elf " in *" sm_$floor "*) ;; *) echo "  MISSING machine code for sm_$floor"; failed=1 ;; esac
      case " $elf " in *" sm_12"*) echo "  carries 12.x machine code — unsafe for MMQ, see candle-kernels/build.rs"; failed=1 ;; esac
      ;;
    *)
      for cap in $floor $(echo "$also" | tr ',;' '  '); do
        case " $elf " in *" sm_$cap "*) ;; *) echo "  MISSING machine code for sm_$cap"; failed=1 ;; esac
      done
      ;;
  esac
done
exit $failed
