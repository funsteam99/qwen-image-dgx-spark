#!/bin/bash
# ComfyUI 產物清理：預設為預演 (dry-run)，只有加上 --apply 才會真的刪除。
#
#   ./cleanup_dgx.sh                      # 預演，用預設天數
#   ./cleanup_dgx.sh --input-days 3       # 預演，input 只留 3 天
#   ./cleanup_dgx.sh --apply              # 實際刪除
#   ./cleanup_dgx.sh --output-days 0 --apply   # output 全部保留 (0 = 不刪)
set -uo pipefail

ROOT="${COMFYUI_ROOT:-$HOME/qwen-image/ComfyUI}"
OUT_DIR="$ROOT/output"
IN_DIR="$ROOT/input"

OUTPUT_DAYS=30      # output 保留天數；0 表示永不刪除
INPUT_DAYS=7        # input 上傳暫存保留天數；0 表示永不刪除
APPLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --output-days) OUTPUT_DAYS="$2"; shift 2 ;;
    --input-days)  INPUT_DAYS="$2";  shift 2 ;;
    --apply)       APPLY=1; shift ;;
    -h|--help)     sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "未知參數: $1" >&2; exit 2 ;;
  esac
done

for d in "$OUT_DIR" "$IN_DIR"; do
  [ -d "$d" ] || { echo "[ERROR] 找不到目錄: $d" >&2; exit 1; }
done

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0}B"; }

sweep() {
  local dir="$1" days="$2" label="$3"
  echo "--- $label : $dir"
  local total count
  total=$(du -sb "$dir" 2>/dev/null | cut -f1)
  count=$(find "$dir" -type f | wc -l)
  echo "    現況        : $count 個檔案, $(human "$total")"

  if [ "$days" -eq 0 ]; then
    echo "    保留天數 0  : 跳過，不刪除任何檔案"
    return
  fi

  # -mtime +N 為「修改時間早於 N 天前」
  mapfile -t victims < <(find "$dir" -type f -mtime "+$days" -print)
  if [ "${#victims[@]}" -eq 0 ]; then
    echo "    超過 $days 天  : 無，不需清理"
    return
  fi

  local freed=0 sz
  for f in "${victims[@]}"; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    freed=$((freed + sz))
  done
  echo "    超過 $days 天  : ${#victims[@]} 個檔案, 可釋放 $(human "$freed")"

  if [ "$APPLY" -eq 1 ]; then
    printf '%s\0' "${victims[@]}" | xargs -0 rm -f --
    echo "    [已刪除] 釋放 $(human "$freed")"
  else
    echo "    [預演] 未刪除。最舊 3 筆："
    printf '        %s\n' "${victims[@]:0:3}" | sed "s#$dir/##"
  fi
}

echo "==================================================="
echo " ComfyUI 產物清理  $( [ "$APPLY" -eq 1 ] && echo '【實際執行】' || echo '【預演模式 — 不會刪除】')"
echo "==================================================="
df -h "$ROOT" | tail -1 | awk '{print " 磁碟: 已用 "$3" / "$2"  可用 "$4"  ("$5")"}'
echo
sweep "$OUT_DIR" "$OUTPUT_DAYS" "輸出圖片 (保留 ${OUTPUT_DAYS} 天)"
echo
sweep "$IN_DIR"  "$INPUT_DAYS"  "上傳暫存 (保留 ${INPUT_DAYS} 天)"
echo
if [ "$APPLY" -eq 0 ]; then
  echo "要實際刪除請加上 --apply"
else
  df -h "$ROOT" | tail -1 | awk '{print " 清理後磁碟: 可用 "$4"  ("$5")"}'
fi
