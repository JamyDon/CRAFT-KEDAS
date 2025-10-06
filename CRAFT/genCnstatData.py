import json
import os
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from cnstats.stats import stats
import time

# -------------------------
# 配置区
# -------------------------
BASE_START = "202407"   # time1 起始（含）
BASE_END   = "202506"   # time1 结束（含）
DBCODE = "hgyd"
ENTITY = "中国"

# 指标类集合（zbcode）
ZB_CODES = [
    "A0D01", "A0C01", "A0C02", "A0B03", "A0A01", "A0A02",
    "A0A03", "A0A04", "A0A05", "A0901", "A0902", "A0903",
    "A0904", "A0906"
]

# 别名映射文件（第一列：中文指标名；第二列：英文别名）
ALIAS_FILE = "indicator_alias.xlsx"

# 输出文件
OUT_JSON = "cnstat_samples.json"

# 缓存目录
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# 工具函数
# -------------------------
def generate_month_list(start: str, end: str) -> List[str]:
    res = []
    sy, sm = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        res.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return res

def shift_year(yyyymm: str, delta_years: int) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:])
    return f"{y + delta_years}{m:02d}"

def human_date(yyyymm: str) -> str:
    return f"{int(yyyymm[:4])}年{int(yyyymm[4:])}月"

def to_number(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return None

def load_alias_map() -> Dict[str, str]:
    try:
        df = pd.read_excel(ALIAS_FILE, header=None)
        df[0] = df[0].astype(str).str.strip()
        df[1] = df[1].astype(str).str.strip()
        return dict(zip(df[0], df[1]))
    except Exception as e:
        print(f"[ERROR] Failed to load alias map: {e}")
        return {}

def fetch_zb_cache(zb: str, times_needed: List[str], max_retry: int = 3) -> Dict[str, List[List[str]]]:
    """抓取 zbcode 数据并缓存到本地，支持断点续抓"""
    cache_file = os.path.join(CACHE_DIR, f"{zb}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
            return {k: v for k, v in cache.items()}  # 确保 dict
    cache = {}
    for t in tqdm(sorted(times_needed), desc=f"Fetching {zb}"):
        for attempt in range(max_retry):
            try:
                result = stats(zbcode=zb, datestr=t, dbcode=DBCODE)
                if result is None:
                    result = []
                else:
                    result = [r for r in result if r[2] == t]#这里非常重要，因为这个sbAPI抓取空月份时会返回超长数据 比如抓202501会返回从202502到202512的所有数据
                cache[t] = result
                break
            except Exception as e:
                print(f"[WARN] Fetch error zb={zb} t={t} attempt {attempt+1}: {e}")
                time.sleep(1)
                if attempt == max_retry - 1:
                    cache[t] = []
    # 保存缓存文件
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return cache

# -------------------------
# 主流程
# -------------------------
def main():
    time1_list = generate_month_list(BASE_START, BASE_END)
    alias_map = load_alias_map()
    all_samples: List[dict] = []
    case_id = 1

    for zb in ZB_CODES:
        # 准备需要抓取的所有时间点
        times_needed = set()
        for t1 in time1_list:
            times_needed.add(t1)
            times_needed.add(shift_year(t1, -1))
            times_needed.add(shift_year(t1, +1))
            times_needed.add(shift_year(t1, -2))

        # 抓取并缓存
        cache = fetch_zb_cache(zb, list(times_needed))

        # 遍历每个 time1 生成样本
        for time1 in time1_list:
            rows_t1 = cache.get(time1, [])
            if not rows_t1:
                continue
            for main_row in rows_t1:
                if len(main_row) < 4:
                    continue
                indicator_cn_api = main_row[0].replace("_", "")
                sub_code = main_row[1]
                raw_value1 = main_row[3]

                # 同月跨年时间
                time2 = shift_year(time1, -1)
                time0 = shift_year(time1, +1)
                time3 = shift_year(time1, -2)

                def lookup_value(t: str) -> Optional[str]:
                    for r in cache.get(t, []):
                        if len(r) >= 4 and r[1] == sub_code:
                            return r[3]
                    return None

                raw_value2 = lookup_value(time2)
                raw_value0 = lookup_value(time0)
                raw_value3 = lookup_value(time3)
                if(raw_value1==None or raw_value2==None):
                    continue

                # 将原始值转换为保留两位小数的值，继续使用 value1, value2, value0, value3 名称
                value1 = round(to_number(raw_value1), 2) if raw_value1 is not None and to_number(raw_value1) is not None else None
                value2 = round(to_number(raw_value2), 2) if raw_value2 is not None and to_number(raw_value2) is not None else None
                value0 = round(to_number(raw_value0), 2) if raw_value0 is not None and to_number(raw_value0) is not None else None
                value3 = round(to_number(raw_value3), 2) if raw_value3 is not None and to_number(raw_value3) is not None else None

                # 匹配 Excel 指标名（API 指标名包含在 Excel 名称里）
                matched_cn = None
                for k in alias_map:
                    if indicator_cn_api in k:
                        matched_cn = k
                        break
                if matched_cn is None:
                    print(f"[SKIP] zb={zb}, time1={time1}, indicator={indicator_cn_api} 找不到匹配别名，已跳过")
                    continue
                indicator_cn = matched_cn
                indicator_alias = alias_map[matched_cn]
                # 组合问（time2 是否高于 time1）
                comp_answer=round(value1-value2, 2)
                # 生成样本
                sample = {
                    "case_id": str(case_id),
                    "entity": ENTITY,
                    "indicator": indicator_cn,
                    "indicator_alias": indicator_alias,
                    "time1": time1,
                    "time2": time2,
                    "time0": time0,
                    "time3": time3,
                    "value1": value1 if value1 else "未知",
                    "value2": value2 if value2 else "未知",
                    "prompt": [
                        f"{human_date(time1)}{ENTITY}的{indicator_cn}是多少？",
                        f"{human_date(time2)}{ENTITY}的{indicator_cn}是多少？"
                    ],
                    "target_new": [
                        value1 if value1 else "未知",
                        value2 if value2 else "未知"
                    ],
                    "portability": {
                        "alias_portability_1": {
                            "prompt": [
                                f"{human_date(time1)}{ENTITY}的{indicator_alias}是多少？",
                                f"{human_date(time2)}{ENTITY}的{indicator_alias}是多少？"
                            ],
                            "ground_truth": [
                                value1 if value1 else "未知",
                                value2 if value2 else "未知"
                            ]
                        },
                        "composite_portability": {
                            "prompt": f"{human_date(time1)}{ENTITY}的{indicator_cn}比{human_date(time2)}的高多少？",
                            "ground_truth": comp_answer
                        }
                    },
                    "locality": {
                        "retention": {
                            "prompt": [
                                f"{human_date(time3)}{ENTITY}的{indicator_cn}是？",
                                f"{human_date(time0)}{ENTITY}的{indicator_cn}是？"
                            ],
                            "ground_truth": [
                                value3 if value3 else "未知",
                                value0 if value0 else "未知"
                            ]
                        }
                    }
                }
                all_samples.append(sample)
                case_id += 1

    # 保存最终 sample
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"✅ 共生成 {len(all_samples)} 条样本，已保存到 {OUT_JSON}")

if __name__ == "__main__":
    main()
