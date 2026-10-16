CMCC A* 每日航道可通行性评估

脚本只输出每一天、每个 SIC 阈值下的 A* 判定，不计算通航率或全年统计。
阈值默认：0.15、0.30、0.70。

降尺度前：CMCC 25 km SIC
降尺度后：TOS+PR 模型输出的 6.25 km SIC

服务器运行前先做单年 smoke test：

python evaluate_cmcc_astar_daily.py \
  --years 2050 \
  --max-days 2 \
  --output-csv local-path/cmcc_astar_smoke.csv

确认路径和变量识别正常后，运行全年：

nohup python -u evaluate_cmcc_astar_daily.py \
  --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045 2046 2047 2048 2049 2050 \
  --output-csv local-path/data_cmcc/astar_daily_passability_2015_2050.csv \
  > local-path/data_cmcc/astar_daily_passability_2015_2050.log 2>&1 &

输入路径若与默认命名不同，可通过 --before-template 和 --after-template 指定，模板中必须包含 {year}。
