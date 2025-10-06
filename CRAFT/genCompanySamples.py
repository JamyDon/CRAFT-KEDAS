import json
import akshare as ak
from tqdm import tqdm

all_companies = ak.stock_zcfz_em(date="20231231")['股票代码'].tolist()

selected_companies = all_companies[:1000]

# 获取所有报表数据
stock_zcfz_em_df_2023 = ak.stock_zcfz_em(date="20231231")
stock_zcfz_em_df_2024 = ak.stock_zcfz_em(date="20241231")
stock_lrb_em_df_2023 = ak.stock_lrb_em(date="20231231")
stock_lrb_em_df_2024 = ak.stock_lrb_em(date="20241231")
stock_xjll_em_df_2023 = ak.stock_xjll_em(date="20231231")
stock_xjll_em_df_2024 = ak.stock_xjll_em(date="20241231")
stock_zcfz_em_df_2022 = ak.stock_zcfz_em(date="20221231")
stock_zcfz_em_df_2021 = ak.stock_zcfz_em(date="20211231")
stock_lrb_em_df_2022 = ak.stock_lrb_em(date="20221231")
stock_lrb_em_df_2021 = ak.stock_lrb_em(date="20211231")
stock_xjll_em_df_2022 = ak.stock_xjll_em(date="20221231")
stock_xjll_em_df_2021 = ak.stock_xjll_em(date="20211231")

# 找出在所有报表中都存在的公司（且在2021、2022、2023和2024年都有数据）
common_companies = set(selected_companies)
for df in [stock_zcfz_em_df_2023, stock_zcfz_em_df_2024,
           stock_lrb_em_df_2023, stock_lrb_em_df_2024,
           stock_xjll_em_df_2023, stock_xjll_em_df_2024,
           stock_zcfz_em_df_2022, stock_zcfz_em_df_2021,
           stock_lrb_em_df_2022, stock_lrb_em_df_2021,
           stock_xjll_em_df_2022, stock_xjll_em_df_2021
           ]:
    common_companies.intersection_update(df['股票代码'].tolist())

final_companies = list(common_companies)[:400]

stock_zcfz_em_df_2023 = stock_zcfz_em_df_2023[stock_zcfz_em_df_2023['股票代码'].isin(final_companies)]
stock_zcfz_em_df_2024 = stock_zcfz_em_df_2024[stock_zcfz_em_df_2024['股票代码'].isin(final_companies)]
stock_lrb_em_df_2023 = stock_lrb_em_df_2023[stock_lrb_em_df_2023['股票代码'].isin(final_companies)]
stock_lrb_em_df_2024 = stock_lrb_em_df_2024[stock_lrb_em_df_2024['股票代码'].isin(final_companies)]
stock_xjll_em_df_2023 = stock_xjll_em_df_2023[stock_xjll_em_df_2023['股票代码'].isin(final_companies)]
stock_xjll_em_df_2024 = stock_xjll_em_df_2024[stock_xjll_em_df_2024['股票代码'].isin(final_companies)]
stock_zcfz_em_df_2022 = stock_zcfz_em_df_2022[stock_zcfz_em_df_2022['股票代码'].isin(final_companies)]
stock_zcfz_em_df_2021 = stock_zcfz_em_df_2021[stock_zcfz_em_df_2021['股票代码'].isin(final_companies)]
stock_lrb_em_df_2022 = stock_lrb_em_df_2022[stock_lrb_em_df_2022['股票代码'].isin(final_companies)]
stock_lrb_em_df_2021 = stock_lrb_em_df_2021[stock_lrb_em_df_2021['股票代码'].isin(final_companies)]
stock_xjll_em_df_2022 = stock_xjll_em_df_2022[stock_xjll_em_df_2022['股票代码'].isin(final_companies)]
stock_xjll_em_df_2021 = stock_xjll_em_df_2021[stock_xjll_em_df_2021['股票代码'].isin(final_companies)]

data = []
data_id = 0

for company in tqdm(final_companies):
    # 资产负债表
    zcfz_2023 = stock_zcfz_em_df_2023[stock_zcfz_em_df_2023['股票代码'] == company].iloc[0]
    zcfz_2024 = stock_zcfz_em_df_2024[stock_zcfz_em_df_2024['股票代码'] == company].iloc[0]
    zcfz_2022 = stock_zcfz_em_df_2022[stock_zcfz_em_df_2022['股票代码'] == company].iloc[0]
    zcfz_2021 = stock_zcfz_em_df_2021[stock_zcfz_em_df_2021['股票代码'] == company].iloc[0]

    # 利润表
    lrb_2023 = stock_lrb_em_df_2023[stock_lrb_em_df_2023['股票代码'] == company].iloc[0]
    lrb_2024 = stock_lrb_em_df_2024[stock_lrb_em_df_2024['股票代码'] == company].iloc[0]
    lrb_2022 = stock_lrb_em_df_2022[stock_lrb_em_df_2022['股票代码'] == company].iloc[0]
    lrb_2021 = stock_lrb_em_df_2021[stock_lrb_em_df_2021['股票代码'] == company].iloc[0]

    # 现金流量表
    xjll_2023 = stock_xjll_em_df_2023[stock_xjll_em_df_2023['股票代码'] == company].iloc[0]
    xjll_2024 = stock_xjll_em_df_2024[stock_xjll_em_df_2024['股票代码'] == company].iloc[0]
    xjll_2022 = stock_xjll_em_df_2022[stock_xjll_em_df_2022['股票代码'] == company].iloc[0]
    xjll_2021 = stock_xjll_em_df_2021[stock_xjll_em_df_2021['股票代码'] == company].iloc[0]

    name = zcfz_2023['股票简称']

    # 资产负债率数据
    debt2asset_ratio_2023 = zcfz_2023['资产负债率']
    debt2asset_ratio_2024 = zcfz_2024['资产负债率']
    total_assets_2023 = zcfz_2023['资产-总资产']
    total_assets_2024 = zcfz_2024['资产-总资产']
    total_assets_2022 = zcfz_2022['资产-总资产']
    total_assets_2021 = zcfz_2021['资产-总资产']
    total_assets_percent_2023 = zcfz_2023['资产-总资产同比']
    total_assets_percent_2024 = zcfz_2024['资产-总资产同比']
    total_debts_2023 = zcfz_2023['负债-总负债']
    total_debts_2024 = zcfz_2024['负债-总负债']
    total_debts_2022 = zcfz_2022['负债-总负债']
    total_debts_2021 = zcfz_2021['负债-总负债']
    total_debts_percent_2023 = zcfz_2023['负债-总负债同比']
    total_debts_percent_2024 = zcfz_2024['负债-总负债同比']
    total_stockholder_2023 = zcfz_2023['股东权益合计']
    total_stockholder_2024 = zcfz_2024['股东权益合计']

    # 利润表数据
    net_profit_percent_2023 = lrb_2023['净利润同比']
    net_profit_percent_2024 = lrb_2024['净利润同比']
    net_profit_2023 = lrb_2023['净利润']
    net_profit_2024 = lrb_2024['净利润']
    net_profit_2022 = lrb_2022['净利润']
    net_profit_2021 = lrb_2021['净利润']
    total_revenue_percent_2023 = lrb_2023['营业总收入同比']
    total_revenue_percent_2024 = lrb_2024['营业总收入同比']
    total_revenue_2023 = lrb_2023['营业总收入']
    total_revenue_2024 = lrb_2024['营业总收入']
    total_revenue_2022 = lrb_2022['营业总收入']
    total_revenue_2021 = lrb_2021['营业总收入']

    # 现金流量表数据
    operating_cash_flow_percent_2023 = xjll_2023['经营性现金流-净现金流占比']
    operating_cash_flow_percent_2024 = xjll_2024['经营性现金流-净现金流占比']
    operating_cash_flow_2023 = xjll_2023['经营性现金流-现金流量净额']
    operating_cash_flow_2024 = xjll_2024['经营性现金流-现金流量净额']
    net_cash_flow_2023 = xjll_2023['净现金流-净现金流']
    net_cash_flow_2024 = xjll_2024['净现金流-净现金流']
    net_cash_flow_2022 = xjll_2022['净现金流-净现金流']
    net_cash_flow_2021 = xjll_2021['净现金流-净现金流']
    investing_cash_flow_percent_2023 = xjll_2023['投资性现金流-净现金流占比']
    investing_cash_flow_percent_2024 = xjll_2024['投资性现金流-净现金流占比']
    financing_cash_flow_percent_2023 = xjll_2023['融资性现金流-净现金流占比']
    financing_cash_flow_percent_2024 = xjll_2024['融资性现金流-净现金流占比']
    investing_cash_flow_2023 = xjll_2023['投资性现金流-现金流量净额']
    investing_cash_flow_2024 = xjll_2024['投资性现金流-现金流量净额']
    financing_cash_flow_2023 = xjll_2023['融资性现金流-现金流量净额']
    financing_cash_flow_2024 = xjll_2024['融资性现金流-现金流量净额']
    net_cash_flow_percent_2023 = xjll_2023['净现金流-同比增长']
    net_cash_flow_percent_2024 = xjll_2024['净现金流-同比增长']




    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "资产负债率（%）",
        "indicator_alias": "Debt to asset ratio(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{debt2asset_ratio_2024:.1f}",
        "value2": f"{debt2asset_ratio_2023:.1f}",
        "prompt": [
            f"2024年{name}的总资产（亿元）是多少？",
            f"2024年{name}的总负债（亿元）是多少？"
        ],
        "target_new": [f"{total_assets_2024/ 100000000:.2f}",f"{total_debts_2024/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Total Assets(100 million yuan)是多少？",
                    f"2024年{name}的Total Liabilities(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{total_assets_2024/ 100000000:.2f}",f"{total_debts_2024/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的资产负债率（%）是多少？",
                "ground_truth": f"{debt2asset_ratio_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2023年{name}的总资产（亿元）是多少？",
                    f"2023年{name}的总负债（亿元）是多少？"
                ],
                "ground_truth": [f"{total_assets_2023 / 100000000:.2f}",f"{total_debts_2023/ 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "总资产同比（%）",
        "indicator_alias": "Total Assets Year-over-Year Growth(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{total_assets_percent_2024:.1f}",
        "value2": f"{total_assets_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的总资产（亿元）是多少？",
            f"2023年{name}的总资产（亿元）是多少？"
        ],
        "target_new": [f"{total_assets_2024/ 100000000:.2f}",f"{total_assets_2023/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Total Assets(100 million yuan)是多少？",
                    f"2023年{name}的Total Assets(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{total_assets_2024/ 100000000:.2f}",f"{total_assets_2023/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的总资产同比（%）是多少？",
                "ground_truth": f"{total_assets_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2022年{name}的总资产（亿元）是多少？",
                    f"2021年{name}的总资产（亿元）是多少？"
                ],
                "ground_truth": [f"{total_assets_2022 / 100000000:.2f}",f"{total_assets_2021 / 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "总负债同比（%）",
        "indicator_alias": "Total Debts Year-over-Year Growth(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{total_debts_percent_2024:.1f}",
        "value2": f"{total_debts_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的总负债（亿元）是多少？",
            f"2023年{name}的总负债（亿元）是多少？"
        ],
        "target_new": [f"{total_debts_2024/ 100000000:.2f}",f"{total_debts_2023/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Total Debts(100 million yuan)是多少？",
                    f"2023年{name}的Total Debts(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{total_debts_2024/ 100000000:.2f}",f"{total_debts_2023/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的总负债同比（%）是多少？",
                "ground_truth": f"{total_debts_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2022年{name}的总负债（亿元）是多少？",
                    f"2021年{name}的总负债（亿元）是多少？"
                ],
                "ground_truth": [f"{total_debts_2022 / 100000000:.2f}",f"{total_debts_2021 / 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "股东权益（亿元）",
        "indicator_alias": "Shareholders' equity(100 million yuan)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{total_stockholder_2024/ 100000000:.2f}",
        "value2": f"{total_stockholder_2023/ 100000000:.2f}",
        "prompt": [
            f"2024年{name}的总资产（亿元）是多少？",
            f"2024年{name}的总负债（亿元）是多少？"
        ],
        "target_new": [f"{total_assets_2024/ 100000000:.2f}",f"{total_debts_2024/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Total Assets(100 million yuan)是多少？",
                    f"2024年{name}的Total Liabilities(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{total_assets_2024/ 100000000:.2f}", f"{total_debts_2024/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的股东权益（亿元）是多少？",
                "ground_truth": f"{total_stockholder_2024/ 100000000:.2f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2023年{name}的总资产（亿元）是多少？",
                    f"2023年{name}的总负债（亿元）是多少？"
                ],
                "ground_truth": [f"{total_assets_2023/ 100000000:.2f}",f"{total_debts_2023/ 100000000:.2f}"],
            }
        },
    })


    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "净利润同比（%）",
        "indicator_alias": "Net Profit Year-over-Year Growth(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{net_profit_percent_2024:.1f}",
        "value2": f"{net_profit_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的净利润（亿元）是多少？",
            f"2023年{name}的净利润（亿元）是多少？"
        ],
        "target_new": [f"{net_profit_2024/ 100000000:.2f}",f"{net_profit_2023/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Net Profit(100 million yuan)是多少？",
                    f"2023年{name}的Net Profit(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{net_profit_2024/ 100000000:.2f}",f"{net_profit_2023/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的净利润同比（%）是多少？",
                "ground_truth": f"{net_profit_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2022年{name}的净利润（亿元）是多少？",
                    f"2021年{name}的净利润（亿元）是多少？"
                ],
                "ground_truth": [f"{net_profit_2022 / 100000000:.2f}",f"{net_profit_2021 / 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "营业总收入同比（%）",
        "indicator_alias": "Total Revenue Year-over-Year Growth(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{total_revenue_percent_2024:.1f}",
        "value2": f"{total_revenue_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的营业总收入（亿元）是多少？",
            f"2023年{name}的营业总收入（亿元）是多少？"
        ],
        "target_new": [f"{total_revenue_2024/ 100000000:.2f}",f"{total_revenue_2023/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Total Revenue(100 million yuan)是多少？",
                    f"2023年{name}的Total Revenue(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{total_revenue_2024/ 100000000:.2f}",f"{total_revenue_2023/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的营业总收入同比（%）是多少？",
                "ground_truth": f"{total_revenue_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2022年{name}的营业总收入（亿元）是多少？",
                    f"2021年{name}的营业总收入（亿元）是多少？"
                ],
                "ground_truth": [f"{total_revenue_2022 / 100000000:.2f}",f"{total_revenue_2021 / 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "净现金流同比增长（%）",
        "indicator_alias": "Net cash flow Year-over-Year Growth(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{net_cash_flow_percent_2024:.1f}",
        "value2": f"{net_cash_flow_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的净现金流（亿元）是多少？",
            f"2023年{name}的净现金流（亿元）是多少？"
        ],
        "target_new": [f"{net_cash_flow_2024/ 100000000:.2f}",f"{net_cash_flow_2023/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Net Cash Flow(100 million yuan)是多少？",
                    f"2023年{name}的Net Cash Flow(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{net_cash_flow_2024/ 100000000:.2f}",f"{net_cash_flow_2023/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的净现金流同比增长（%）是多少？",
                "ground_truth": f"{net_cash_flow_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2022年{name}的净现金流（亿元）是多少？",
                    f"2021年{name}的净现金流（亿元）是多少？"
                ],
                "ground_truth": [f"{net_cash_flow_2022 / 100000000:.2f}",f"{net_cash_flow_2021 / 100000000:.2f}"],
            }
        },
    })

    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "经营性现金流-净现金流占比（%）",
        "indicator_alias": "Cash Flow from Operating Activities Ratio(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{operating_cash_flow_percent_2024:.1f}",
        "value2": f"{operating_cash_flow_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的经营性现金流（亿元）是多少？",
            f"2024年{name}的净现金流（亿元）是多少？"
        ],
        "target_new": [f"{operating_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Cash Flow from Operating Activities(100 million yuan)是多少？",
                    f"2024年{name}的Net Cash Flow(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{operating_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的经营性现金流-净现金流占比（%）是多少？",
                "ground_truth":  f"{operating_cash_flow_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2023年{name}的经营性现金流（亿元）是多少？",
                    f"2023年{name}的净现金流（亿元）是多少？"
                ],
                "ground_truth": [f"{operating_cash_flow_2023/ 100000000:.2f}", f"{net_cash_flow_2023/ 100000000:.2f}"],
            }
        },
    })


    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "投资性现金流-净现金流占比（%）",
        "indicator_alias": "Cash Flow from Investing Activities Ratio(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{investing_cash_flow_percent_2024:.1f}",
        "value2": f"{investing_cash_flow_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的投资性现金流（亿元）是多少？",
            f"2024年{name}的净现金流（亿元）是多少？"
        ],
        "target_new": [f"{investing_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Cash Flow from Investing Activities(100 million yuan)是多少？",
                    f"2024年{name}的Net Cash Flow(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{investing_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的投资性现金流-净现金流占比（%）是多少？",
                "ground_truth": f"{investing_cash_flow_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2023年{name}的投资性现金流（亿元）是多少？",
                    f"2023年{name}的净现金流（亿元）是多少？"
                ],
                "ground_truth": [f"{investing_cash_flow_2023/ 100000000:.2f}", f"{net_cash_flow_2023/ 100000000:.2f}"],
            }
        },
    })


    data_id += 1
    data.append({
        "case_id": data_id,
        "entity": name,
        "indicator": "融资性现金流-净现金流占比（%）",
        "indicator_alias": "Cash Flow from Financing Activities Ratio(%)",
        "time1": "2024",
        "time2": "2023",
        "value1": f"{financing_cash_flow_percent_2024:.1f}",
        "value2": f"{financing_cash_flow_percent_2023:.1f}",
        "prompt": [
            f"2024年{name}的融资性现金流（亿元）是多少？",
            f"2024年{name}的净现金流（亿元）是多少？"
        ],
        "target_new": [f"{financing_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"],
        "portability": {
            "alias_portability_1": {
                "prompt": [
                    f"2024年{name}的Cash Flow from Financing Activities(100 million yuan)是多少？",
                    f"2024年{name}的Net Cash Flow(100 million yuan)是多少？"
                ],
                "ground_truth": [f"{financing_cash_flow_2024/ 100000000:.2f}", f"{net_cash_flow_2024/ 100000000:.2f}"]
            },
            "composite_portability": {
                "prompt": f"2024年{name}的融资性现金流-净现金流占比（%）是多少？",
                "ground_truth": f"{financing_cash_flow_percent_2024:.1f}"
            }
        },
        "locality": {
            "retention": {
                "prompt": [
                    f"2023年{name}的融资性现金流（亿元）是多少？",
                    f"2023年{name}的净现金流（亿元）是多少？"
                ],
                "ground_truth": [f"{financing_cash_flow_2023/ 100000000:.2f}", f"{net_cash_flow_2023/ 100000000:.2f}"],
            }
        },
    })

with open("company_samples.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)