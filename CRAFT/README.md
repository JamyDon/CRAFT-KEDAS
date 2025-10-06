# Construction of CRAFT

## Prerequisites
```bash
pip install cn-stats
pip install akshare
```

## Quick Start
```bash
bash run.sh
```
This will execute the following scripts in sequence:

```
python genCnstatData.py
python genCompanySamples.py
python KnowEditformat.py
python split.py
```

## The Finance Subset

This program extracts financial data for Chinese companies from multiple years and generates a structured JSON output.

### Usage

 1. Running the Script
To generate the sample data, simply run:
```bash
python company_samples.py
```

This will create a `company_samples.json` file containing financial data for companies.

 2. Customizing Data Years
To change the years of data being collected:

1. Modify these variables in the script:
```python
# Current year settings (modify these):
stock_zcfz_em_df_2023 = ak.stock_zcfz_em(date="20231231")  # Change "20231231" to your desired year
stock_zcfz_em_df_2024 = ak.stock_zcfz_em(date="20241231")  # Change "20241231" to your desired year
# ... (similar for other variables)
```

4. Update all year references throughout the script (2021, 2022, 2023, 2024) to match your desired years.


### Notes
- The script uses the `akshare` package to fetch financial data
- Data is filtered to include only companies with complete data across all selected years
- By default, it processes the first 400 companies with complete data

For any issues with data fetching, check your internet connection or try again later.

## The Statistics Subset
### **genCnstatData.py**

Fetch data from National Bureau of Statistics of China.

```
BASE_START = "202407"   # time1 start
BASE_END   = "202506"   # time1 end
```

Set the start time and end time, recommended for the past year.

```
# Set of indicator categories (zbcode)
ZB_CODES = [
    "A0901",  # 货运量_当期值
    # "A0D01", "A0C01", "A0C02", "A0B03", "A0A01", "A0A02",
    # "A0A03", "A0A04", "A0A05", "A0901", "A0902", "A0903",
    # "A0904", "A0906"
]
```

Supported indicator categories:

| A0D01 | 货币供应量 |
| ----- | ---------- |

| A0C01 | 国家财政预算收入 |
| ----- | ---------------- |
| A0C02 | 国家财政预算支出 |

| A0B03 | 综合**PMI**产出指数 |
| ----- | ------------------- |

| A0A01 | 邮电业务总量 |
| ----- | ------------ |
| A0A02 | 邮电业务收入 |
| A0A03 | 邮政业务量   |
| A0A04 | 电信业务量   |
| A0A05 | 软件业务收入 |

| A0901 | 货物运输量                      |
| ----- | ------------------------------- |
| A0902 | 货物周转量                      |
| A0903 | 旅客运输量                      |
| A0904 | 旅客周转量                      |
| A0906 | 全国港口货物吞吐量（**2019-**） |

### **KnowEditformat.py**

Convert data to the final format.

### **split.py**

Split the dataset into training and testing sets.


### **indicator_alias.xlsx**

Indicator names and their aliases, the English names of all indicators in the above categories, manually verified.

### **c3dataset**

[https://github.com/nlpdata/c3/](https://github.com/nlpdata/c3/)

free-form multiple-Choice Chinese machine reading Comprehension dataset (C^3), containing 13,369 documents (dialogues or more formally written mixed-genre texts) and their associated 19,577 multiple-choice free-form questions collected from Chinese-as-a-second-language examinations.

Sample common sense QA from c3 dataset.