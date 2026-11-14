# Structural Inequality in 120 Years of Olympic Competition

Code and data for the paper **"A Multi-Method Data Analysis Framework for Quantifying Structural Inequality in 120 Years of Olympic Competition"** by Sujan Saravanan (North Creek High School, Bothell, WA).

The notebook regenerates every figure, table, and number reported in the paper from the raw data, top to bottom.

## Contents

| Path | What it is |
|---|---|
| `olympic_inequality_analysis.ipynb` | The full analysis, executed, with all outputs |
| `olympic_inequality_analysis.py` | The same notebook as a plain Python script (jupytext percent format) |
| `paper_numbers.json` | Every number quoted in the paper, written by the notebook |
| `figures/` | Figures 1 to 14 (`sarav1.png` to `sarav14.png`) and the cluster-selection plot |
| `data/` | Input data (see below) |
| `requirements.txt` | Python packages |

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute olympic_inequality_analysis.ipynb --output olympic_inequality_analysis.ipynb
```

The notebook reads from `data/` by default (set the `OLYMPIC_DATA` environment variable to use another folder) and writes figures to `figures/`. K-means uses a fixed random seed, so results are deterministic.

## Data sources

- `data/athlete_events.csv`, `data/noc_regions.csv`: "120 years of Olympic history: athletes and results" (Kaggle, 2018), https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
- `data/gdp.csv`, `data/population.csv`: World Bank World Development Indicators (GDP in current US$, total population), https://data.worldbank.org, via the mirrors at https://github.com/datasets/gdp and https://github.com/datasets/population

The data files are redistributed here unchanged for reproducibility; credit belongs to the original sources above.

## Note on AI assistance

The AI system Claude (Anthropic) was used to help write and debug this code. The author reviewed the code and verified all results against the source data.

## Fonts

Figures use Arial (or Liberation Sans) for text and Times New Roman (or Liberation Serif) for subfigure labels. If none of these are installed, the notebook falls back to DejaVu Sans / DejaVu Serif (bundled with matplotlib) without warnings; only the typeface changes, not the results.
