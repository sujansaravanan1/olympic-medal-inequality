# Structural Inequality in 120 Years of Olympic Competition

This repository contains the data, code, and figures for the paper:

> Saravanan, S. (2026). *A multi-method data analysis framework for quantifying structural inequality in 120 years of Olympic competition* [Manuscript submitted for publication]. North Creek High School, Bothell, WA.

Every figure, table, and number reported in the paper is produced by a single notebook in this repository, working directly from the raw public data. Nothing in the paper was calculated by hand or copied from another source.

## About the study

The modern Olympic Games are built on the idea that talent can come from anywhere, yet the same small group of nations wins a disproportionate share of the medals at every Games. That gap has persisted for more than a century, even as the Games have grown from 176 athletes in 1896 to more than 11,000 in 2016. This study asks why medals remain so concentrated even as the Games have expanded, and it approaches the question from several directions using 271,116 athlete-event records from 1896 to 2016.

The analysis has five parts:

1. **Athlete archetypes.** K-means clustering groups 85,383 male Summer Games athletes into five body-type archetypes based on height, weight, and age, and tracks whether those archetypes have grown more or less distinct since 1960.
2. **How the Games have changed.** The share of athletes in each sport, the number of athletes from each continent, and the ratio of women to men in each sport category are traced across the history of the Games.
3. **Medal inequality.** Lorenz curves and Gini coefficients measure how unevenly medals are spread across nations, for every Summer Games.
4. **The role of national wealth.** A regression of 2016 medal counts on national GDP is extended to a negative binomial panel model of six Summer Games (1996 to 2016) that also accounts for population, host status, post-Soviet history, and each nation's previous medal count.
5. **Efficiency.** Two measures, medal success rate and medals per billion dollars of GDP, show which nations turn their athletes and resources into medals most effectively, and how the gap between the most and least efficient nations has changed by decade.

### Main findings

- The five athlete archetypes overlap substantially (mean silhouette coefficient 0.28), but their average heights and weights have drifted slightly further apart since 1960.
- Among medal-winning nations, the Gini coefficient of the 2016 medal distribution was 0.692, and it has stayed between 0.65 and 0.73 at every Summer Games since 1960.
- National GDP alone explains about 74% of the cross-national variation in 2016 medal counts.
- At the same level of GDP, host nations win about 2.5 times as many medals as they otherwise would, and nations of the former Soviet Union, Warsaw Pact, and Yugoslavia win about 5.2 times as many. A nation's previous medal count is the strongest single predictor of its next one.
- Small economies such as Jamaica and Cuba convert national income into medals at far higher rates than the largest economies.
- Participation has broadened across continents, and gender balance has moved toward parity in most sports, while medal concentration has not declined.

## Repository contents

| Path | Description |
|---|---|
| `olympic_inequality_analysis.ipynb` | The complete analysis, saved with all outputs so it can be read without running anything. |
| `olympic_inequality_analysis.py` | The same analysis as a plain Python script, in jupytext percent format. |
| `paper_numbers.json` | Every number reported in the paper, written automatically by the notebook. Each value can be checked against the text. |
| `figures/` | Figures 1 to 14 of the paper (`sarav1.png` to `sarav14.png`, 600 dpi) and the plot used to choose the number of clusters (`supp_cluster_selection.png`). |
| `data/` | The raw input data, described below. |
| `requirements.txt` | The Python packages required to run the analysis. |

The notebook follows the order of the paper. The table below shows where each result in the paper is produced.

| Paper section | Figures and tables |
|---|---|
| Athlete archetypes | Figures 1 to 3, Table 1 |
| Sport composition and globalization of participation | Figures 4 and 5 |
| Medal inequality | Figures 6 and 7 |
| GDP as a predictor of medal success | Figure 8 |
| Resource-adjusted efficiency | Figures 9 to 12, Table 2 |
| Multivariate model of medal counts | Figure 13, Table 3 |
| Gender participation | Figure 14 |

## Reproducing the results

The analysis was developed and tested with Python 3.13. To reproduce it, clone this repository and run the following from its root folder:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute olympic_inequality_analysis.ipynb --output olympic_inequality_analysis.ipynb
```

Alternatively, open the notebook in Jupyter and choose **Run All**. A full run takes about a minute on a typical computer.

The notebook reads its input from `data/` and writes the figures to `figures/` and the reported numbers to `paper_numbers.json`. To use data stored somewhere else, set the `OLYMPIC_DATA` environment variable to that folder before running.

The results are fully deterministic. K-means clustering and the silhouette sample both use a fixed random seed, so every run reproduces the same clusters and the same numbers. A fresh copy of this repository regenerates `paper_numbers.json` exactly.

## Data

All input data is publicly available. Copies are included in `data/`, unmodified, so that the analysis can be reproduced without locating the original files. All cleaning, filtering, and merging is done inside the notebook.

| File | Source | Contents |
|---|---|---|
| `athlete_events.csv` | Kaggle (2018), [120 years of Olympic history: athletes and results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) | One row per athlete per event from Athens 1896 to Rio 2016, including sex, age, height, weight, National Olympic Committee (NOC), sport, event, and medal. |
| `noc_regions.csv` | Same dataset | The country or region associated with each NOC code. |
| `gdp.csv` | World Bank, [World Development Indicators](https://data.worldbank.org), via [github.com/datasets/gdp](https://github.com/datasets/gdp) | Gross domestic product by country and year, in current US dollars. |
| `population.csv` | World Bank, [World Development Indicators](https://data.worldbank.org), via [github.com/datasets/population](https://github.com/datasets/population) | Total population by country and year. |

Credit for these datasets belongs to their original creators. World Bank data is distributed under the Creative Commons Attribution 4.0 license.

## Figures and fonts

Following the journal figure guidelines the paper was prepared for, figure text is set in Arial and subfigure labels in Times New Roman. When those fonts are not installed, the notebook uses metric-compatible substitutes (Liberation Sans and Liberation Serif) or, failing those, the DejaVu fonts bundled with matplotlib. A substitution changes only the typeface, never the data or the results.

## Use of AI

The AI system Claude (Anthropic) was used to help write and debug the code in this repository. The author reviewed all of the code, verified every reported result against the source data, and takes full responsibility for the analysis. The use of AI is also disclosed in the paper's acknowledgments.

## Citation

If you use this code or build on this analysis, please cite the paper:

> Saravanan, S. (2026). *A multi-method data analysis framework for quantifying structural inequality in 120 years of Olympic competition* [Manuscript submitted for publication]. North Creek High School, Bothell, WA.

## Contact

Questions, corrections, and suggestions are welcome. Please open an issue on this repository.
