# Structural Inequality in 120 Years of Olympic Competition

This repository holds the code, data, and figures behind my paper, **"A Multi-Method Data Analysis Framework for Quantifying Structural Inequality in 120 Years of Olympic Competition."** I'm Sujan Saravanan, a student at North Creek High School in Bothell, Washington.

The project started with a simple question: if the Olympics keep adding more sports, more nations, and more athletes, why do the medals still end up with roughly the same small group of countries? To get at that, the analysis looks at the problem from a few different angles. It clusters athletes by body type to see how specialized sports have become, measures how unevenly medals are spread across nations with Lorenz curves and Gini coefficients, tests how much of a country's medal count comes down to its economy, and checks which nations get the most out of the resources they have.

Everything in the paper comes out of one notebook. If you run it from top to bottom, it rebuilds every figure, every table, and every number I quote, straight from the raw data. Nothing was typed in by hand.

## What's in here

| File or folder | What it is |
|---|---|
| `olympic_inequality_analysis.ipynb` | The full analysis. It's already been run, so you can scroll through the outputs and charts without installing anything. |
| `olympic_inequality_analysis.py` | The exact same notebook saved as a plain Python script, in case you'd rather read or diff it that way. |
| `paper_numbers.json` | Every number that appears in the paper, written out automatically by the notebook. If you want to check a figure from the text, this is the fastest place to look. |
| `figures/` | Figures 1 through 14 from the paper (`sarav1.png` to `sarav14.png`), plus the plot I used to choose the number of athlete clusters. |
| `data/` | The raw input data. More on where it came from below. |
| `requirements.txt` | The Python packages the notebook needs. |

## Running it yourself

You'll need Python 3 and a few common data science libraries. From the main folder of this repository, run:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute olympic_inequality_analysis.ipynb --output olympic_inequality_analysis.ipynb
```

Or just open the notebook in Jupyter and run all the cells. On a normal laptop it takes a few minutes, and most of that time goes into the clustering step.

The notebook looks for the data in the `data/` folder, so you shouldn't need to change anything. If you keep the data somewhere else, set the `OLYMPIC_DATA` environment variable to that folder before you run it. The figures get saved into `figures/`.

You should get exactly the same results I did. The clustering step uses a fixed random seed, so it splits the athletes the same way every time instead of changing slightly from run to run.

## Where the data comes from

None of this data is mine. I'm including copies here so anyone can reproduce the results without hunting for the exact same files, but all the credit goes to the original sources:

- **Athlete records** (`data/athlete_events.csv` and `data/noc_regions.csv`) come from the "120 years of Olympic history: athletes and results" dataset on Kaggle (2018). It covers every athlete-event entry from the 1896 Athens Games through Rio 2016, with each athlete's height, weight, age, country, sport, and medal. You can find it at https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results.
- **Economic data** (`data/gdp.csv` and `data/population.csv`) comes from the World Bank's World Development Indicators, which give each country's GDP in current US dollars and its total population. The original source is https://data.worldbank.org, and I downloaded these copies from the mirrors at https://github.com/datasets/gdp and https://github.com/datasets/population.

The files haven't been edited. All the cleaning and filtering happens inside the notebook, so you can see every step.

## A note on fonts

The figures use Arial for regular text and Times New Roman for the small (a), (b), and (c) panel labels, since those are the fonts the journal guidelines asked for. Not every computer has them, though. If yours doesn't, the notebook quietly switches to a lookalike (Liberation Sans or Liberation Serif) or to the DejaVu fonts that come with matplotlib. The charts will look a little different, but the data and results stay exactly the same.

## How AI was used

I used Claude, an AI assistant made by Anthropic, to help write and debug the code in this project. I reviewed all of the code myself and checked every result against the source data, so any mistakes are my responsibility. The paper has a full disclosure in its acknowledgments section.

## Questions

If something doesn't run, or a number doesn't match the paper, please open an issue on this repository and I'll take a look.
