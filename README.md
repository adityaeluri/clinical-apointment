# clinical-apointment

An educational appointment attendance exercise using synthetic data. The Python script cleans the data, summarizes reminder outcomes with SQL, trains and evaluates a logistic regression model, and scores a sample appointment.

## Run locally

Install the requirements and run the script:

```sh
python3 -m pip install -r requirements.txt
python3 project.py
```

The script writes model metrics and a chart to `outputs/`, and exports the trained model parameters and report assets into `site/`.

## Deploy to Netlify

The `site/` directory contains the interactive static website. Netlify uses `netlify.toml` to install the Python requirements, run the model script, and publish the generated site.

To deploy, push this repository to GitHub, then import that repository in Netlify. No server-side prediction function is needed; the static site calculates scores in the browser using the parameters generated at build time.

The dataset and scores are synthetic and for educational demonstration only; do not use them for clinical or operational decisions.
