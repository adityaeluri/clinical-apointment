const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");

async function loadModel() {
  const response = await fetch("/model.json");
  if (!response.ok) {
    throw new Error(`Could not load model parameters (${response.status}).`);
  }
  return response.json();
}

function predict(input, model) {
  let logOdds = model.intercept;
  for (let index = 0; index < model.features.length; index += 1) {
    const feature = model.features[index];
    const value = input[feature] ?? model.imputation_values[index];
    const standardized = (value - model.means[index]) / model.scales[index];
    logOdds += standardized * model.coefficients[index];
  }
  return 1 / (1 + Math.exp(-logOdds));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.classList.remove("error");
  result.textContent = "Calculating…";

  try {
    const model = await loadModel();
    const values = new FormData(form);
    const input = {
      lead_days: Number(values.get("lead_days")),
      prior_missed: Number(values.get("prior_missed")),
      reminder_sent: Number(values.get("reminder_sent")),
    };
    const probability = predict(input, model);
    result.textContent = `Estimated missed-appointment score: ${(probability * 100).toFixed(1)}% (prediction at 50% threshold: ${probability >= 0.5 ? "missed" : "attended"}).`;
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
  }
});

fetch("/metrics.txt")
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Could not load evaluation metrics (${response.status}).`);
    }
    return response.text();
  })
  .then((metrics) => {
    document.querySelector("#metrics").textContent = metrics;
  })
  .catch((error) => {
    document.querySelector("#metrics").textContent = error.message;
  });
