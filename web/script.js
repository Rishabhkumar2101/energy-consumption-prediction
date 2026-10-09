
const API_URL = "http://127.0.0.1:8000";

const sample = {
    lights: 0,
    T1: 24.2,
    RH_1: 39.79,
    T2: 22.5,
    RH_2: 41.09,
    T3: 25.76,
    RH_3: 39.0,
    T4: 24.1,
    RH_4: 38.1633333333,
    T5: 22.29,
    RH_5: 49.79,
    T6: 14.04,
    RH_6: 29.874,
    T7: 23.83,
    RH_7: 41.536,
    T8: 24.5,
    RH_8: 44.6633333333,
    T9: 22.6,
    RH_9: 45.678,
    T_out: 13.7333333333,
    Press_mm_hg: 751.3833333333,
    RH_out: 81.1666666667,
    Windspeed: 1,
    Visibility: 28.3333333333,
    Tdewpoint: 10.5333333333,
    rv1: 20.1062472304,
    rv2: 20.1062472304,
    hour: 5,
    day_of_week: 6,
    month: 5,
    is_weekend: 1,
    hour_sin: 0.9659258263,
    hour_cos: 0.2588190451,
    day_sin: -0.7818314825,
    day_cos: 0.6234898019,
    lag_1h: 60,
    lag_24h: 50,
    lag_7d: 50,
    rolling_mean_1h: 56.6666666667
};

const groups = {
    "environment-fields": [
        ["lights", "Lights (Wh)"],
        ["T1", "Indoor temperature T1 (°C)"],
        ["RH_1", "Indoor humidity RH_1 (%)"],
        ["T_out", "Outdoor temperature (°C)"],
        ["RH_out", "Outdoor humidity (%)"],
        ["Press_mm_hg", "Atmospheric pressure (mmHg)"],
        ["Windspeed", "Wind speed"],
        ["Visibility", "Visibility"],
        ["Tdewpoint", "Dew point (°C)"]
    ],
    "time-fields": [
        ["hour", "Hour (0–23)"],
        ["day_of_week", "Day of week (0–6)"],
        ["month", "Month (1–12)"],
        ["is_weekend", "Weekend? (0 or 1)"]
    ],
    "lag-fields": [
        ["lag_1h", "Previous hour consumption"],
        ["lag_24h", "Previous 24h consumption"],
        ["lag_7d", "Previous 7d consumption"],
        ["rolling_mean_1h", "Rolling mean consumption"]
    ]
};

const advancedFeatures = [
    ["T2", "Indoor temperature T2 (°C)"],
    ["RH_2", "Indoor humidity RH_2 (%)"],
    ["T3", "Indoor temperature T3 (°C)"],
    ["RH_3", "Indoor humidity RH_3 (%)"],
    ["T4", "Indoor temperature T4 (°C)"],
    ["RH_4", "Indoor humidity RH_4 (%)"],
    ["T5", "Indoor temperature T5 (°C)"],
    ["RH_5", "Indoor humidity RH_5 (%)"],
    ["T6", "Indoor temperature T6 (°C)"],
    ["RH_6", "Indoor humidity RH_6 (%)"],
    ["T7", "Indoor temperature T7 (°C)"],
    ["RH_7", "Indoor humidity RH_7 (%)"],
    ["T8", "Indoor temperature T8 (°C)"],
    ["RH_8", "Indoor humidity RH_8 (%)"],
    ["T9", "Indoor temperature T9 (°C)"],
    ["RH_9", "Indoor humidity RH_9 (%)"],
    ["rv1", "Random variable 1"],
    ["rv2", "Random variable 2"],
    ["hour_sin", "Hour sine"],
    ["hour_cos", "Hour cosine"],
    ["day_sin", "Day sine"],
    ["day_cos", "Day cosine"]
];

function createInput(name, label) {
    const wrapper = document.createElement("label");
    wrapper.textContent = label;

    const input = document.createElement("input");
    input.name = name;
    input.type = "number";
    input.step = "any";
    input.required = true;
    input.value = sample[name];

    if (["lights", "lag_1h", "lag_24h", "lag_7d", "rolling_mean_1h"].includes(name)) {
        input.min = "0";
    }

    if (name === "hour") {
        input.min = "0";
        input.max = "23";
        input.step = "1";
    } else if (name === "day_of_week") {
        input.min = "0";
        input.max = "6";
        input.step = "1";
    } else if (name === "month") {
        input.min = "1";
        input.max = "12";
        input.step = "1";
    } else if (name === "is_weekend") {
        input.min = "0";
        input.max = "1";
        input.step = "1";
    }

    wrapper.appendChild(input);
    return wrapper;
}

function renderInputs() {
    for (const [containerId, fields] of Object.entries(groups)) {
        const container = document.getElementById(containerId);

        fields.forEach(([name, label]) => {
            container.appendChild(createInput(name, label));
        });
    }

    const advancedContainer = document.getElementById("advanced-fields");

    advancedFeatures.forEach(([name, label]) => {
        advancedContainer.appendChild(createInput(name, label));
    });
}

function setMessage(elementId, message, isError = false) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.style.color = isError ? "#f87171" : "";
}

async function checkApiStatus() {
    const status = document.getElementById("api-status");

    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API health check failed");
        }

        const data = await response.json();

        status.textContent = data.model_loaded ? "Connected" : "Model unavailable";
        status.style.color = data.model_loaded ? "#34d399" : "#f87171";
    } catch (error) {
        status.textContent = "Offline";
        status.style.color = "#f87171";
    }
}

async function predictConsumption(event) {
    event.preventDefault();

    const form = document.getElementById("prediction-form");
    const button = document.getElementById("predict-button");
    const resultValue = document.getElementById("result-value");
    const resultMessage = document.getElementById("result-message");

    if (!form.reportValidity()) {
        return;
    }

    const formData = new FormData(form);
    const payload = {};

    for (const [name, value] of formData.entries()) {
        payload[name] = Number(value);
    }

    button.disabled = true;
    button.textContent = "Predicting...";
    setMessage("form-message", "Sending readings to the ML model...");

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = typeof data.detail === "string"
                ? data.detail
                : "Please check your input values.";
            throw new Error(detail);
        }

        resultValue.replaceChildren();

        const value = document.createTextNode(
            Number(data.predicted_appliances_consumption).toFixed(2) + " "
        );
        const unit = document.createElement("small");
        unit.textContent = data.unit || "Wh";

        resultValue.append(value, unit);
        resultMessage.textContent = "Prediction completed successfully.";
        setMessage("form-message", "Prediction generated successfully.");
        checkApiStatus();
    } catch (error) {
        resultMessage.textContent = "Could not generate prediction.";
        setMessage(
            "form-message",
            `${error.message} Check that the API is running and CORS is configured.`,
            true
        );
    } finally {
        button.disabled = false;
        button.textContent = "⚡ Predict energy consumption";
    }
}

renderInputs();
document
    .getElementById("prediction-form")
    .addEventListener("submit", predictConsumption);

checkApiStatus(); 