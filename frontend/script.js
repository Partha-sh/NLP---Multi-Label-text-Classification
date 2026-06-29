const predictButton = document.getElementById("predict-button");
const commentInput = document.getElementById("comment-input");
const status = document.getElementById("status");
const resultsPanel = document.getElementById("results-panel");
const resultsGrid = document.getElementById("results-grid");
const resultsComment = document.getElementById("results-comment");

async function predictComment() {
  const text = commentInput.value.trim();

  if (!text) {
    status.textContent = "Please type a comment before analyzing.";
    return;
  }

  status.textContent = "Analyzing comment...";
  resultsPanel.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
    status.textContent = "Analysis complete.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    console.error(error);
  }
}

function renderResults(data) {
  resultsGrid.innerHTML = "";
  resultsComment.textContent = `Comment: “${data.comment}”`;

  Object.entries(data.results).forEach(([label, result]) => {
    const card = document.createElement("div");
    card.className = "tag-card";

    const title = document.createElement("div");
    title.className = "tag-title";
    title.textContent = label.replace(/_/g, " ");

    const value = document.createElement("div");
    value.className = "tag-value";

    const statusChip = document.createElement("span");
    statusChip.className = `status-chip ${result.prediction}`;
    statusChip.textContent = result.prediction ? "Toxic" : "Clean";

    const confidence = document.createElement("span");
    confidence.className = "confidence";
    confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;

    value.append(statusChip, confidence);
    card.append(title, value);
    resultsGrid.append(card);
  });

  resultsPanel.classList.remove("hidden");
}

predictButton.addEventListener("click", predictComment);
commentInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    predictComment();
  }
});
