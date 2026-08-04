const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const sessionInput = document.querySelector("#session-id");
const routeOverride = document.querySelector("#route-override");
const primaryButton = form.querySelector(".primary");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");
const traceEmpty = document.querySelector("#trace-empty");
const traceEventCount = document.querySelector("#trace-event-count");
const timeline = document.querySelector("#timeline");
const eventTemplate = document.querySelector("#event-template");
const routeCards = document.querySelector("#route-cards");
const routeTemplate = document.querySelector("#route-template");
const setupForm = document.querySelector("#setup-form");
const githubRepoInput = document.querySelector("#github-repo");
const addRepoButton = document.querySelector("#add-repo");
const addDemoRepoButton = document.querySelector("#add-demo-repo");
const repoList = document.querySelector("#repo-list");
const repoTemplate = document.querySelector("#repo-template");
const repoMessage = document.querySelector("#repo-message");
const trainingRouteList = document.querySelector("#training-route-list");
const trainingRouteTemplate = document.querySelector("#training-route-template");
const tasksPerRepoInput = document.querySelector("#tasks-per-repo");
const repetitionsInput = document.querySelector("#repetitions");
const averageRunCostInput = document.querySelector("#average-run-cost");
const privacyModeInput = document.querySelector("#privacy-mode");
const routerRungInputs = document.querySelectorAll('input[name="router-rung"]');
const routerModelField = document.querySelector("#router-model-field");
const routerModelInput = document.querySelector("#router-model");
const generateEnvironmentButton = document.querySelector("#generate-environment");
const launchGuidedDemoButton = document.querySelector("#launch-guided-demo");
const workflowDemo = document.querySelector("#workflow-demo");
const demoStageList = document.querySelector("#demo-stage-list");
const demoEmpty = document.querySelector("#demo-empty");
const demoEvent = document.querySelector("#demo-event");
const demoNextButton = document.querySelector("#run-next-demo-stage");
const demoResetButton = document.querySelector("#reset-workflow-demo");
const demoArtifactButton = document.querySelector("#open-demo-artifact");
const demoArtifactDialog = document.querySelector("#demo-artifact-dialog");
const closeDemoArtifactButton = document.querySelector("#close-demo-artifact");

let configuredRepositories = [];
let trainingRoutes = [];
let runnerSelectionTouched = false;
let currentTrainingWorkspaceId = null;
let workflowDemoState = null;
let selectedDemoEventIndex = null;

sessionInput.value = `demo-${crypto.randomUUID().slice(0, 8)}`;
loadTrainingCatalog();
loadRouterStatus();
updateRunnerGuidance();
updateRouterRungGuidance();

launchGuidedDemoButton.addEventListener("click", async () => {
  launchGuidedDemoButton.disabled = true;
  launchGuidedDemoButton.textContent = "Creating Click workspace…";
  try {
    configuredRepositories = [
      {
        full_name: "pallets/click",
        html_url: "https://github.com/pallets/click",
        default_branch: "main",
        language: "Python",
        visibility: "public",
        verification: "verified_public",
      },
    ];
    renderRepositories();
    updateRunnerGuidance();
    if (!trainingRoutes.length) await loadTrainingCatalog();
    if (!trainingRoutes.length) {
      throw new Error("The route catalog is not available yet.");
    }
    updateEstimate();
    setupForm.requestSubmit();
  } catch (error) {
    showRepoMessage(error.message, true);
    launchGuidedDemoButton.disabled = false;
    launchGuidedDemoButton.textContent = "Launch guided Click demo →";
  }
});

addRepoButton.addEventListener("click", () => addRepository());
addDemoRepoButton.addEventListener("click", () => {
  addRepository("https://github.com/pallets/click");
});
githubRepoInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addRepository();
  }
});

[tasksPerRepoInput, repetitionsInput, averageRunCostInput].forEach((input) => {
  input.addEventListener("input", updateEstimate);
});
privacyModeInput.addEventListener("change", () => {
  runnerSelectionTouched = true;
  updateRunnerGuidance();
});
routerRungInputs.forEach((input) => {
  input.addEventListener("change", updateRouterRungGuidance);
});
demoNextButton.addEventListener("click", runNextDemoStage);
demoResetButton.addEventListener("click", resetWorkflowDemo);
demoArtifactButton.addEventListener("click", openDemoArtifact);
closeDemoArtifactButton.addEventListener("click", () => demoArtifactDialog.close());
demoArtifactDialog.addEventListener("click", (event) => {
  if (event.target === demoArtifactDialog) demoArtifactDialog.close();
});

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selectedRouteIds = selectedTrainingRouteIds();
  if (!configuredRepositories.length) {
    showRepoMessage("Add at least one GitHub repository.", true);
    githubRepoInput.focus();
    return;
  }
  if (selectedRouteIds.length < 2) {
    showRepoMessage("Select at least two routes so the router has a choice.", true);
    return;
  }

  setSetupLoading(true);
  try {
    const response = await fetch("/api/training/environments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repositories: configuredRepositories,
        selected_route_ids: selectedRouteIds,
        tasks_per_repo: Number(tasksPerRepoInput.value),
        repetitions: Number(repetitionsInput.value),
        average_run_cost_usd: Number(averageRunCostInput.value),
        privacy_mode: privacyModeInput.value,
        router_rung: selectedRouterRung(),
        router_model: routerModelInput.value.trim(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not generate the training workspace.");
    }
    renderEnvironment(payload);
    showRepoMessage("Training workspace generated without copying repository source.", false);
  } catch (error) {
    showRepoMessage(error.message, true);
  } finally {
    setSetupLoading(false);
    launchGuidedDemoButton.disabled = false;
    launchGuidedDemoButton.textContent = "Launch guided Click demo →";
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.dataset.prompt;
    question.focus();
  });
});

async function loadTrainingCatalog() {
  try {
    const response = await fetch("/api/training/catalog");
    const payload = await response.json();
    if (!response.ok) throw new Error("Could not load the route catalog.");
    trainingRoutes = payload.routes || [];
    renderTrainingRoutes();
    updateEstimate();
  } catch (error) {
    trainingRouteList.innerHTML = `<div class="setup-empty">${error.message}</div>`;
  }
}

async function loadRouterStatus() {
  const runtime = document.querySelector("#router-runtime");
  const label = document.querySelector("#router-runtime-label");
  const detail = document.querySelector("#router-runtime-detail");
  const prompt = document.querySelector("#router-system-prompt");
  try {
    const response = await fetch("/api/router/status");
    const payload = await response.json();
    if (!response.ok) throw new Error("Router status is unavailable.");
    label.textContent = `${payload.label} via ${payload.transport}`;
    detail.textContent =
      payload.status === "stock_untrained"
        ? "Stock checkpoint · output shape is enforced · scores are not calibrated yet"
        : `${payload.status.replaceAll("_", " ")} · ${payload.policy}`;
    prompt.textContent = payload.system_prompt;
    runtime.classList.toggle("offline", !payload.enabled);
  } catch (error) {
    label.textContent = "Router unavailable";
    detail.textContent = error.message;
    prompt.textContent = "The router prompt could not be loaded.";
    runtime.classList.add("offline");
  }
}

async function addRepository(repositoryOverride = "") {
  const repository = repositoryOverride || githubRepoInput.value.trim();
  if (!repository) {
    showRepoMessage("Enter a GitHub repository URL or owner/repo.", true);
    return;
  }
  addRepoButton.disabled = true;
  addDemoRepoButton.disabled = true;
  addRepoButton.textContent = "Checking…";
  if (repositoryOverride) addDemoRepoButton.textContent = "Adding Click…";
  showRepoMessage("Checking the public GitHub metadata…", false);
  try {
    const response = await fetch("/api/github/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repository }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not add repository.");
    const inspected = payload.repository;
    if (configuredRepositories.some((item) => item.full_name === inspected.full_name)) {
      throw new Error(`${inspected.full_name} is already configured.`);
    }
    configuredRepositories.push(inspected);
    githubRepoInput.value = "";
    renderRepositories();
    updateRunnerGuidance();
    updateEstimate();
    showRepoMessage(
      payload.warning || `${inspected.full_name} was verified and added.`,
      Boolean(payload.warning),
    );
  } catch (error) {
    showRepoMessage(error.message, true);
  } finally {
    addRepoButton.disabled = false;
    addDemoRepoButton.disabled = false;
    addRepoButton.textContent = "Add";
    addDemoRepoButton.textContent = "Try with pallets/click →";
  }
}

function renderRepositories() {
  repoList.replaceChildren();
  if (!configuredRepositories.length) {
    const empty = document.createElement("div");
    empty.className = "setup-empty";
    empty.textContent = "No repository connected";
    repoList.append(empty);
    return;
  }
  configuredRepositories.forEach((repository) => {
    const fragment = repoTemplate.content.cloneNode(true);
    fragment.querySelector(".repo-name").textContent = repository.full_name;
    fragment.querySelector(".repo-details").textContent =
      `${repository.default_branch} · ${repository.language || repository.visibility}`;
    const verification = fragment.querySelector(".repo-verification");
    verification.textContent =
      repository.verification === "verified_public" ? "Verified" : "App required";
    verification.classList.toggle(
      "warning",
      repository.verification !== "verified_public",
    );
    fragment.querySelector(".remove-repo").addEventListener("click", () => {
      configuredRepositories = configuredRepositories.filter(
        (item) => item.full_name !== repository.full_name,
      );
      renderRepositories();
      updateRunnerGuidance();
      updateEstimate();
    });
    repoList.append(fragment);
  });
}

function updateRunnerGuidance() {
  const hasPrivateRepository = configuredRepositories.some(
    (repository) =>
      repository.visibility === "private" ||
      repository.verification !== "verified_public",
  );
  if (!runnerSelectionTouched) {
    privacyModeInput.value = hasPrivateRepository
      ? "customer_runner"
      : "castform_hosted";
  }

  const guidance = document.querySelector("#runner-guidance");
  if (privacyModeInput.value === "customer_runner") {
    guidance.textContent = hasPrivateRepository
      ? "Recommended for this private repository. Code and credentials stay inside your cloud or VPC."
      : "Runs inside your cloud or VPC. You provision and maintain the worker.";
    guidance.classList.toggle("recommended-private", hasPrivateRepository);
    return;
  }

  guidance.textContent = hasPrivateRepository
    ? "Private code will be cloned into an isolated, temporary Castform sandbox. Confirm your organization allows hosted execution."
    : "Zero setup. Runs in an isolated, temporary Castform sandbox.";
  guidance.classList.remove("recommended-private");
}

function renderTrainingRoutes() {
  trainingRouteList.replaceChildren();
  const defaults = new Set([
    "claude-code/opus@anthropic",
    "claude-code/sonnet@anthropic",
    "codex/5.6-balanced@openai",
  ]);
  trainingRoutes.forEach((route) => {
    const fragment = trainingRouteTemplate.content.cloneNode(true);
    const checkbox = fragment.querySelector(".training-route-checkbox");
    checkbox.value = route.route_id;
    checkbox.checked = defaults.has(route.route_id);
    checkbox.addEventListener("change", updateEstimate);
    fragment.querySelector(".training-route-label").textContent = route.label;
    fragment.querySelector(".training-route-meta").textContent =
      `${route.harness} · ${route.provider}`;
    trainingRouteList.append(fragment);
  });
}

function selectedTrainingRouteIds() {
  return Array.from(
    trainingRouteList.querySelectorAll(".training-route-checkbox:checked"),
  ).map((checkbox) => checkbox.value);
}

function selectedRouterRung() {
  return document.querySelector('input[name="router-rung"]:checked')?.value || "knn";
}

function updateRouterRungGuidance() {
  const usesModel = selectedRouterRung() !== "knn";
  routerModelField.classList.toggle("hidden", !usesModel);
  routerModelInput.disabled = !usesModel;
}

function updateEstimate() {
  const tasks =
    configuredRepositories.length * Math.max(0, Number(tasksPerRepoInput.value) || 0);
  const rollouts =
    tasks *
    selectedTrainingRouteIds().length *
    Math.max(0, Number(repetitionsInput.value) || 0);
  const cost = rollouts * Math.max(0, Number(averageRunCostInput.value) || 0);
  document.querySelector("#estimated-tasks").textContent = tasks.toLocaleString();
  document.querySelector("#planned-rollouts").textContent = rollouts.toLocaleString();
  document.querySelector("#estimated-cost").textContent = cost.toLocaleString(
    undefined,
    { style: "currency", currency: "USD" },
  );
}

function setSetupLoading(loading) {
  generateEnvironmentButton.disabled = loading;
  generateEnvironmentButton.textContent = loading
    ? "Creating environment…"
    : "Create training environment";
}

function showRepoMessage(message, isError) {
  repoMessage.textContent = message;
  repoMessage.classList.toggle("error", isError);
  repoMessage.classList.remove("hidden");
}

function renderEnvironment(payload) {
  document.querySelector("#environment-id").textContent = payload.workspace_id;
  document.querySelector("#environment-path").textContent = payload.workspace_path;
  document.querySelector("#environment-command").textContent = payload.next_command;
  const files = document.querySelector("#environment-files");
  files.replaceChildren();
  (payload.files || []).forEach((filename) => {
    const item = document.createElement("li");
    item.textContent = filename;
    files.append(item);
  });
  document.querySelector("#environment-result").classList.remove("hidden");
  currentTrainingWorkspaceId = payload.workspace_id;
  loadWorkflowDemo(payload.workspace_id);
}

async function loadWorkflowDemo(workspaceId) {
  try {
    const response = await fetch(
      `/api/training/environments/${workspaceId}/demo`,
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not load the workflow demo.");
    }
    workflowDemoState = payload;
    selectedDemoEventIndex = payload.events.length - 1;
    workflowDemo.classList.remove("hidden");
    renderWorkflowDemo();
    workflowDemo.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showRepoMessage(error.message, true);
  }
}

async function runNextDemoStage() {
  if (!currentTrainingWorkspaceId || workflowDemoState?.complete) return;
  demoNextButton.disabled = true;
  demoNextButton.textContent = "Running simulated step…";
  try {
    const response = await fetch(
      `/api/training/environments/${currentTrainingWorkspaceId}/demo/next`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The demo step failed.");
    workflowDemoState = payload;
    selectedDemoEventIndex = payload.events.length - 1;
    renderWorkflowDemo();
  } catch (error) {
    showRepoMessage(error.message, true);
  } finally {
    demoNextButton.disabled = Boolean(workflowDemoState?.complete);
  }
}

async function resetWorkflowDemo() {
  if (!currentTrainingWorkspaceId) return;
  demoResetButton.disabled = true;
  try {
    const response = await fetch(
      `/api/training/environments/${currentTrainingWorkspaceId}/demo/reset`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not reset the demo.");
    workflowDemoState = payload;
    selectedDemoEventIndex = -1;
    renderWorkflowDemo();
  } catch (error) {
    showRepoMessage(error.message, true);
  } finally {
    demoResetButton.disabled = false;
  }
}

function renderWorkflowDemo() {
  if (!workflowDemoState) return;
  document.querySelector("#demo-workspace-id").textContent =
    workflowDemoState.workspace_id;
  document.querySelector("#demo-notice-copy").textContent =
    workflowDemoState.notice;
  renderDemoStages();
  renderDemoEvent();
  renderDemoControls();
}

function renderDemoStages() {
  demoStageList.replaceChildren();
  workflowDemoState.stages.forEach((stage, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `demo-stage ${stage.status}`;
    if (index === selectedDemoEventIndex) button.classList.add("selected");
    button.disabled = stage.status !== "completed";
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const copy = document.createElement("span");
    const label = document.createElement("strong");
    const status = document.createElement("small");
    label.textContent = stage.label;
    status.textContent =
      stage.status === "completed"
        ? "Complete"
        : stage.status === "next"
        ? stage.spends_model_credits
          ? "Next · simulated spend"
          : "Next"
        : "Waiting";
    copy.append(label, status);
    button.append(number, copy);
    if (stage.status === "completed") {
      button.addEventListener("click", () => {
        selectedDemoEventIndex = index;
        renderWorkflowDemo();
      });
    }
    demoStageList.append(button);
  });
}

function renderDemoEvent() {
  const event = workflowDemoState.events[selectedDemoEventIndex];
  demoEmpty.classList.toggle("hidden", Boolean(event));
  demoEvent.classList.toggle("hidden", !event);
  if (!event) return;

  document.querySelector("#demo-event-kicker").textContent =
    `STEP ${String(selectedDemoEventIndex + 1).padStart(2, "0")} · ${event.id}`;
  document.querySelector("#demo-event-title").textContent = event.label;
  document.querySelector("#demo-event-summary").textContent = event.summary;
  document.querySelector("#demo-artifact-path").textContent = event.artifact;

  const metrics = document.querySelector("#demo-metrics");
  metrics.replaceChildren();
  (event.metrics || []).forEach((metric) => {
    const card = document.createElement("div");
    const label = document.createElement("small");
    const value = document.createElement("strong");
    label.textContent = metric.label;
    value.textContent = metric.value;
    card.append(label, value);
    metrics.append(card);
  });

  const commands = document.querySelector("#demo-commands");
  commands.replaceChildren();
  (event.commands || []).forEach((command, index) => {
    const block = document.createElement("pre");
    block.textContent = `$ ${formatDemoCommand(command)}`;
    if (index > 1) block.classList.add("secondary-command");
    commands.append(block);
  });
  renderDemoOutput(event.output || {});
}

function renderDemoOutput(output) {
  const container = document.querySelector("#demo-output");
  container.replaceChildren();
  if (Array.isArray(output.candidate_tasks)) {
    const preview = document.createElement("div");
    preview.className = "candidate-preview";

    const notice = document.createElement("p");
    const limit = Number(output.configured_pr_limit) || 0;
    notice.textContent = output.preview_only
      ? `${output.preview_count} illustrative examples shown · live mining limit: up to ${limit}`
      : `${output.candidate_tasks.length} candidates mined`;
    preview.append(notice);

    const list = document.createElement("ol");
    output.candidate_tasks.forEach((task) => {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      const meta = document.createElement("small");
      title.textContent = task.title;
      meta.textContent = `${task.candidate_id} · ${task.status}`;
      item.append(title, meta);
      list.append(item);
    });
    preview.append(list);
    container.append(preview);
    return;
  }
  const rows = output.route_results || output.rows;
  if (Array.isArray(rows)) {
    const table = document.createElement("table");
    const columns = output.route_results
      ? ["model", "solve_rate", "cost_per_task_usd"]
      : ["policy", "solve_rate", "cost_per_task_usd"];
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("th");
      cell.textContent =
        column === "solve_rate"
          ? "Solve"
          : column === "cost_per_task_usd"
          ? "$/task"
          : column === "model"
          ? "Route model"
          : "Policy";
      headRow.append(cell);
    });
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const rowElement = document.createElement("tr");
      columns.forEach((column) => {
        const cell = document.createElement("td");
        const value = row[column];
        cell.textContent =
          column === "solve_rate"
            ? `${Math.round(Number(value) * 100)}%`
            : column === "cost_per_task_usd"
            ? `$${Number(value).toFixed(2)}`
            : value;
        rowElement.append(cell);
      });
      body.append(rowElement);
    });
    table.append(head, body);
    container.append(table);
    const remainder = { ...output };
    delete remainder.route_results;
    delete remainder.rows;
    if (Object.keys(remainder).length) {
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(remainder, null, 2);
      container.append(pre);
    }
    return;
  }
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(output, null, 2);
  container.append(pre);
}

function openDemoArtifact() {
  const event = workflowDemoState?.events[selectedDemoEventIndex];
  if (!event) return;

  document.querySelector("#artifact-dialog-title").textContent =
    `${event.label} output`;
  document.querySelector("#artifact-dialog-path").textContent = event.artifact;
  document.querySelector("#artifact-dialog-notice").textContent =
    event.artifact_notice ||
    "This is the illustrative stage output. A live runner writes the artifact at the path above.";
  document.querySelector("#artifact-dialog-content").textContent = JSON.stringify(
    {
      simulation: true,
      stage: event.id,
      artifact: event.artifact,
      output: event.output || {},
    },
    null,
    2,
  );
  demoArtifactDialog.showModal();
}

function renderDemoControls() {
  const nextStage = workflowDemoState.stages[workflowDemoState.current_stage + 1];
  const label = document.querySelector("#demo-next-label");
  const note = document.querySelector("#demo-next-note");
  if (!nextStage) {
    label.textContent = "Experiment complete";
    note.textContent = "Review the scoreboard before deciding whether to train.";
    demoNextButton.textContent = "Demo complete";
    demoNextButton.disabled = true;
    return;
  }
  label.textContent = nextStage.label;
  note.textContent = nextStage.spends_model_credits
    ? "Simulation is $0 · a live runner would spend model credits"
    : "Safe simulated stage · no model credits";
  demoNextButton.textContent = nextStage.spends_model_credits
    ? "Simulate paid step ($0) →"
    : "Run next step →";
  demoNextButton.disabled = false;
}

function formatDemoCommand(command) {
  return command
    .map((part) => (/[\s]/.test(part) ? JSON.stringify(part) : part))
    .join(" ");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  timeline.replaceChildren();
  traceEmpty.textContent = "Tracing request…";
  traceEmpty.classList.remove("hidden");
  traceEventCount.textContent = "Recording low-level events…";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question.value,
        session_id: sessionInput.value,
        route_override: routeOverride.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "The traced request failed.");
    }
    renderResult(payload);
    renderRoutes(payload);
    renderTimeline(payload.events || []);
  } catch (error) {
    emptyResult.classList.remove("hidden");
    resultContent.classList.add("hidden");
    emptyResult.querySelector("p").textContent = error.message;
    traceEmpty.textContent = "The trace stopped before completion.";
    traceEventCount.textContent = "Low-level trace unavailable";
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  primaryButton.disabled = loading;
  primaryButton.querySelector("span").textContent = loading
    ? "Qwen is scoring routes…"
    : "Route task through demo";
}

function renderResult(payload) {
  const response = payload.response || {};
  const choice = response.choices?.[0]?.message?.content || "No completion returned.";
  const usage = response.usage || {};
  const route = payload.selected_route || {};

  document.querySelector("#selected-route").textContent =
    route.route_id || "unknown";
  document.querySelector("#selected-harness").textContent =
    route.harness || "unknown";
  document.querySelector("#selected-model").textContent =
    route.model || response.model || "unknown";
  document.querySelector("#selected-provider").textContent =
    route.provider || "unknown";
  document.querySelector("#router-duration").textContent =
    payload.router_duration_ms == null ? "—" : `${payload.router_duration_ms} ms`;
  document.querySelector("#duration").textContent = `${payload.duration_ms} ms`;
  document.querySelector("#tokens").textContent =
    usage.total_tokens == null ? "—" : usage.total_tokens;
  document.querySelector("#router-model-version").textContent =
    payload.router_model_label || payload.router_output?.router_model_version || "unknown";
  document.querySelector("#decision-reason").textContent =
    payload.decision?.reason?.replaceAll("_", " ") || "unknown";
  document.querySelector("#answer").textContent = choice;
  document.querySelector("#trace-id").textContent = payload.trace_id;
  emptyResult.classList.add("hidden");
  resultContent.classList.remove("hidden");
}

function renderRoutes(payload) {
  const predictions = payload.router_output?.predictions || [];
  const selected = payload.selected_route || {};
  const routeRegistryEvent = (payload.events || []).find(
    (event) => event.stage === "route.candidates_built",
  );
  const candidates = routeRegistryEvent?.output?.candidate_routes || [];
  const predictionsByRoute = new Map(
    predictions.map((prediction) => [prediction.route_id, prediction]),
  );

  routeCards.replaceChildren();
  candidates.forEach((route) => {
    const prediction = predictionsByRoute.get(route.route_id) || {};
    const fragment = routeTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".route-card");
    const isSelected = route.route_id === selected.route_id;
    card.classList.toggle("selected", isSelected);
    fragment.querySelector(".route-state").textContent = isSelected
      ? payload.decision?.cache_hit
        ? "Pinned route"
        : "Selected route"
      : "Candidate";
    fragment.querySelector(".route-cost").textContent =
      `$${route.estimated_cost_usd.toFixed(2)} est.`;
    fragment.querySelector(".route-id").textContent = route.route_id;
    fragment.querySelector(".route-harness").textContent = route.harness;
    fragment.querySelector(".route-model").textContent = route.model;
    fragment.querySelector(".route-provider").textContent = route.provider;
    fragment.querySelector(".route-success").textContent =
      prediction.success_probability == null
        ? "—"
        : `${Math.round(prediction.success_probability * 100)}%`;
    const expectedTokens = [
      prediction.expected_input_tokens,
      prediction.expected_cache_read_tokens,
      prediction.expected_output_tokens,
    ].every((value) => Number.isFinite(value))
      ? prediction.expected_input_tokens +
        prediction.expected_cache_read_tokens +
        prediction.expected_output_tokens
      : prediction.expected_total_tokens;
    fragment.querySelector(".route-tokens").textContent =
      expectedTokens?.toLocaleString() || "—";
    fragment.querySelector(".route-uncertainty").textContent =
      prediction.uncertainty == null
        ? "—"
        : prediction.uncertainty.toFixed(2);
    routeCards.append(fragment);
  });
}

function renderTimeline(events) {
  timeline.replaceChildren();
  if (!events.length) {
    traceEmpty.textContent = "No events were recorded.";
    traceEmpty.classList.remove("hidden");
    traceEventCount.textContent = "No low-level events recorded";
    return;
  }
  traceEventCount.textContent = `${events.length} low-level events recorded`;
  traceEmpty.classList.add("hidden");
  const started = Number(events[0].timestamp_ns || 0);

  events.forEach((event, index) => {
    const fragment = eventTemplate.content.cloneNode(true);
    fragment.querySelector(".step-number").textContent = String(index + 1).padStart(2, "0");
    fragment.querySelector(".actor").textContent = event.actor;
    fragment.querySelector(".summary-text").textContent = event.summary;
    fragment.querySelector(".stage").textContent = event.stage;
    const elapsedMs = (Number(event.timestamp_ns || started) - started) / 1_000_000;
    fragment.querySelector(".elapsed").textContent = `+${elapsedMs.toFixed(1)} ms`;

    const payload = fragment.querySelector(".payload");
    [
      ["Input", event.input],
      ["Output", event.output],
      ["Details", event.details],
    ].forEach(([label, value]) => {
      if (value === undefined) return;
      const block = document.createElement("section");
      const heading = document.createElement("small");
      const pre = document.createElement("pre");
      heading.textContent = label;
      pre.textContent = JSON.stringify(value, null, 2);
      block.append(heading, pre);
      payload.append(block);
    });
    timeline.append(fragment);
  });
}
