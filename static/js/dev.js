// Developer panel (Phase 1). DOM-only rendering: textContent and
// createElement only, never innerHTML. No new libraries.
(function () {
  "use strict";

  function el(tag, text) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function showError(message) {
    var box = document.getElementById("devError");
    var text = document.getElementById("devErrorText");
    var content = document.getElementById("devContent");
    text.textContent = message;
    box.classList.remove("hidden");
    content.classList.add("hidden");
  }

  function fillRows(tbodyId, emptyId, rows, columns) {
    var tbody = document.getElementById(tbodyId);
    var empty = document.getElementById(emptyId);
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
    if (!rows || rows.length === 0) {
      empty.classList.remove("hidden");
      return;
    }
    empty.classList.add("hidden");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      columns.forEach(function (col) { tr.appendChild(el("td", col(row))); });
      tbody.appendChild(tr);
    });
  }

  function formatTime(epochSeconds) {
    if (!epochSeconds) return "—";
    var d = new Date(epochSeconds * 1000);
    return isNaN(d.getTime()) ? "—" : d.toLocaleString();
  }

  function loadJson(path) {
    return fetch(path).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) {
          var err = new Error((data && data.error) || ("Request failed: " + res.status));
          err.status = res.status;
          throw err;
        }
        return data;
      });
    });
  }

  function loadAll() {
    loadJson("/api/dev/overview").then(function (data) {
      document.getElementById("storeKind").textContent = data.store || "—";
      document.getElementById("ragBackend").textContent = data.rag_backend || "—";
      document.getElementById("llmInfo").textContent =
        (data.llm_mode || "—") + (data.llm_model ? " (" + data.llm_model + ")" : "");
      document.getElementById("totalUsers").textContent = data.total_users;
      document.getElementById("loginsToday").textContent = data.logins_today;
      document.getElementById("logins7d").textContent = data.logins_7d;
      document.getElementById("questionsToday").textContent = data.questions_today;
      document.getElementById("questions7d").textContent = data.questions_7d;
      var rate = data.fallback_rate_7d;
      document.getElementById("fallbackRate").textContent =
        (typeof rate === "number" ? (rate * 100).toFixed(1) + "%" : "—");
      document.getElementById("documentsTotal").textContent = data.documents_total;
      var settings = data.settings || {};
      document.getElementById("webSearchToggle").checked = settings.web_search_enabled !== false;
      document.getElementById("ensembleToggle").checked = settings.ensemble_enabled === true;
      document.getElementById("announcementInput").value = settings.announcement || "";
    }).catch(function (err) { showError(err.message); });

    loadJson("/api/dev/failed-questions").then(function (data) {
      fillRows("failedBody", "failedEmpty", data.failed_questions, [
        function (r) { return r.question; },
        function (r) { return r.count; }
      ]);
    }).catch(function () {});

    loadJson("/api/dev/top-questions").then(function (data) {
      fillRows("topBody", "topEmpty", data.top_questions, [
        function (r) { return r.question; },
        function (r) { return r.count; }
      ]);
    }).catch(function () {});

    loadJson("/api/dev/users").then(function (data) {
      fillRows("usersBody", "usersEmpty", data.users, [
        function (u) { return (u.user_id || "").slice(0, 12) + "…"; },
        function (u) { return u.display_name || "(no name yet)"; },
        function (u) { return u.avatar_id || "—"; },
        function (u) { return u.message_count; },
        function (u) { return formatTime(u.last_active); }
      ]);
      // User IDs are shortened in the table (first 12 characters).
    }).catch(function () {});
  }

  var template = document.getElementById("pihuTemplate");
  template.value =
    "Pihu, from the developer panel:\n" +
    "Please improve Shibu AI based on the failed questions shown in the panel. " +
    "For each failed question, add a rule or a fix, test it, and report what changed. " +
    "Do not push to GitHub until I say merge for that batch.";

  document.getElementById("copyBtn").addEventListener("click", function () {
    var status = document.getElementById("copyStatus");
    function done() { status.textContent = "Copied — paste it in chat with Pihu."; }
    function failed() {
      template.select();
      status.textContent = "Copy did not work automatically — the text is selected, press Ctrl+C.";
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(template.value).then(done, failed);
    } else {
      failed();
    }
  });

  document.getElementById("settingsForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var status = document.getElementById("settingsStatus");
    status.textContent = "Saving…";
    var announcement = document.getElementById("announcementInput").value;
    var saves = [
      { key: "web_search_enabled", value: document.getElementById("webSearchToggle").checked },
      { key: "ensemble_enabled", value: document.getElementById("ensembleToggle").checked },
      { key: "announcement", value: announcement }
    ];
    var chain = Promise.resolve();
    saves.forEach(function (item) {
      chain = chain.then(function () {
        return fetch("/api/dev/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item)
        }).then(function (res) {
          return res.json().then(function (data) {
            if (!res.ok) throw new Error((data && data.error) || ("Save failed: " + res.status));
          });
        });
      });
    });
    chain.then(function () { status.textContent = "Saved."; })
      .catch(function (err) { status.textContent = err.message; });
  });

  loadAll();
})();
