/* DLA Suite — application shell.
 *
 * Two views in one page: the upload form and a job. Which one is shown is
 * derived from the URL (`/` or `/jobs/<id>`), so a running analysis has a real,
 * shareable address and a reload does not lose it. There is no client-side
 * state worth persisting: the server's status.json is the single source of
 * truth and this file only polls it. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (t, a, h) => { const n = document.createElement(t); if (a) for (const k in a) n.setAttribute(k, a[k]); if (h != null) n.innerHTML = h; return n; };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* ---------- theme ---------- */
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const now = cur || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", now === "dark" ? "light" : "dark");
  });

  /* ---------- state ---------- */
  let CFG = null, file = null, profile = null, poll = null;

  const jobFromUrl = () => {
    const m = location.pathname.match(/^\/jobs\/([0-9]{8}-[0-9]{6}-[0-9a-f]{6})$/);
    return m ? m[1] : null;
  };

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) { /* not JSON */ }
      throw new Error(msg);
    }
    return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
  }

  /* ---------- upload view ---------- */
  function renderProfiles() {
    const host = $("#profiles");
    host.innerHTML = "";
    CFG.profiles.forEach(p => {
      // `ready` is how many of the profile's models have an environment on
      // disk. A profile with none cannot produce a report, so it is shown and
      // disabled rather than hidden — the reason is more useful than the gap.
      const ready = p.ready == null ? p.systems : p.ready;
      const short = ready < p.systems;
      const b = el("button", {
        class: "opt", type: "button",
        "aria-pressed": String(p.name === profile),
        ...(ready ? {} : { disabled: "disabled", title: "no environment built for this profile" })
      }, `<b>${esc(p.name)} · ${ready}${short ? ` of ${p.systems}` : ""} models${ready ? "" : " — not installed"}</b>
          <span>${esc(p.description || "")}</span>`);
      if (ready) b.addEventListener("click", () => {
        profile = p.name;
        [...host.children].forEach(c => c.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
      });
      host.appendChild(b);
    });
    // Never leave a disabled profile selected.
    if (!(CFG.profiles.find(p => p.name === profile)?.ready)) {
      const first = CFG.profiles.find(p => p.ready);
      if (first) { profile = first.name; }
      [...host.children].forEach((c, i) =>
        c.setAttribute("aria-pressed", String(CFG.profiles[i].name === profile)));
    }
    $("#limits").textContent = CFG.systems_ready === 0
      ? "No model environments are built yet — run `make setup` before analysing anything."
      : `Up to ${CFG.max_pages} pages and ${CFG.max_upload_mb} MB · ` +
        `${CFG.systems_ready} of ${CFG.systems_total} registered systems have an environment built`;
  }

  function pickFile(f) {
    if (!f) return;
    if (!/\.pdf$/i.test(f.name)) { showUploadErr("That is not a PDF."); return; }
    if (f.size > CFG.max_upload_mb * 1024 * 1024) {
      showUploadErr(`That file is ${(f.size / 1048576).toFixed(0)} MB; the limit is ${CFG.max_upload_mb} MB.`);
      return;
    }
    file = f;
    $("#fileName").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
    $("#goBtn").disabled = false;
    hide("#uploadErr");
  }

  const hide = (s) => $(s).classList.add("hidden");
  const show = (s) => $(s).classList.remove("hidden");
  function showUploadErr(m) { $("#uploadErr").textContent = m; show("#uploadErr"); }

  function wireUpload() {
    const drop = $("#drop"), input = $("#fileInput");
    drop.addEventListener("click", () => input.click());
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => pickFile(input.files[0]));
    ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", e => pickFile(e.dataTransfer.files[0]));

    $("#goBtn").addEventListener("click", async () => {
      if (!file) return;
      $("#goBtn").disabled = true;
      $("#goBtn").textContent = "Uploading…";
      const fd = new FormData();
      fd.append("file", file);
      fd.append("profile", profile);
      try {
        const res = await api("/api/jobs", { method: "POST", body: fd });
        history.pushState({}, "", `/jobs/${res.job_id}`);
        openJob(res.job_id);
      } catch (e) {
        showUploadErr(String(e.message || e));
        $("#goBtn").disabled = false;
        $("#goBtn").textContent = "Analyse document";
      }
    });
  }

  async function renderJobs() {
    let jobs = [];
    try { jobs = await api("/api/jobs?limit=25"); } catch (e) { return; }
    const wrap = $("#jobsWrap");
    if (!jobs.length) { wrap.innerHTML = '<p class="small">No analyses yet.</p>'; return; }
    const badge = (s) => s === "done" ? "ok" : (s === "failed" ? "bad" : "run");
    wrap.innerHTML = `<div style="overflow-x:auto"><table><thead><tr>
      <th>Document</th><th>Models</th><th>Pages</th><th>Started</th><th>State</th><th></th>
      </tr></thead><tbody>${jobs.map(j => `
      <tr><td class="doc">${esc((j.documents || [])[0] || j.job_id)}</td>
        <td>${esc(j.profile || "—")}</td>
        <td>${j.pages == null ? "—" : j.pages}</td>
        <td class="tiny">${esc((j.created || "").replace("T", " ").replace("+00:00", ""))}</td>
        <td><span class="badge ${badge(j.state)}">${esc(j.state)}</span></td>
        <td><a class="btn" href="/jobs/${j.job_id}">Open</a></td></tr>`).join("")}
      </tbody></table></div>`;
    wrap.querySelectorAll("a.btn").forEach(a => a.addEventListener("click", e => {
      e.preventDefault();
      history.pushState({}, "", a.getAttribute("href"));
      openJob(a.getAttribute("href").split("/").pop());
    }));
  }

  /* ---------- job view ---------- */
  function renderStatus(st) {
    const stages = st.stages || [];
    const doneCount = stages.filter(s => s.state === "done" || s.state === "skipped").length;
    const sys = st.systems || {};

    $("#jobState").textContent = st.state;
    $("#jobState").className = "badge " + (st.state === "done" ? "ok" : st.state === "failed" ? "bad" : "run");
    $("#jobTitle").textContent = (st.meta || {}).documents ? (st.meta.documents[0] || "Analysis") : "Analysis";
    $("#jobSub").textContent = [
      (st.meta || {}).profile ? `profile: ${st.meta.profile}` : null,
      st.pages != null ? `${st.pages} pages` : null,
      sys.total ? `${sys.done}/${sys.total} models` : null,
    ].filter(Boolean).join(" · ");

    // Two thirds of a run is the model stage, so weight the bar by that rather
    // than by stage count — otherwise it sits at 27% for the entire run.
    const runIdx = stages.findIndex(s => s.name === "run");
    const before = stages.slice(0, runIdx).filter(s => s.state === "done" || s.state === "skipped").length;
    const after = stages.slice(runIdx + 1).filter(s => s.state === "done" || s.state === "skipped").length;
    const runFrac = sys.total ? sys.done / sys.total : (stages[runIdx]?.state === "done" ? 1 : 0);
    const pct = Math.round(100 * (0.15 * (before / Math.max(runIdx, 1)) + 0.7 * runFrac +
      0.15 * (after / Math.max(stages.length - runIdx - 1, 1))));
    $("#bar").style.width = Math.max(2, Math.min(100, pct)) + "%";
    $("#barNote").textContent = sys.current
      ? `running ${sys.current}`
      : (st.state === "done" ? `finished in ${elapsed(st)}` : `${doneCount}/${stages.length} stages`);

    $("#stages").innerHTML = stages.map(s => `
      <div class="stage ${esc(s.state)}">
        <span class="dot"></span>
        <span><span class="nm">${esc(s.name)}</span>
          <span class="ds">${esc(s.description || "")}</span></span>
        <span class="t">${s.seconds != null ? s.seconds + "s" : ""}</span>
      </div>`).join("");

    if (st.error) { $("#jobErr").textContent = st.error; show("#jobErr"); } else hide("#jobErr");

    if (st.state === "done" && st.report) {
      show("#reportCard");
      const url = `/api/jobs/${st.job_id}/report`;
      if ($("#reportFrame").getAttribute("src") !== url) $("#reportFrame").setAttribute("src", url);
      $("#openReport").href = url;
      $("#downloadData").href = `/api/jobs/${st.job_id}/bundle`;
    }
    return st.state === "done" || st.state === "failed";
  }

  function elapsed(st) {
    if (!st.started || !st.ended) return "—";
    const s = Math.round((new Date(st.ended) - new Date(st.started)) / 1000);
    return s < 90 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  async function tick(id) {
    let st;
    try { st = await api(`/api/jobs/${id}`); }
    catch (e) { $("#jobErr").textContent = String(e.message || e); show("#jobErr"); return true; }
    const finished = renderStatus(st);
    // While the models are running, show the tail of that stage's log: it is the
    // only place a per-page failure is visible while the job is still going.
    const running = (st.stages || []).find(s => s.state === "running");
    if (running) {
      try {
        const txt = await api(`/api/jobs/${id}/log/${running.name}`);
        const lines = String(txt).trimEnd().split("\n").slice(-14).join("\n");
        $("#jobLog").textContent = lines;
        show("#jobLog");
      } catch (e) { /* the log may not exist for a moment */ }
    } else if (finished) hide("#jobLog");
    return finished;
  }

  async function openJob(id) {
    hide("#viewUpload"); show("#viewJob");
    hide("#reportCard"); hide("#jobErr");
    if (poll) { clearInterval(poll); poll = null; }
    const step = async () => { if (await tick(id) && poll) { clearInterval(poll); poll = null; } };
    await step();
    poll = setInterval(step, 2000);

    $("#deleteJob").onclick = async () => {
      if (!confirm("Delete this analysis and everything it produced?")) return;
      try { await api(`/api/jobs/${id}`, { method: "DELETE" }); } catch (e) { alert(e.message); return; }
      history.pushState({}, "", "/");
      openUpload();
    };
  }

  function openUpload() {
    if (poll) { clearInterval(poll); poll = null; }
    show("#viewUpload"); hide("#viewJob");
    $("#goBtn").disabled = !file;
    $("#goBtn").textContent = "Analyse document";
    renderJobs();
  }

  /* ---------- boot ---------- */
  (async function () {
    try { CFG = await api("/api/config"); }
    catch (e) { document.body.innerHTML = `<div class="wrap" style="padding:60px 24px"><h1>Backend unavailable</h1><p class="small">${esc(e.message)}</p></div>`; return; }
    profile = CFG.default_profile;
    renderProfiles();
    wireUpload();
    $("#navNew").addEventListener("click", e => { e.preventDefault(); history.pushState({}, "", "/"); openUpload(); });
    addEventListener("popstate", () => { const id = jobFromUrl(); id ? openJob(id) : openUpload(); });
    const id = jobFromUrl();
    if (id) openJob(id); else openUpload();
  })();
})();
