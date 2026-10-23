/* Browser smoke test, injected into a copy of the explorer by browser_check.py (never shipped).
 * Visits every view of every section, then on each chart: hovers it, toggles the first legend
 * entry twice and opens the table view. Appends the JSON report to the page in a pre element
 * with id smoke-result. */
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("explorer-data").textContent);
  var report = { views: 0, charts: 0, hovers: 0, tooltips: 0, legends: 0, tables: 0, failures: [] };
  var errors = [];
  window.addEventListener("error", function (e) { errors.push(e.message || String(e)); });

  function visit(hash) {
    history.replaceState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  }

  function fail(where, what) { report.failures.push(where + ": " + what); }

  function center(el) {
    var r = el.getBoundingClientRect();
    return { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
  }

  function hover(fig, where) {
    var tip = document.getElementById("tooltip");
    var target = fig.querySelector("rect[tabindex]") ||
      fig.querySelector("svg circle[r='12']") ||
      fig.querySelector("svg rect[rx='3']") ||
      fig.querySelector("svg rect[style*='transparent']");
    if (!target) { fail(where, "no hover target"); return; }
    report.hovers++;
    var pos = center(target);
    if (target.tagName === "circle") pos = { clientX: pos.clientX, clientY: pos.clientY };
    target.dispatchEvent(new PointerEvent("pointermove", Object.assign({ bubbles: true }, pos)));
    if (!tip.hidden && tip.textContent.length > 0) report.tooltips++;
    else (report.noTooltip = report.noTooltip || []).push(where + " [" + target.tagName + "]");
    target.dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
  }

  function checkView(where) {
    var main = document.getElementById("main");
    if (main.querySelector("[data-error]")) fail(where, main.querySelector("[data-error]").textContent);
    var figs = main.querySelectorAll("figure.figure");
    if (!figs.length) fail(where, "no charts");
    figs.forEach(function (fig, i) {
      var w = where + " chart " + i;
      report.charts++;
      if (!fig.querySelector("svg.chart-svg")) fail(w, "no svg");
      hover(fig, w);
      var leg = fig.querySelector(".legend button");
      if (leg) {
        leg.click();
        var again = fig.querySelector(".legend button");
        if (!fig.querySelector("svg.chart-svg")) fail(w, "legend toggle removed the chart");
        if (again) again.click();
        report.legends++;
      }
      var tb = fig.querySelector(".tbl-btn");
      tb.click();
      var rows = fig.querySelectorAll("table.data tbody tr").length;
      if (!rows) fail(w, "empty table view");
      else report.tables++;
      fig.querySelector(".tbl-btn").click();
    });
  }

  function combos(controls) {
    var out = [{}];
    controls.forEach(function (c) {
      var next = [];
      out.forEach(function (o) {
        c.options.forEach(function (opt) { var n = Object.assign({}, o); n[c.id] = opt.value; next.push(n); });
      });
      out = next;
    });
    return out;
  }

  visit("#/");
  data.phases.forEach(function (p) {
    visit("#/" + p.id);
    p.sections.forEach(function (sec) {
      combos(sec.controls).forEach(function (vals) {
        var q = Object.keys(vals).map(function (k) { return k + "=" + encodeURIComponent(vals[k]); }).join("&");
        var where = p.id + "/" + sec.id + (q ? "?" + q : "");
        visit("#/" + where);
        report.views++;
        checkView(where);
      });
    });
  });
  errors.forEach(function (e) { fail("window", e); });
  var pre = document.createElement("pre");
  pre.id = "smoke-result";
  pre.textContent = JSON.stringify(report);
  document.body.appendChild(pre);
})();
