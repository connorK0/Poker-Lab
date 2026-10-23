/* Results explorer: renders the JSON built by experiments/explorer/build.py.
 *
 * No dependencies. Pages: home -> phase overview -> section. A section has controls (one row,
 * above everything it scopes), and each combination of control values selects a precomputed
 * view: a list of chart specs. Four chart types are drawn as inline SVG: line, dots, heatmap,
 * scatter. Every chart has a legend that toggles series, a hover/focus tooltip, and a table
 * view with the same numbers. Text from the data is always inserted with textContent.
 */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("explorer-data").textContent);
  var NAV = document.getElementById("nav");
  var MAIN = document.getElementById("main");
  var TIP = document.getElementById("tooltip");
  var SVGNS = "http://www.w3.org/2000/svg";
  var MINUS = "−";

  var state = { route: { phase: null, section: null }, values: {}, hidden: {}, tables: {} };

  // ---------------------------------------------------------------- DOM helpers

  function h(tag, props) {
    var node = document.createElement(tag);
    props = props || {};
    Object.keys(props).forEach(function (k) {
      var v = props[k];
      if (v == null) return;
      if (k === "text") node.textContent = v;
      else if (k === "className") node.className = v;
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }

  function append(node, child) {
    if (child == null || child === false) return;
    if (Array.isArray(child)) child.forEach(function (c) { append(node, c); });
    else if (typeof child === "string") node.appendChild(document.createTextNode(child));
    else node.appendChild(child);
  }

  function s(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] != null) node.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function stext(parent, x, y, str, cls, anchor, extra) {
    var t = s("text", Object.assign({ x: x, y: y, class: cls, "text-anchor": anchor || "start" }, extra || {}), parent);
    t.textContent = str;
    return t;
  }

  function color(token) {
    return token === "muted" ? "var(--muted)" : "var(--" + token + ")";
  }

  var measureCtx = document.createElement("canvas").getContext("2d");
  function textWidth(str, size) {
    measureCtx.font = (size || 11) + "px system-ui, -apple-system, 'Segoe UI', sans-serif";
    return measureCtx.measureText(str).width;
  }

  function truncate(str, size, maxW) {
    if (textWidth(str, size) <= maxW) return str;
    var lo = 0, hi = str.length;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      if (textWidth(str.slice(0, mid) + "…", size) <= maxW) lo = mid; else hi = mid - 1;
    }
    return str.slice(0, lo) + "…";
  }

  // ---------------------------------------------------------------- number formatting

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function compact(v) {
    var a = Math.abs(v);
    if (a >= 1e6) return trim(v / 1e6, 1) + "M";
    if (a >= 1e4) return trim(v / 1e3, 1) + "K";
    return Math.round(v).toLocaleString("en-US");
  }

  function trim(v, d) { return Number(v.toFixed(d)).toString(); }

  function sig3(v) {
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1000) return Math.round(v).toLocaleString("en-US");
    if (a < 0.001) return v.toExponential(2).replace("-", MINUS);
    return Number(v.toPrecision(3)).toString().replace("-", MINUS);
  }

  function fmt(v, kind) {
    if (!isNum(v)) return "–";
    switch (kind) {
      case "int": return Math.round(v).toLocaleString("en-US");
      case "pct": return (v * 100).toFixed(1) + "%";
      case "sci": return Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? sig3(v) : v.toExponential(2).replace("-", MINUS);
      case "signed": return (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(3);
      case "x": return "×" + sig3(v);
      default: return sig3(v);
    }
  }

  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  function pow10(k) { return "10" + String(k).split("").map(function (c) { return SUP[c]; }).join(""); }

  function decimals(step) { return Math.max(0, -Math.floor(Math.log10(step) + 1e-9)); }

  /* ``sup``: the log axis spans values below 0.001 or above 10,000, so every tick is written
     as a power of ten (or m×10^k) for one consistent notation. */
  function fmtTick(v, ax, step, sup) {
    var kind = ax.format;
    if (ax.scale === "log") {
      var k = Math.floor(Math.log10(v) + 1e-9);
      var m = Math.round(v / Math.pow(10, k));
      if (kind === "int") return compact(v);
      if (kind === "pct") return trim(v * 100, 2) + "%";
      if (!sup) return trim(v, 4);
      return m === 1 ? pow10(k) : m + "×" + pow10(k);
    }
    if (kind === "pct") return trim(v * 100, Math.max(0, decimals(step * 100))) + "%";
    if (kind === "int") return compact(v);
    var str = v.toFixed(decimals(step));
    if (kind === "signed" && v > 0) str = "+" + str;
    return str.replace("-", MINUS);
  }

  // ---------------------------------------------------------------- scales and ticks

  function niceStep(span, count) {
    var raw = span / Math.max(1, count);
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var e = raw / p;
    return (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * p;
  }

  function niceLogBound(v, up) {
    var k = Math.floor(Math.log10(v));
    var cands = [1, 2, 5, 10].map(function (m) { return m * Math.pow(10, k); });
    if (up) { for (var i = 0; i < cands.length; i++) if (cands[i] >= v * 0.999999) return cands[i]; }
    else { for (var j = cands.length - 1; j >= 0; j--) if (cands[j] <= v * 1.000001) return cands[j]; }
    return v;
  }

  /* Build a scale for one axis: {map, ticks, step, d0, d1}. */
  function makeAxis(ax, values, r0, r1, tickCount) {
    var vals = values.filter(isNum);
    if (ax.scale === "point") {
      var n = ax.categories.length;
      var band = (r1 - r0) / n;
      return {
        map: function (i) { return r0 + (i + 0.5) * band; },
        ticks: ax.categories.map(function (_, i) { return i; }),
        label: function (i) { return ax.categories[i]; },
        d0: -0.5, d1: n - 0.5, step: 1, invert: function (px) { return Math.round((px - r0) / band - 0.5); }
      };
    }
    if (ax.scale === "log") {
      var pos = vals.filter(function (v) { return v > 0; });
      var lo = pos.length ? Math.min.apply(null, pos) : 1, hi = pos.length ? Math.max.apply(null, pos) : 10;
      if (isNum(ax.min) && ax.min > 0) lo = ax.min;
      if (isNum(ax.max)) hi = ax.max;
      if (hi <= lo) { hi = lo * 10; }
      var d0 = niceLogBound(lo, false), d1 = niceLogBound(hi, true);
      var l0 = Math.log10(d0), l1 = Math.log10(d1);
      var decades = l1 - l0;
      var mults = decades <= 1.01 ? [1, 2, 3, 5] : decades <= 3.01 ? [1, 2, 5] : [1];
      var every = decades > 9 ? 2 : 1;
      var ticks = [];
      for (var k = Math.floor(l0); k <= Math.ceil(l1); k++) {
        mults.forEach(function (m) {
          var v = m * Math.pow(10, k);
          if (v >= d0 * 0.9999 && v <= d1 * 1.0001 && (m !== 1 || ((k % every) + every) % every === 0)) ticks.push(v);
        });
      }
      var map = function (v) { return r0 + (Math.log10(Math.max(v, d0 * 1e-6)) - l0) / (l1 - l0) * (r1 - r0); };
      var sup = d0 < 0.001 || d1 > 1e4;
      return { map: map, ticks: ticks, d0: d0, d1: d1, step: null, label: function (v) { return fmtTick(v, ax, null, sup); } };
    }
    var mn = vals.length ? Math.min.apply(null, vals) : 0, mx = vals.length ? Math.max.apply(null, vals) : 1;
    if (isNum(ax.min)) mn = Math.min(ax.min, mn);
    if (isNum(ax.max)) mx = Math.max(ax.max, mx);
    if (mx === mn) { mx = mn + (Math.abs(mn) || 1) * 0.5; mn = mn - (Math.abs(mn) || 1) * 0.5; }
    var step = niceStep(mx - mn, tickCount || 5);
    var a0 = isNum(ax.min) && ax.min <= mn ? ax.min : Math.floor(mn / step) * step;
    var a1 = isNum(ax.max) && ax.max >= mx ? ax.max : Math.ceil(mx / step) * step;
    var t = [];
    for (var v = Math.ceil(a0 / step - 1e-9) * step; v <= a1 + step * 1e-9; v += step) t.push(Number(v.toFixed(12)));
    return {
      map: function (x) { return r0 + (x - a0) / (a1 - a0) * (r1 - r0); },
      ticks: t, d0: a0, d1: a1, step: step,
      label: function (x) { return fmtTick(x, ax, step); }
    };
  }

  // ---------------------------------------------------------------- tooltip

  function showTip(evt, build) {
    TIP.textContent = "";
    build(TIP);
    TIP.hidden = false;
    var x, y;
    if (evt && evt.clientX != null) { x = evt.clientX; y = evt.clientY; }
    else if (evt && evt.anchor) { var r = evt.anchor.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top; }
    else { x = 20; y = 20; }
    var w = TIP.offsetWidth, hgt = TIP.offsetHeight;
    var left = x + 16, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 16;
    if (top + hgt > window.innerHeight - 8) top = y - hgt - 14;
    TIP.style.left = Math.max(8, left) + "px";
    TIP.style.top = Math.max(8, top) + "px";
  }

  function hideTip() { TIP.hidden = true; }

  function tipRow(parent, tok, value, name, ciText, dot) {
    var row = h("div", { className: "tt-row" },
      h("span", { className: "tt-key" + (dot ? " dot" : ""), style: dot ? "background:" + color(tok) : "border-color:" + color(tok) }),
      h("span", { className: "tt-val", text: value }),
      h("span", { className: "tt-name", text: name }),
      ciText ? h("span", { className: "tt-ci", text: ciText }) : null);
    parent.appendChild(row);
  }

  function ciText(p, kind) {
    if (p.length > 3 && isNum(p[2]) && isNum(p[3])) return "[" + fmt(p[2], kind) + ", " + fmt(p[3], kind) + "]";
    return "";
  }

  // ---------------------------------------------------------------- legend

  function legendKey(item, kind) {
    var svg = s("svg", { width: 18, height: 12, class: "key", "aria-hidden": "true" });
    if (kind === "rect") s("rect", { x: 1, y: 2, width: 16, height: 8, rx: 2, style: "fill:" + color(item.color) }, svg);
    else if (item.line === false || kind === "dot") drawMarker(svg, item.shape, 9, 6, 4, item.color);
    else {
      s("line", { x1: 1, y1: 6, x2: 17, y2: 6, style: "stroke:" + color(item.color) + ";stroke-width:2.5;stroke-linecap:round" + (item.dash ? ";stroke-dasharray:4 3" : "") }, svg);
    }
    return svg;
  }

  function renderLegend(items, hiddenKey, rerender, kind) {
    var names = [], byName = {};
    items.forEach(function (it) {
      var name = it.group || it.name;
      if (!(name in byName)) { byName[name] = it; names.push(name); }
    });
    if (names.length < 2) return null;
    var wrap = h("div", { className: "legend", role: "group", "aria-label": "Series (click to hide or show)" });
    names.forEach(function (name) {
      var key = hiddenKey + "|" + name;
      var on = !isHidden(key, byName[name]);
      var btn = h("button", {
        type: "button", "aria-pressed": String(on), title: on ? "Hide " + name : "Show " + name,
        onclick: function () { state.hidden[key] = on; rerender(); }
      }, legendKey(byName[name], kind), name);
      wrap.appendChild(btn);
    });
    wrap.appendChild(h("span", { className: "legend-hint", text: "click to hide/show" }));
    return wrap;
  }

  function isHidden(key, item) {
    if (key in state.hidden) return state.hidden[key];
    return !!item.hidden;
  }

  // ---------------------------------------------------------------- markers

  function drawMarker(parent, shape, x, y, r, tok, extra) {
    var style = "fill:" + color(tok) + ";stroke:var(--surface);stroke-width:2";
    var a = Object.assign({ style: style }, extra || {});
    switch (shape) {
      case "square":
        return s("rect", Object.assign({ x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8, rx: 1 }, a), parent);
      case "triangle":
        return s("path", Object.assign({ d: "M" + x + "," + (y - r * 1.15) + "L" + (x + r * 1.05) + "," + (y + r * 0.8) + "L" + (x - r * 1.05) + "," + (y + r * 0.8) + "Z" }, a), parent);
      case "diamond":
        return s("path", Object.assign({ d: "M" + x + "," + (y - r * 1.25) + "L" + (x + r * 1.1) + "," + y + "L" + x + "," + (y + r * 1.25) + "L" + (x - r * 1.1) + "," + y + "Z" }, a), parent);
      case "cross":
        var w = r * 0.45, L = r * 1.1;
        return s("path", Object.assign({ d: "M" + (x - w) + "," + (y - L) + "h" + 2 * w + "v" + (L - w) + "h" + (L - w) + "v" + 2 * w + "h" + (w - L) + "v" + (L - w) + "h" + -2 * w + "v" + (w - L) + "h" + (w - L) + "v" + -2 * w + "h" + (L - w) + "Z" }, a), parent);
      default:
        return s("circle", Object.assign({ cx: x, cy: y, r: r }, a), parent);
    }
  }

  // ---------------------------------------------------------------- shared chart frame

  function frame(box, W, H) {
    var svg = s("svg", { class: "chart-svg", viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img" });
    box.appendChild(svg);
    return svg;
  }

  function drawYAxis(svg, Y, ax, x0, x1, labelX) {
    var g = s("g", {}, svg);
    Y.ticks.forEach(function (t) {
      var y = Y.map(t);
      s("line", { x1: x0, x2: x1, y1: y, y2: y, style: "stroke:var(--grid);stroke-width:1" }, g);
      stext(g, labelX, y + 3.5, Y.label(t), "tick", "end");
    });
  }

  function drawXAxis(svg, X, ax, y, x0, x1, vertical, yTop) {
    var g = s("g", {}, svg);
    s("line", { x1: x0, x2: x1, y1: y, y2: y, style: "stroke:var(--axis);stroke-width:1" }, g);
    var lastRight = -1e9;
    X.ticks.forEach(function (t) {
      var x = X.map(t);
      if (vertical) s("line", { x1: x, x2: x, y1: yTop, y2: y, style: "stroke:var(--grid);stroke-width:1" }, g);
      s("line", { x1: x, x2: x, y1: y, y2: y + 4, style: "stroke:var(--axis);stroke-width:1" }, g);
      var lab = X.label(t);
      var w = textWidth(lab, 11);
      if (x - w / 2 > lastRight + 6) {
        stext(g, x, y + 17, lab, "tick", "middle");
        lastRight = x + w / 2;
      }
    });
  }

  function yAxisWidth(Y) {
    var w = 0;
    Y.ticks.forEach(function (t) { w = Math.max(w, textWidth(Y.label(t), 11)); });
    return w;
  }

  /* Reference lines with labels. A label goes at the right or left end of its line, whichever
     covers fewer data points (``pts`` are pixel positions), and has a surface halo. */
  function drawRefs(svg, refs, X, Y, x0, x1, y0, y1, pts) {
    var placed = [];
    function clutter(xa, xb, ya, yb) {
      var n = 0;
      (pts || []).forEach(function (p) { if (p[0] >= xa && p[0] <= xb && p[1] >= ya && p[1] <= yb) n++; });
      return n;
    }
    (refs || []).forEach(function (r) {
      if (!isNum(r.value)) return;
      if (r.axis === "x") {
        var x = X.map(r.value);
        if (x < x0 - 1 || x > x1 + 1) return;
        s("line", { x1: x, x2: x, y1: y0, y2: y1, style: "stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4" }, svg);
        stext(svg, x + 4, y0 + 11, r.label, "ref-label", "start");
        return;
      }
      if (!Y) return;
      var y = Y.map(r.value);
      if (y < y0 - 1 || y > y1 + 1) return;
      s("line", { x1: x0, x2: x1, y1: y, y2: y, style: "stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4" }, svg);
      // Four candidate spots (either end, above or below the line): least clutter wins, and a
      // spot another label already took is skipped.
      var w = textWidth(r.label, 11) + 8;
      var best = null;
      [[true, true], [false, true], [true, false], [false, false]].forEach(function (c) {
        var right = c[0], above = c[1];
        var ty = above ? y - 4 : y + 13;
        if (ty - 11 < y0 - 2 || ty > y1) return;
        if (placed.some(function (p) { return p.right === right && Math.abs(p.ty - ty) < 13; })) return;
        var n = right ? clutter(x1 - w, x1, ty - 11, ty + 3) : clutter(x0, x0 + w, ty - 11, ty + 3);
        if (!best || n < best.n) best = { n: n, right: right, ty: ty };
      });
      if (!best) best = { right: true, ty: y - 4 };
      stext(svg, best.right ? x1 - 4 : x0 + 6, best.ty, r.label, "ref-label", best.right ? "end" : "start");
      placed.push({ ty: best.ty, right: best.right });
    });
  }

  // ---------------------------------------------------------------- line chart

  function renderLine(box, spec, hk) {
    var W = Math.max(280, box.clientWidth || 600);
    var H = Math.round(Math.min(400, Math.max(270, W * 0.56)));
    var visible = spec.series.filter(function (se) { return !isHidden(hk + "|" + (se.group || se.name), se); });
    var xs = [], ys = [];
    spec.series.forEach(function (se) {
      se.points.forEach(function (p) {
        xs.push(p[0]); ys.push(p[1]);
        if (p.length > 3) { ys.push(p[2]); ys.push(p[3]); }
      });
    });
    (spec.refs || []).forEach(function (r) { (r.axis === "x" ? xs : ys).push(r.value); });
    if (spec.y.scale === "log") ys = ys.filter(function (v) { return v > 0; });
    var top = 26, bottom = 46, right = 14;
    var Y0 = makeAxis(spec.y, ys, H - bottom, top, 5);
    var left = Math.ceil(yAxisWidth(Y0)) + 14;
    var X = makeAxis(spec.x, xs, left, W - right, Math.max(3, Math.floor(W / 110)));
    var Y = Y0;
    var svg = frame(box, W, H);
    svg.setAttribute("aria-label", spec.title);
    stext(svg, 0, 13, truncate(spec.y.label, 12, W), "axis-title", "start");
    drawYAxis(svg, Y, spec.y, left, W - right, left - 8);
    drawXAxis(svg, X, spec.x, H - bottom, left, W - right, false);
    stext(svg, left + (W - right - left) / 2, H - 8, truncate(spec.x.label, 12, W - right - left), "axis-title", "middle");
    var clipId = "clip" + Math.random().toString(36).slice(2);
    var defs = s("defs", {}, svg);
    var cp = s("clipPath", { id: clipId }, defs);
    s("rect", { x: left, y: top - 6, width: W - right - left, height: H - bottom - top + 12 }, cp);
    var plot = s("g", { "clip-path": "url(#" + clipId + ")" }, svg);

    function px(p) { return X.map(p[0]); }
    function py(v) { return Y.map(spec.y.scale === "log" ? Math.max(v, Y.d0 * 1e-3) : v); }

    visible.forEach(function (se) {
      var pts = se.points.filter(function (p) { return p.length > 3 && isNum(p[2]) && isNum(p[3]); });
      if (pts.length < 2) return;
      var d = "M" + pts.map(function (p) { return px(p) + "," + py(p[3]); }).join("L") +
        "L" + pts.slice().reverse().map(function (p) { return px(p) + "," + py(p[2]); }).join("L") + "Z";
      s("path", { d: d, style: "fill:" + color(se.color) + ";fill-opacity:0.12;stroke:none" }, plot);
    });
    var pix = [];
    visible.forEach(function (se) {
      se.points.forEach(function (p, i) {
        pix.push([px(p), py(p[1])]);
        var q = se.points[i + 1];  // midpoints too, so a label avoids the segments between markers
        if (q && se.line !== false) pix.push([(px(p) + px(q)) / 2, (py(p[1]) + py(q[1])) / 2]);
      });
    });
    drawRefs(svg, spec.refs, X, Y, left, W - right, top, H - bottom, pix);
    visible.forEach(function (se) {
      if (se.line === false || se.points.length < 2) return;
      var d = "M" + se.points.map(function (p) { return px(p) + "," + py(p[1]); }).join("L");
      s("path", {
        d: d,
        style: "fill:none;stroke:" + color(se.color) + ";stroke-width:2;stroke-linejoin:round;stroke-linecap:round" +
          (se.dash ? ";stroke-dasharray:6 5" : "") + (se.note ? ";stroke-opacity:0.75" : "")
      }, plot);
    });
    visible.forEach(function (se) {
      var showMarkers = se.marker !== false && se.points.length <= 40;
      if (!showMarkers && se.line !== false) return;
      se.points.forEach(function (p) { drawMarker(plot, se.shape, px(p), py(p[1]), se.line === false ? 5 : 4, se.color); });
    });

    // hover layer: a crosshair that snaps to the nearest x (shared grids) or nearest point
    var allX = {};
    visible.forEach(function (se) { se.points.forEach(function (p) { allX[p[0]] = true; }); });
    var xKeys = Object.keys(allX).map(Number).sort(function (a, b) { return a - b; });
    var maxLen = Math.max.apply(null, visible.map(function (se) { return se.points.length; }).concat([1]));
    var shared = xKeys.length <= maxLen * 1.25;
    var hover = s("g", { "pointer-events": "none" }, svg);
    var overlay = s("rect", { x: left, y: top, width: W - right - left, height: H - bottom - top, style: "fill:transparent", tabindex: "0", class: "hover-target", "aria-label": spec.title + ": use arrow keys to read values" }, svg);
    var idx = -1;

    function showAt(i, evt) {
      hover.textContent = "";
      if (!xKeys.length) return;
      idx = Math.max(0, Math.min(xKeys.length - 1, i));
      var xv = xKeys[idx], x = X.map(xv);
      s("line", { x1: x, x2: x, y1: top, y2: H - bottom, style: "stroke:var(--axis);stroke-width:1" }, hover);
      var rows = [];
      visible.forEach(function (se) {
        var p = se.points.find(function (q) { return q[0] === xv; });
        if (p) rows.push({ se: se, p: p });
      });
      rows.sort(function (a, b) { return b.p[1] - a.p[1]; });
      rows.forEach(function (r) { drawMarker(hover, r.se.shape, x, py(r.p[1]), 5, r.se.color); });
      var xlabel = spec.x.scale === "point" ? spec.x.categories[xv] : fmt(xv, spec.x.format);
      showTip(evt || { anchor: overlay }, function (tip) {
        tip.appendChild(h("div", { className: "tt-head", text: spec.x.label + ": " + xlabel }));
        rows.forEach(function (r) {
          tipRow(tip, r.se.color, fmt(r.p[1], spec.y.format), r.se.name + (r.se.note ? " · " + r.se.note : ""), ciText(r.p, spec.y.format));
        });
      });
    }

    function nearest(evt) {
      var rect = svg.getBoundingClientRect();
      var scale = W / rect.width;
      var mx = (evt.clientX - rect.left) * scale, my = (evt.clientY - rect.top) * scale;
      if (shared) {
        var best = 0, bd = Infinity;
        xKeys.forEach(function (xv, i) { var d = Math.abs(X.map(xv) - mx); if (d < bd) { bd = d; best = i; } });
        showAt(best, evt);
        return;
      }
      var hit = null, hd = Infinity;
      visible.forEach(function (se) {
        se.points.forEach(function (p) {
          var dx = px(p) - mx, dy = py(p[1]) - my, d = dx * dx + dy * dy;
          if (d < hd) { hd = d; hit = { se: se, p: p }; }
        });
      });
      hover.textContent = "";
      if (!hit) return;
      var x = px(hit.p);
      s("line", { x1: x, x2: x, y1: top, y2: H - bottom, style: "stroke:var(--axis);stroke-width:1" }, hover);
      drawMarker(hover, hit.se.shape, x, py(hit.p[1]), 5, hit.se.color);
      showTip(evt, function (tip) {
        tip.appendChild(h("div", { className: "tt-head", text: hit.se.name + (hit.se.note ? " · " + hit.se.note : "") }));
        tipRow(tip, hit.se.color, fmt(hit.p[1], spec.y.format), spec.y.label, ciText(hit.p, spec.y.format));
        tip.appendChild(h("div", { className: "tt-ci", text: spec.x.label + ": " + fmt(hit.p[0], spec.x.format) }));
      });
    }

    overlay.addEventListener("pointermove", nearest);
    overlay.addEventListener("pointerleave", function () { hover.textContent = ""; hideTip(); });
    overlay.addEventListener("blur", function () { hover.textContent = ""; hideTip(); });
    overlay.addEventListener("focus", function () { if (shared) showAt(idx < 0 ? xKeys.length - 1 : idx); });
    overlay.addEventListener("keydown", function (e) {
      if (!shared) return;
      if (e.key === "ArrowLeft") { showAt(idx - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { showAt(idx + 1); e.preventDefault(); }
    });
  }

  // ---------------------------------------------------------------- dot plot

  function renderDots(box, spec, hk) {
    var W = Math.max(280, box.clientWidth || 600);
    var visible = spec.series.filter(function (se) { return !isHidden(hk + "|" + se.name, se); });
    var k = Math.max(1, visible.length);
    var rowH = k > 2 ? 40 : 34;
    var top = 10, bottom = 46, right = 16;
    var H = top + spec.rows.length * rowH + bottom;
    var labelW = Math.min(W * 0.46, 300);
    var maxLab = 0;
    spec.rows.forEach(function (r) { maxLab = Math.max(maxLab, textWidth(r, 12)); });
    var left = Math.min(labelW, maxLab) + 16;
    var vals = [];
    spec.series.forEach(function (se) { se.values.forEach(function (v) { if (v) v.forEach(function (u) { vals.push(u); }); }); });
    (spec.ranges || []).forEach(function (r) { if (r) { vals.push(r[0]); vals.push(r[1]); } });
    (spec.refs || []).forEach(function (r) { vals.push(r.value); });
    if (spec.x.scale === "log") vals = vals.filter(function (v) { return v > 0; });
    var X = makeAxis(spec.x, vals, left, W - right, Math.max(3, Math.floor(W / 110)));
    var svg = frame(box, W, H);
    svg.setAttribute("aria-label", spec.title);
    var y1 = H - bottom;
    drawXAxis(svg, X, spec.x, y1, left, W - right, true, top);
    stext(svg, left + (W - right - left) / 2, H - 8, truncate(spec.x.label, 12, W - right - left), "axis-title", "middle");
    drawRefs(svg, spec.refs, X, null, left, W - right, top, y1);
    spec.rows.forEach(function (row, i) {
      var cy = top + (i + 0.5) * rowH;
      var lab = stext(svg, left - 10, cy + 4, truncate(row, 12, left - 12), "row-label", "end");
      if (lab.textContent !== row) s("title", {}, lab).textContent = row;
      var rg = spec.ranges && spec.ranges[i];
      if (rg) {
        var r = s("rect", { x: X.map(rg[0]), y: cy - 9, width: Math.max(2, X.map(rg[1]) - X.map(rg[0])), height: 18, rx: 4, style: "fill:var(--wash)" }, svg);
        r.addEventListener("pointermove", function (e) {
          showTip(e, function (tip) {
            tip.appendChild(h("div", { className: "tt-head", text: row }));
            tip.appendChild(h("div", { text: (spec.rangeLabel || "Range") + ": " + fmt(rg[0], spec.x.format) + " to " + fmt(rg[1], spec.x.format) }));
          });
        });
        r.addEventListener("pointerleave", hideTip);
      }
      visible.forEach(function (se, j) {
        var v = se.values[i];
        if (!v || !isNum(v[0])) return;
        var dy = (j - (k - 1) / 2) * (k > 2 ? 9 : 10);
        var y = cy + dy;
        if (v.length > 2 && isNum(v[1]) && isNum(v[2])) {
          s("line", { x1: X.map(v[1]), x2: X.map(v[2]), y1: y, y2: y, style: "stroke:" + color(se.color) + ";stroke-width:1.5;stroke-linecap:round" }, svg);
        }
        var dot = drawMarker(svg, "circle", X.map(v[0]), y, 5, se.color);
        var hit = s("circle", { cx: X.map(v[0]), cy: y, r: 12, style: "fill:transparent" }, svg);
        hit.addEventListener("pointermove", function (e) {
          dot.setAttribute("r", 6.5);
          showTip(e, function (tip) {
            tip.appendChild(h("div", { className: "tt-head", text: row }));
            tipRow(tip, se.color, fmt(v[0], spec.x.format), se.name, v.length > 2 && isNum(v[1]) ? "[" + fmt(v[1], spec.x.format) + ", " + fmt(v[2], spec.x.format) + "]" : "", true);
            if (rg) tip.appendChild(h("div", { className: "tt-ci", text: (spec.rangeLabel || "Range") + ": " + fmt(rg[0], spec.x.format) + " to " + fmt(rg[1], spec.x.format) }));
          });
        });
        hit.addEventListener("pointerleave", function () { dot.setAttribute("r", 5); hideTip(); });
      });
    });
  }

  // ---------------------------------------------------------------- heatmap

  function renderHeatmap(box, spec) {
    var W = Math.max(280, box.clientWidth || 600);
    var maxLab = 0;
    spec.rows.forEach(function (r) { maxLab = Math.max(maxLab, textWidth(r, 12)); });
    var left = Math.min(maxLab + 14, W * 0.3);
    var top = 46, right = 8, bottom = 10;
    var cw = (W - left - right) / spec.cols.length;
    var ch = Math.max(32, Math.min(56, cw * 0.62));
    var H = top + ch * spec.rows.length + bottom;
    var vmax = isNum(spec.max) ? spec.max : Math.max.apply(null, spec.values.map(function (r) { return Math.max.apply(null, r); }));
    var svg = frame(box, W, H);
    svg.setAttribute("aria-label", spec.title);
    if (spec.colLabel) stext(svg, left, 12, spec.colLabel + " →", "axis-title", "start");
    if (spec.rowLabel) stext(svg, 0, 12, spec.rowLabel + " ↓", "axis-title", "start");
    spec.cols.forEach(function (c, j) {
      stext(svg, left + (j + 0.5) * cw, top - 8, truncate(c, 11.5, cw - 4), "tick", "middle");
    });
    spec.rows.forEach(function (r, i) {
      stext(svg, left - 10, top + (i + 0.5) * ch + 4, truncate(r, 12, left - 14), "row-label", "end");
      spec.values[i].forEach(function (v, j) {
        var step = v <= 0 ? 0 : Math.min(8, 1 + Math.floor((v / vmax) * 7.999));
        var x = left + j * cw, y = top + i * ch;
        var cell = s("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, style: "fill:var(--seq" + step + ")" }, svg);
        var str = fmt(v, spec.format);
        var cellText = spec.labels ? spec.labels[i][j] : str;
        if (textWidth(cellText, 11.5) < cw - 6) {
          s("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "cell-label", style: "fill:var(" + (step >= 5 ? "--seq-ink-hi" : "--seq-ink-lo") + ")" }, svg).textContent = cellText;
        }
        cell.addEventListener("pointermove", function (e) {
          cell.style.stroke = "var(--ink)"; cell.style.strokeWidth = "1.5";
          showTip(e, function (tip) {
            var head = spec.labels ? spec.labels[i][j] : (spec.rowLabel || "Row") + ": " + r + " → " + (spec.colLabel || "Column") + ": " + spec.cols[j];
            tip.appendChild(h("div", { className: "tt-head", text: head }));
            tip.appendChild(h("div", { className: "tt-val", text: str }));
            if (spec.counts) tip.appendChild(h("div", { className: "tt-ci", text: spec.counts[i][j].toLocaleString("en-US") + " of " + spec.counts[i].reduce(function (a, b) { return a + b; }, 0).toLocaleString("en-US") }));
          });
        });
        cell.addEventListener("pointerleave", function () { cell.style.stroke = ""; hideTip(); });
      });
    });
  }

  // ---------------------------------------------------------------- scatter

  function groupPoints(g, tables) {
    if (!g.table) return g.points;
    var t = tables[g.table];
    var out = [];
    var col = t[g.where ? g.where[0] : g.x];
    for (var i = 0; i < col.length; i++) {
      if (g.where && t[g.where[0]][i] !== g.where[1]) continue;
      var x = t[g.x][i], y = t[g.y][i];
      if (isNum(x) && isNum(y)) out.push([x, y, t[g.label] ? t[g.label][i] : ""]);
    }
    return out;
  }

  function renderScatter(box, spec, hk, tables) {
    var W = Math.max(280, box.clientWidth || 600);
    var H = Math.round(Math.min(420, Math.max(280, W * 0.6)));
    var groups = spec.groups.map(function (g) { return Object.assign({}, g, { pts: groupPoints(g, tables) }); });
    var visible = groups.filter(function (g) { return !isHidden(hk + "|" + g.name, g); });
    var xs = [], ys = [];
    groups.forEach(function (g) { g.pts.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
    (spec.refs || []).forEach(function (r) { (r.axis === "x" ? xs : ys).push(r.value); });
    var top = 26, bottom = 46, right = 16;
    var Y = makeAxis(spec.y, ys, H - bottom, top, 5);
    var left = Math.ceil(yAxisWidth(Y)) + 14;
    var X = makeAxis(spec.x, xs, left, W - right, Math.max(3, Math.floor(W / 110)));
    var svg = frame(box, W, H);
    svg.setAttribute("aria-label", spec.title);
    stext(svg, 0, 13, truncate(spec.y.label, 12, W), "axis-title", "start");
    drawYAxis(svg, Y, spec.y, left, W - right, left - 8);
    drawXAxis(svg, X, spec.x, H - bottom, left, W - right, false);
    stext(svg, left + (W - right - left) / 2, H - 8, truncate(spec.x.label, 12, W - right - left), "axis-title", "middle");
    var pix = [];
    visible.forEach(function (g) { g.pts.forEach(function (p) { pix.push([X.map(p[0]), Y.map(p[1])]); }); });
    drawRefs(svg, spec.refs, X, Y, left, W - right, top, H - bottom, pix);
    if (spec.diagonal) {
      var lo = Math.max(X.d0, Y.d0), hi = Math.min(X.d1, Y.d1);
      s("line", { x1: X.map(lo), y1: Y.map(lo), x2: X.map(hi), y2: Y.map(hi), style: "stroke:var(--muted);stroke-dasharray:4 4" }, svg);
    }
    var dense = visible.reduce(function (n, g) { return n + g.pts.length; }, 0) > 120;
    visible.forEach(function (g) {
      if (!g.connect || g.pts.length < 2) return;
      s("path", { d: "M" + g.pts.map(function (p) { return X.map(p[0]) + "," + Y.map(p[1]); }).join("L"), style: "fill:none;stroke:" + color(g.color) + ";stroke-width:2;stroke-linejoin:round" + (g.dash ? ";stroke-dasharray:6 5" : "") }, svg);
    });
    visible.forEach(function (g) {
      g.pts.forEach(function (p) { drawMarker(svg, g.shape, X.map(p[0]), Y.map(p[1]), dense ? 3.6 : 4.5, g.color, dense ? { "fill-opacity": 0.85 } : null); });
    });
    var hover = s("g", { "pointer-events": "none" }, svg);
    var overlay = s("rect", { x: left - 10, y: top - 10, width: W - right - left + 20, height: H - bottom - top + 20, style: "fill:transparent" }, svg);
    overlay.addEventListener("pointermove", function (evt) {
      var rect = svg.getBoundingClientRect();
      var sc = W / rect.width;
      var mx = (evt.clientX - rect.left) * sc, my = (evt.clientY - rect.top) * sc;
      var hit = null, hd = 30 * 30;
      visible.forEach(function (g) {
        g.pts.forEach(function (p) {
          var dx = X.map(p[0]) - mx, dy = Y.map(p[1]) - my, d = dx * dx + dy * dy;
          if (d < hd) { hd = d; hit = { g: g, p: p }; }
        });
      });
      hover.textContent = "";
      if (!hit) { hideTip(); return; }
      drawMarker(hover, hit.g.shape, X.map(hit.p[0]), Y.map(hit.p[1]), 7, hit.g.color);
      showTip(evt, function (tip) {
        tip.appendChild(h("div", { className: "tt-head", text: hit.p[2] || hit.g.name }));
        if (hit.p[2]) tip.appendChild(h("div", { className: "tt-name", text: hit.g.name }));
        tipRow(tip, hit.g.color, fmt(hit.p[1], spec.y.format), spec.y.label, "", true);
        tipRow(tip, hit.g.color, fmt(hit.p[0], spec.x.format), spec.x.label, "", true);
      });
    });
    overlay.addEventListener("pointerleave", function () { hover.textContent = ""; hideTip(); });
  }

  // ---------------------------------------------------------------- table views

  function dataTable(spec, tables) {
    var tbl = h("table", { className: "data" });
    var thead = h("thead"), tbody = h("tbody");
    tbl.appendChild(thead); tbl.appendChild(tbody);
    function row(cells, head) {
      var tr = h("tr");
      cells.forEach(function (c) { tr.appendChild(h(head ? "th" : "td", { text: c })); });
      (head ? thead : tbody).appendChild(tr);
    }
    function valCI(p, kind, i) { return fmt(p[i], kind) + (p.length > i + 2 && isNum(p[i + 1]) ? " [" + fmt(p[i + 1], kind) + ", " + fmt(p[i + 2], kind) + "]" : ""); }
    if (spec.type === "line") {
      var xsAll = {};
      spec.series.forEach(function (se) { se.points.forEach(function (p) { xsAll[p[0]] = true; }); });
      var xk = Object.keys(xsAll).map(Number).sort(function (a, b) { return a - b; });
      var xl = function (x) { return spec.x.scale === "point" ? spec.x.categories[x] : fmt(x, spec.x.format); };
      if (xk.length <= 40 && spec.series.length <= 12) {
        row([spec.x.label].concat(spec.series.map(function (se) { return se.name + (se.note ? " (" + se.note + ")" : ""); })), true);
        xk.forEach(function (x) {
          row([xl(x)].concat(spec.series.map(function (se) {
            var p = se.points.find(function (q) { return q[0] === x; });
            return p ? valCI(p, spec.y.format, 1) : "";
          })));
        });
      } else {
        row(["Series", spec.x.label, spec.y.label], true);
        spec.series.forEach(function (se) {
          se.points.forEach(function (p) { row([se.name + (se.note ? " (" + se.note + ")" : ""), xl(p[0]), valCI(p, spec.y.format, 1)]); });
        });
      }
    } else if (spec.type === "dots") {
      row([""].concat(spec.series.map(function (se) { return se.name; })).concat(spec.ranges ? [spec.rangeLabel || "Range"] : []), true);
      spec.rows.forEach(function (r, i) {
        var cells = [r].concat(spec.series.map(function (se) { var v = se.values[i]; return v ? valCI(v, spec.x.format, 0) : ""; }));
        if (spec.ranges) { var rg = spec.ranges[i]; cells.push(rg ? fmt(rg[0], spec.x.format) + " to " + fmt(rg[1], spec.x.format) : ""); }
        row(cells);
      });
    } else if (spec.type === "heatmap") {
      row([(spec.rowLabel || "") + " \\ " + (spec.colLabel || "")].concat(spec.cols), true);
      spec.rows.forEach(function (r, i) {
        row([r].concat(spec.values[i].map(function (v, j) { return (spec.labels ? spec.labels[i][j] + " " : "") + fmt(v, spec.format) + (spec.counts ? " (" + spec.counts[i][j] + ")" : ""); })));
      });
    } else if (spec.type === "scatter") {
      row(["Group", "Point", spec.x.label, spec.y.label], true);
      spec.groups.forEach(function (g) {
        groupPoints(g, tables).forEach(function (p) { row([g.name, p[2] || "", fmt(p[0], spec.x.format), fmt(p[1], spec.y.format)]); });
      });
    }
    return h("div", { className: "table-wrap" }, tbl);
  }

  // ---------------------------------------------------------------- figures

  var RENDER = { line: renderLine, dots: renderDots, heatmap: renderHeatmap, scatter: renderScatter };

  function figure(spec, hk, tables) {
    var fig = h("figure", { className: "figure" });
    var tblKey = hk + "|table";
    var showTable = !!state.tables[tblKey];
    var body = h("div");
    var drawnWidth = 0;
    function draw() {
      body.textContent = "";
      hideTip();
      drawnWidth = fig.clientWidth;
      if (showTable) { body.appendChild(dataTable(spec, tables)); return; }
      var items = spec.series || spec.groups || [];
      var legend = renderLegend(items, hk, draw, spec.type === "scatter" ? "dot" : spec.type === "dots" ? "dot" : "line");
      if (legend && spec.type === "dots" && spec.ranges) {
        legend.insertBefore(h("span", { className: "legend-hint" }, legendKey({ color: "wash" }, "rect"), " " + (spec.rangeLabel || "")), legend.lastChild);
      } else if (!legend && spec.type === "dots" && spec.ranges) {
        legend = h("div", { className: "legend" }, h("span", { className: "legend-hint" }, legendKey({ color: "wash" }, "rect"), " " + (spec.rangeLabel || "")));
      }
      if (legend) body.appendChild(legend);
      var box = h("div", { className: "chart-box" });
      body.appendChild(box);
      RENDER[spec.type](box, spec, hk, tables);
    }
    var btn = h("button", {
      type: "button", className: "tbl-btn", "aria-pressed": String(showTable),
      onclick: function () { showTable = !showTable; state.tables[tblKey] = showTable; btn.setAttribute("aria-pressed", String(showTable)); btn.textContent = showTable ? "Chart" : "Table"; draw(); }
    }, showTable ? "Chart" : "Table");
    fig.appendChild(h("div", { className: "fig-head" },
      h("div", {}, h("div", { className: "fig-title", text: spec.title }), spec.subtitle ? h("div", { className: "fig-sub", text: spec.subtitle }) : null),
      h("div", { className: "fig-tools" }, btn)));
    fig.appendChild(body);
    fig._draw = draw;
    // Charts are drawn at their container's pixel width, so redraw when that width changes
    // (grid columns settling, scrollbars, window resizes).
    if (window.ResizeObserver) {
      var pending = null;
      new ResizeObserver(function () {
        if (!fig.isConnected || Math.abs(fig.clientWidth - drawnWidth) < 6) return;
        clearTimeout(pending);
        pending = setTimeout(draw, 60);
      }).observe(fig);
    }
    return fig;
  }

  // ---------------------------------------------------------------- routing

  function findPhase(id) { return DATA.phases.find(function (p) { return p.id === id; }); }
  function findSection(phase, id) { return phase && phase.sections.find(function (sec) { return sec.id === id; }); }

  function parseHash() {
    var raw = (location.hash || "").replace(/^#\/?/, "");
    var q = raw.split("?");
    var parts = q[0].split("/").filter(Boolean);
    var params = {};
    (q[1] || "").split("&").forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf("=");
      params[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1));
    });
    return { phase: parts[0] || null, section: parts[1] || null, params: params };
  }

  function hashFor(phase, section, params) {
    var hsh = "#/" + (phase || "") + (section ? "/" + section : "");
    var keys = Object.keys(params || {});
    if (keys.length) hsh += "?" + keys.map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(params[k]); }).join("&");
    return hsh;
  }

  function go(phase, section) {
    location.hash = hashFor(phase, section);
  }

  function route() {
    var r = parseHash();
    var phase = findPhase(r.phase);
    var section = findSection(phase, r.section);
    state.route = { phase: phase ? phase.id : null, section: section ? section.id : null };
    if (section) {
      var vals = state.values[section.id] || {};
      section.controls.forEach(function (c) {
        var v = r.params[c.id];
        if (v != null && c.options.some(function (o) { return o.value === v; })) vals[c.id] = v;
        if (vals[c.id] == null) vals[c.id] = c["default"];
      });
      state.values[section.id] = vals;
    }
    renderNav();
    MAIN.textContent = "";
    hideTip();
    if (section) renderSection(phase, section);
    else if (phase) renderPhase(phase);
    else renderHome();
    MAIN.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }

  // ---------------------------------------------------------------- navigation

  var STATUS_BADGE = { active: "in progress", planned: "planned" };  // done: no nav badge

  function navLink(label, cls, current, onclick, badge) {
    return h("button", { type: "button", className: "nav-link " + cls, "aria-current": current ? "page" : null, onclick: onclick },
      h("span", { text: label }), badge ? h("span", { className: "nav-badge", text: badge }) : null);
  }

  function renderNav() {
    NAV.textContent = "";
    NAV.appendChild(h("a", { className: "nav-title", href: "#/", text: "Results explorer" }));
    NAV.appendChild(h("p", { className: "nav-sub", text: "Poker opponent modeling & exploitation" }));
    var tree = h("div", { className: "nav-tree" });
    var select = h("select", { className: "select nav-select", "aria-label": "Go to page", onchange: function (e) { location.hash = e.target.value; } });
    select.appendChild(h("option", { value: "#/", text: "Home" }));
    DATA.phases.forEach(function (p) {
      var wrap = h("div", { className: "nav-phase" });
      var planned = p.status === "planned";
      wrap.appendChild(navLink(p.label + ": " + p.title, "phase" + (planned ? " planned" : ""),
        state.route.phase === p.id && !state.route.section, function () { go(p.id); }, STATUS_BADGE[p.status]));
      select.appendChild(h("option", { value: hashFor(p.id), text: p.label + ": " + p.title + (STATUS_BADGE[p.status] ? " (" + STATUS_BADGE[p.status] + ")" : "") }));
      p.sections.forEach(function (sec) {
        wrap.appendChild(navLink(sec.title, "section", state.route.section === sec.id && state.route.phase === p.id, function () { go(p.id, sec.id); }));
        select.appendChild(h("option", { value: hashFor(p.id, sec.id), text: " " + sec.title }));
      });
      tree.appendChild(wrap);
    });
    select.value = hashFor(state.route.phase, state.route.section);
    if (!state.route.phase) select.value = "#/";
    NAV.appendChild(tree);
    NAV.appendChild(select);
    NAV.appendChild(h("div", { className: "nav-footer" }, themeButton()));
  }

  // ---------------------------------------------------------------- theme

  var THEMES = ["auto", "light", "dark"];
  function getTheme() { try { return localStorage.getItem("explorer-theme") || "auto"; } catch (e) { return "auto"; } }
  function applyTheme(t) {
    if (t === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
  }
  function themeButton() {
    var t = getTheme();
    return h("button", {
      type: "button", className: "theme-btn", title: "Switch light / dark / follow system",
      onclick: function () {
        var next = THEMES[(THEMES.indexOf(getTheme()) + 1) % THEMES.length];
        try { localStorage.setItem("explorer-theme", next); } catch (e) { /* storage unavailable */ }
        applyTheme(next);
        renderNav();
      }
    }, "Theme: " + t.charAt(0).toUpperCase() + t.slice(1));
  }

  // ---------------------------------------------------------------- pages

  function tiles(list) {
    if (!list || !list.length) return null;
    return h("div", { className: "tiles" }, list.map(function (t) {
      return h("div", { className: "tile" },
        h("div", { className: "tile-label", text: t.label }),
        h("div", { className: "tile-value", text: t.value }),
        t.note ? h("div", { className: "tile-note", text: t.note }) : null);
    }));
  }

  function renderHome() {
    document.title = "Poker Results Explorer";
    MAIN.appendChild(h("h1", { text: DATA.title }));
    MAIN.appendChild(h("p", { className: "lead", text: DATA.intro }));
    MAIN.appendChild(h("div", { className: "cards" }, DATA.phases.map(function (p) {
      var meta = p.status === "planned" ? "planned" : p.sections.length + " sections" + (p.status === "active" ? ", in progress" : "");
      return h("button", { type: "button", className: "card", onclick: function () { go(p.id); } },
        h("div", { className: "card-meta", text: p.label + " · " + meta }),
        h("div", { className: "card-title", text: p.title }),
        h("div", { className: "card-text", text: p.summary }));
    })));
  }

  function provenanceBlock(list) {
    if (!list || !list.length) return null;
    return h("div", { className: "sources" },
      h("div", { text: "Results files (seed, commit that produced them):" }),
      list.map(function (pv) {
        var bits = [];
        if (pv.seed != null && pv.seed !== "") bits.push("seed " + pv.seed);
        if (pv.commit) bits.push("commit " + pv.commit + (pv.dirty ? " (uncommitted changes)" : ""));
        if (pv.fingerprint) bits.push("simulation " + pv.fingerprint);
        return h("div", {}, h("code", { text: pv.file }), bits.length ? " · " + bits.join(" · ") : "");
      }));
  }

  function renderPhase(p) {
    document.title = p.label + " · Poker Results Explorer";
    MAIN.appendChild(h("p", { className: "crumb" }, h("button", { type: "button", onclick: function () { go(); }, text: "Home" }), " / " + p.label));
    MAIN.appendChild(h("h1", {}, p.title, h("span", { className: "badge", text: STATUS_BADGE[p.status] || "complete" })));
    MAIN.appendChild(h("p", { className: "lead", text: p.summary }));
    if (p.status === "planned") {
      MAIN.appendChild(h("div", { className: "empty", text: "No results yet. Sections appear here as this phase's experiments produce numbers; every quantified result gets its own page." }));
      return;
    }
    append(MAIN, tiles(p.tiles));
    MAIN.appendChild(h("h2", { text: "Sections" }));
    MAIN.appendChild(h("div", { className: "cards" }, p.sections.map(function (sec) {
      return h("button", { type: "button", className: "card", onclick: function () { go(p.id, sec.id); } },
        h("div", { className: "card-title", text: sec.title }),
        h("div", { className: "card-text", text: sec.lead }),
        sec.controls.length ? h("div", { className: "card-meta", text: "Explore by: " + sec.controls.map(function (c) { return c.label.toLowerCase(); }).join(", ") }) : null);
    })));
    append(MAIN, provenanceBlock(p.provenance));
  }

  var current = { figures: [] };

  function renderSection(p, sec) {
    document.title = sec.title + " · Poker Results Explorer";
    MAIN.appendChild(h("p", { className: "crumb" },
      h("button", { type: "button", onclick: function () { go(); }, text: "Home" }), " / ",
      h("button", { type: "button", onclick: function () { go(p.id); }, text: p.label })));
    MAIN.appendChild(h("h1", {}, sec.title, sec.postHoc ? h("span", { className: "badge", title: "Includes analyses added after seeing a first run", text: "includes post hoc analyses" }) : null));
    MAIN.appendChild(h("p", { className: "lead", text: sec.lead }));
    var vals = state.values[sec.id];
    var chartsWrap = h("div", { className: "charts" });
    var notes = h("div");

    function drawView() {
      var key = sec.controls.map(function (c) { return vals[c.id]; }).join("|");
      var v = sec.views[key];
      chartsWrap.textContent = "";
      notes.textContent = "";
      current.figures = [];
      if (!v) { chartsWrap.appendChild(h("div", { className: "empty", text: "No data for this combination." })); return; }
      v.charts.forEach(function (spec, i) {
        var fig = figure(spec, sec.id + "|" + i, sec.tables);
        if (v.charts.length === 1) fig.classList.add("wide");
        chartsWrap.appendChild(fig);
        current.figures.push(fig);
      });
      current.figures.forEach(function (f) { f._draw(); });
      (v.notes || []).forEach(function (n) { notes.appendChild(h("p", { className: "notes", text: n })); });
    }

    if (sec.controls.length) {
      var bar = h("div", { className: "controls", role: "group", "aria-label": "View options" });
      sec.controls.forEach(function (c) {
        var label = h("span", { className: "control-label", text: c.label });
        var widget;
        if (c.options.length > 7) {
          widget = h("select", { className: "select", "aria-label": c.label, onchange: function (e) { set(c.id, e.target.value); } });
          c.options.forEach(function (o) { widget.appendChild(h("option", { value: o.value, text: o.label })); });
          widget.value = vals[c.id];
        } else {
          widget = h("div", { className: "seg", role: "group", "aria-label": c.label });
          c.options.forEach(function (o) {
            widget.appendChild(h("button", { type: "button", "aria-pressed": String(vals[c.id] === o.value), "data-value": o.value, onclick: function () { set(c.id, o.value); } }, o.label));
          });
        }
        bar.appendChild(h("div", {}, label, widget));
      });
      MAIN.appendChild(bar);
    }

    function set(id, value) {
      vals[id] = value;
      MAIN.querySelectorAll(".controls .seg").forEach(function (seg, i) {
        var c = sec.controls.filter(function (cc) { return cc.options.length <= 7; })[i];
        if (!c) return;
        seg.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(vals[c.id] === b.getAttribute("data-value"))); });
      });
      history.replaceState(null, "", hashFor(p.id, sec.id, vals));
      drawView();
    }

    MAIN.appendChild(chartsWrap);
    MAIN.appendChild(notes);
    drawView();
    if (sec.takeaways.length) {
      MAIN.appendChild(h("h2", { text: "What it shows" }));
      MAIN.appendChild(h("ul", { className: "takeaways" }, sec.takeaways.map(function (t) { return h("li", { text: t }); })));
    }
    if (sec.method) MAIN.appendChild(h("details", { className: "method" }, h("summary", { text: "How it was computed" }), h("p", { text: sec.method })));
    var prov = p.provenance.filter(function (pv) { return sec.sources.indexOf(pv.file) >= 0; });
    append(MAIN, provenanceBlock(prov.length ? prov : sec.sources.map(function (f) { return { file: f, seed: "", commit: "" }; })));
  }

  // ---------------------------------------------------------------- start

  window.addEventListener("error", function (e) {
    var msg = h("div", { className: "empty", role: "alert", "data-error": "1", text: "Something went wrong drawing this page: " + (e.message || e) });
    MAIN.insertBefore(msg, MAIN.firstChild);
  });

  window.addEventListener("hashchange", route);
  window.addEventListener("scroll", hideTip, { passive: true });
  applyTheme(getTheme());
  route();
})();
