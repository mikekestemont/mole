"""Bokeh map + image viewer for the review sheet (dark, full-width, zoomable).

Why Bokeh: wheel-zoom, box-zoom, pan and reset are the tools people already know
from every plotting tool, and getting them right by hand in SVG is a lot of
fiddly code that would still behave subtly differently. The charter viewer uses
the *same* toolbar, so zooming into a hand on the map and zooming into the ink on
the page feel identical.

The cost is honest and worth stating: ``INLINE`` resources embed ~3.9 MB of
BokehJS in the file. That is the price of staying a SINGLE offline document — a
CDN link would be smaller but would break the moment the file is opened on a
train, which is exactly when a colleague reads it. :func:`bokehjs_bytes` reports
the cost so the image budget can subtract it and ``--max-mb`` keeps meaning what
it says.

Two figures, one document:

* **map** — one point per charter, coloured by the active scheme. Tap a point and
  the viewer loads that page; hovering shows filename and hand.
* **viewer** — the page as an ``image_url`` glyph in data space, so the same
  zoom/pan tools apply and the aspect ratio is preserved.

Both are driven from plain JS through ``window.MOLE`` (see :func:`glue_js`), so
the suggestion lists — ordinary HTML — can recolour and highlight the map without
Bokeh needing to know they exist.
"""

from __future__ import annotations


def available() -> bool:
    try:
        import bokeh  # noqa: F401
    except ImportError:
        return False
    return True


def bokehjs_bytes() -> int:
    """Size of the inlined BokehJS, so the image budget can account for it."""
    from bokeh.resources import INLINE

    return sum(len(s) for s in INLINE.js_raw) + sum(len(s) for s in INLINE.css_raw)


_HIGHLIGHT_STROKE = "#CC0000"
_NN_ACCENT = "#5b9bff"          # connector/ring colour for the nearest-neighbour link
_HULL_GREY = "#9aa0b0"          # convex-hull fill/line: neutral, reads on both themes

# Figure palette per theme, mirrored into the glue JS so a live toggle and the initial
# render agree. ``label`` is the class-id text colour (keyed to the BACKGROUND, not the
# point, so it stays legible in both themes); ``bg`` doubles as the label halo.
THEME = {
    "dark": dict(bg="#12131a", border="#2a2c39", grid="#20222e", pt_line="#0008",
                 label="#e8eaed"),
    "light": dict(bg="#ffffff", border="#c9ccd6", grid="#eceff3", pt_line="#00000030",
                  label="#14161a"),
}


def build(coords, names, hands, colors, *, highlight_idx=None, point_size: float = 9.0,
          show_labels: bool = False, label_cats=None, theme: str = "dark"):
    """Return ``(script, map_div, view_div, css, js)`` ready to drop into a page.

    ``highlight_idx`` marks target documents with a thick red stroke + slightly
    larger marker on the main scatter (plus a stem label) — the Sluis pattern.
    Highlights are NOT a second overlay circle: those used to sit on top and
    swallow taps. ``point_size`` sets the base marker size. ``show_labels``
    prints the active category id inside each circle (initial state; toggled
    live). ``label_cats`` are the first scheme's per-point categories.
    ``theme`` is ``dark`` (review) or ``light`` (publication).
    """
    from bokeh.embed import components
    from bokeh.models import (
        ColumnDataSource,
        HoverTool,
        LabelSet,
        LassoSelectTool,
        Range1d,
        TapTool,
    )
    from bokeh.plotting import figure
    from bokeh.resources import INLINE
    from bokeh.themes import built_in_themes

    from mole.viz.scatter import _label_text, _text_on

    theme = "light" if str(theme).lower() == "light" else "dark"
    pal = THEME[theme]
    n = coords.shape[0]
    label_cats = list(label_cats) if label_cats is not None else [""] * n
    hi_set = set(highlight_idx or [])

    xs = [float(v) for v in coords[:, 0]]
    ys = [float(v) for v in coords[:, 1]]
    # Highlight = red stroke + bump on the SAME scatter (one hit target). A
    # stacked hollow ring overlay used to intercept taps on targets.
    sizes, line_colors, line_widths, hl_flags = [], [], [], []
    for i in range(n):
        is_hi = i in hi_set
        hl_flags.append(1 if is_hi else 0)
        sizes.append(float(point_size) + (6.0 if is_hi else 0.0))
        line_colors.append(_HIGHLIGHT_STROKE if is_hi else pal["pt_line"])
        line_widths.append(3.0 if is_hi else 0.5)
    src = ColumnDataSource(dict(
        x=xs, y=ys, name=list(names),
        hand=[h or "not attributed" for h in hands],
        color=list(colors), alpha=[0.85] * n, size=sizes,
        line_color=line_colors, line_width=line_widths, hl=hl_flags,
        label=[_label_text(c) for c in label_cats],
        text_color=[_text_on(c) for c in colors]),
        name="scatter")

    p = figure(name="map", sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom", toolbar_location="above",
               x_axis_location=None, y_axis_location=None,
               background_fill_color=pal["bg"], border_fill_color=pal["bg"],
               outline_line_color=pal["border"])
    p.grid.grid_line_color = pal["grid"]
    # legend-tap convex hull (drawn UNDER the points, ABOVE the grid): the region a
    # category's charters occupy, filled a faint neutral grey. Populated from the page
    # via MOLE.showHull; p.patches wants list-of-lists so one polygon is [[...]].
    hull_src = ColumnDataSource(dict(xs=[], ys=[]), name="hull")
    p.patches("xs", "ys", source=hull_src, fill_color=_HULL_GREY, fill_alpha=0.14,
              line_color=_HULL_GREY, line_alpha=0.5, line_width=1, level="underlay")
    # nearest-neighbour link (drawn UNDER the points): connectors from the selected
    # charter to its top-k neighbours, populated from the page via MOLE.markNeighbors.
    nn_seg = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]), name="nn_seg")
    p.segment("x0", "y0", "x1", "y1", source=nn_seg, line_color=_NN_ACCENT,
              line_alpha=0.7, line_width=1.5, level="underlay")

    r = p.scatter("x", "y", source=src, size="size", color="color",
                  fill_alpha="alpha", line_alpha="alpha",
                  line_color="line_color", line_width="line_width",
                  # Bokeh's default non-selection alpha (0.2) is invisible on a dark
                  # background: selecting one charter must not erase the context it
                  # is being judged against. (Hidden points are removed by SIZE, not
                  # alpha — nonselection_fill_alpha would otherwise resurrect them.)
                  nonselection_fill_alpha=0.55, nonselection_line_alpha=0.25,
                  selection_line_color="#5a7fd6", selection_line_width=2.5)
    # Explicit tools with renderers pinned to the main scatter. Overlay glyphs
    # (NN rings, etc.) must never be in this list — they sit on top and would
    # swallow taps so highlighted / ringed charters never open in the viewer.
    tap = TapTool(renderers=[r])
    lasso = LassoSelectTool(renderers=[r])
    p.add_tools(HoverTool(renderers=[r], tooltips=[("charter", "@name"),
                                                   ("hand", "@hand")]),
                tap, lasso)

    # class-id labels, centred in each circle; toggled live from the page. Text colour
    # keys to the theme background (not the point) with a halo so it reads either way.
    labels = LabelSet(x="x", y="y", text="label", source=src, text_color=pal["label"],
                      text_align="center", text_baseline="middle",
                      background_fill_color=pal["bg"], background_fill_alpha=0.45,
                      text_font_size=f"{max(7.0, point_size * 0.9):.0f}pt",
                      text_font_style="bold", name="class_labels")
    labels.visible = bool(show_labels)
    p.add_layout(labels)

    # nearest-neighbour ring markers (over the points). Underlay would hide them
    # under the dots; keep overlay but leave TapTool.renderers=[r] so they don't
    # steal clicks (re-assert after this glyph in case Bokeh expands the list).
    nn_pts = ColumnDataSource(dict(x=[], y=[]), name="nn_pts")
    p.scatter("x", "y", source=nn_pts, size=float(point_size) + 8, marker="circle",
              fill_alpha=0.0, line_color=_NN_ACCENT, line_width=2.0, level="overlay")

    # Target names only (no second hit-tested circle). Highlight itself is the
    # thicker red stroke + slightly larger size on the MAIN scatter above — a
    # stacked hollow ring used to sit on top and eat taps even with TapTool
    # pinned to ``r``.
    hi = list(highlight_idx or [])
    if hi:
        hl_src = ColumnDataSource(dict(
            x=[xs[i] for i in hi], y=[ys[i] for i in hi],
            label=[f"  {names[i]}" for i in hi]), name="highlights")
        p.add_layout(LabelSet(x="x", y="y", text="label", source=hl_src,
                              text_color=_HIGHLIGHT_STROKE, text_font_size="11pt",
                              text_font_style="bold"))

    tap.renderers = [r]
    lasso.renderers = [r]
    # the viewer: an image in data space, so zoom/pan behave like the map
    # every column must start empty: a url=[] beside x=[0] trips a BokehUserWarning
    img = ColumnDataSource(dict(url=[], x=[], y=[], w=[], h=[]), name="page")
    # match_aspect governs AUTO-ranging only; these ranges are set explicitly from
    # the frame's pixel size in glue_js (`fit`), which is the only way to letterbox
    # a page of arbitrary shape without distorting it.
    v = figure(name="viewer", sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom", toolbar_location="above",
               x_axis_location=None, y_axis_location=None,
               background_fill_color=pal["bg"], border_fill_color=pal["bg"],
               outline_line_color=pal["border"])
    v.grid.grid_line_color = None
    v.x_range = Range1d(0, 1)
    v.y_range = Range1d(0, 1)
    v.image_url(url="url", x="x", y="y", w="w", h="h", source=img,
                anchor="top_left")

    bokeh_theme = built_in_themes["caliber" if theme == "light" else "dark_minimal"]
    script, divs = components({"map": p, "view": v}, theme=bokeh_theme)
    css = "\n".join(INLINE.css_raw)
    js = "\n".join(INLINE.js_raw)
    return script, divs["map"], divs["view"], css, js


def glue_js() -> str:
    """`window.MOLE`: recolour, highlight and load a page, from ordinary JS.

    Bokeh models are looked up by name once the document exists — the review
    panel is plain HTML and must not have to know anything about Bokeh beyond
    these three calls.
    """
    theme_js = __import__("json").dumps(THEME)
    return r"""
window.MOLE = (function(){
  var THEME = __THEME__;
  var scatter=null, page=null, viewer=null, mapf=null, classLabels=null;
  var nnSeg=null, nnPts=null, hull=null;
  var ready=false, queue=[], baseSize=9;
  function grab(){
    if(!window.Bokeh || !Bokeh.documents || !Bokeh.documents.length) return false;
    var doc = Bokeh.documents[0];
    scatter     = doc.get_model_by_name('scatter');
    page        = doc.get_model_by_name('page');
    viewer      = doc.get_model_by_name('viewer');
    mapf        = doc.get_model_by_name('map');
    classLabels = doc.get_model_by_name('class_labels');
    nnSeg       = doc.get_model_by_name('nn_seg');
    nnPts       = doc.get_model_by_name('nn_pts');
    hull        = doc.get_model_by_name('hull');
    if(!scatter) return false;
    if(scatter.data.size && scatter.data.size.length){
      var hl0 = (scatter.data.hl && scatter.data.hl[0]) ? 6 : 0;
      baseSize = scatter.data.size[0] - hl0;
    }
    ready = true;
    watchFrame();
    while(queue.length) queue.shift()();
    return true;
  }
  (function wait(n){ if(grab()||n>200) return; setTimeout(function(){wait(n+1)}, 50); })(0);
  // the pane is resizable (the divider) — re-letterbox whenever the frame changes
  function watchFrame(){
    if(!viewer) return;
    ['inner_width','inner_height'].forEach(function(prop){
      viewer.properties[prop].change.connect(function(){ fit(); });
    });
  }
  function later(fn){ ready ? fn() : queue.push(fn); }
  // highlighted points are drawn slightly larger on the same scatter (no overlay ring)
  function sized(i, d, px){ return px + ((d.hl && d.hl[i]) ? 6 : 0); }

  function textOn(hex){
    var h = String(hex||'').replace('#','');
    if(h.length !== 6) return '#111';
    var r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
    return (0.299*r + 0.587*g + 0.114*b) > 140 ? '#111' : '#fff';
  }
  function labelText(cat){
    var s = String(cat==null?'':cat).trim().toLowerCase();
    var bad = {unlabeled:1,'-1':1,'—':1,'-':1,none:1,nan:1,unknown:1,na:1,'n/a':1,'?':1,'':1};
    if(bad[s]) return '';
    var c = String(cat);
    return c.length > 8 ? c.slice(0,7)+'…' : c;
  }

  function setColors(cols){
    later(function(){
      // replace the whole data object: mutating a column in place does not
      // always trigger a redraw in BokehJS
      var d = Object.assign({}, scatter.data);
      d.color = cols.slice();
      d.text_color = cols.map(textOn);
      scatter.data = d;
    });
  }
  function setLabels(cats){
    later(function(){
      var d = Object.assign({}, scatter.data);
      d.label = cats.map(labelText);
      scatter.data = d;
    });
  }
  function showLabels(on){
    later(function(){ if(classLabels) classLabels.visible = !!on; });
  }
  function setSize(px){
    later(function(){
      baseSize = px;
      var d = Object.assign({}, scatter.data);
      var s = new Array(d.x.length);
      // alpha 0 == "hidden by a toggle": keep it gone by giving it zero size, so a
      // later selection (nonselection_fill_alpha) can't bring it back on screen.
      for(var i=0;i<s.length;i++) s[i] = (d.alpha[i] === 0) ? 0 : sized(i, d, px);
      d.size = s;
      scatter.data = d;
      if(classLabels) classLabels.text_font_size = Math.max(7, px*0.9).toFixed(0)+'pt';
    });
  }
  function setTheme(dark){
    later(function(){
      var pal = dark ? THEME.dark : THEME.light;
      [mapf, viewer].forEach(function(f){
        if(!f) return;
        f.background_fill_color = pal.bg;
        f.border_fill_color = pal.bg;
        f.outline_line_color = pal.border;
      });
      var grids = (mapf && mapf.grid) || [];
      for(var g=0; g<grids.length; g++) grids[g].grid_line_color = pal.grid;
      if(classLabels){                      // keep class ids legible against the new bg
        classLabels.text_color = pal.label;
        classLabels.background_fill_color = pal.bg;
      }
    });
  }
  // draw connectors from a selected charter to its nearest neighbours
  function markNeighbors(center, idxs){
    later(function(){
      if(!nnSeg || !nnPts) return;
      var X = scatter.data.x, Y = scatter.data.y;
      var x0=[], y0=[], x1=[], y1=[], px=[], py=[];
      if(center != null && idxs && X){
        var cx = X[center], cy = Y[center];
        idxs.forEach(function(j){
          if(X[j] == null) return;
          x0.push(cx); y0.push(cy); x1.push(X[j]); y1.push(Y[j]);
          px.push(X[j]); py.push(Y[j]);
        });
      }
      nnSeg.data = {x0:x0, y0:y0, x1:x1, y1:y1};
      nnPts.data = {x:px, y:py};
    });
  }
  function clearNeighbors(){ markNeighbors(null, null); }
  // convex hull (Andrew's monotone chain) of the points at the given indices, drawn
  // as a single faint-grey patch under the dots to show a category's territory.
  function convexHull(pts){
    if(pts.length < 3) return pts.slice();          // degenerate: use points as-is
    var p = pts.slice().sort(function(a,b){ return a[0]-b[0] || a[1]-b[1]; });
    function cross(o,a,b){ return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]); }
    var lower=[];
    for(var i=0;i<p.length;i++){
      while(lower.length>=2 && cross(lower[lower.length-2],lower[lower.length-1],p[i])<=0) lower.pop();
      lower.push(p[i]);
    }
    var upper=[];
    for(var j=p.length-1;j>=0;j--){
      while(upper.length>=2 && cross(upper[upper.length-2],upper[upper.length-1],p[j])<=0) upper.pop();
      upper.push(p[j]);
    }
    lower.pop(); upper.pop();
    var h = lower.concat(upper);
    return h.length >= 3 ? h : pts.slice();
  }
  function showHull(indices){
    later(function(){
      if(!hull) return;
      var X = scatter.data.x, Y = scatter.data.y, pts = [];
      (indices||[]).forEach(function(i){
        if(X[i] != null && Y[i] != null) pts.push([X[i], Y[i]]);
      });
      if(!pts.length){ hull.data = {xs:[], ys:[]}; return; }
      var h = convexHull(pts);
      hull.data = {xs:[h.map(function(q){return q[0];})],
                   ys:[h.map(function(q){return q[1];})]};
    });
  }
  function clearHull(){ later(function(){ if(hull) hull.data = {xs:[], ys:[]}; }); }
  function dataExtent(){
    var X = scatter.data.x, Y = scatter.data.y;
    var xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    for(var i=0;i<X.length;i++){
      if(X[i]==null||Y[i]==null) continue;
      if(X[i]<xmin) xmin=X[i]; if(X[i]>xmax) xmax=X[i];
      if(Y[i]<ymin) ymin=Y[i]; if(Y[i]>ymax) ymax=Y[i];
    }
    return {xmin:xmin,xmax:xmax,ymin:ymin,ymax:ymax};
  }
  // recenter/zoom the map on one point. ``span`` is the fraction of the data extent
  // the window should cover (defaults to a gentle 12% zoom-in).
  function centerOn(idx, span){
    later(function(){
      if(!mapf || !mapf.x_range || !mapf.y_range) return;
      var X = scatter.data.x, Y = scatter.data.y;
      if(X[idx]==null || Y[idx]==null) return;
      var e = dataExtent();
      var fx = (span || 0.12), fy = (span || 0.12);
      var wx = (e.xmax - e.xmin) * fx, wy = (e.ymax - e.ymin) * fy;
      if(!(wx > 0)) wx = 1; if(!(wy > 0)) wy = 1;
      mapf.x_range.start = X[idx] - wx/2; mapf.x_range.end = X[idx] + wx/2;
      mapf.y_range.start = Y[idx] - wy/2; mapf.y_range.end = Y[idx] + wy/2;
    });
  }
  var flashTimer = null;
  // briefly enlarge one point so a searched charter is easy to spot, then restore.
  function flash(idx){
    later(function(){
      var d = Object.assign({}, scatter.data);
      var s = d.size.slice();
      if(s[idx] == null) return;
      var prev = s[idx];
      s[idx] = Math.max(prev, baseSize) * 3.2 + 6;
      d.size = s;
      scatter.data = d;
      if(flashTimer) clearTimeout(flashTimer);
      flashTimer = setTimeout(function(){
        var d2 = Object.assign({}, scatter.data);
        var s2 = d2.size.slice();
        // respect the "hidden by toggle" convention (alpha 0 == size 0)
        s2[idx] = (d2.alpha[idx] === 0) ? 0 : sized(idx, d2, baseSize);
        d2.size = s2;
        scatter.data = d2;
        flashTimer = null;
      }, 1200);
    });
  }
  var pulseRAF = null, pulseSet = null;
  function restorePulse(set){
    var d = Object.assign({}, scatter.data), s = d.size.slice();
    (set || pulseSet || []).forEach(function(i){
      s[i] = (d.alpha[i] === 0) ? 0 : sized(i, d, baseSize);
    });
    d.size = s; scatter.data = d;
  }
  // pop a whole GROUP of points (e.g. every charter of a clicked hand): one quick
  // grow-and-settle so the category jumps off the page on click. Honours the hidden
  // convention (alpha 0 stays size 0) and settles back to baseSize when done.
  function pulse(idxs){
    later(function(){
      if(pulseRAF){ cancelAnimationFrame(pulseRAF); pulseRAF = null; restorePulse(); }
      var set = (idxs || []).filter(function(i){ return scatter.data.alpha[i] !== 0; });
      if(!set.length) return;
      pulseSet = set;
      var dur = 640, peak = 2.7, t0 = performance.now();
      function frame(now){
        var t = Math.min(1, (now - t0) / dur);
        var mult = 1 + (peak - 1) * Math.sin(Math.PI * t);   // 0→peak→0, smooth
        var d = Object.assign({}, scatter.data), s = d.size.slice();
        for(var k=0;k<set.length;k++){
          var i=set[k];
          s[i] = (d.alpha[i]===0) ? 0 : sized(i, d, baseSize)*mult;
        }
        d.size = s; scatter.data = d;
        if(t < 1){ pulseRAF = requestAnimationFrame(frame); }
        else { pulseRAF = null; pulseSet = null; restorePulse(set); }
      }
      pulseRAF = requestAnimationFrame(frame);
    });
  }
  function setAlphas(alphas, sizes){
    later(function(){
      var d = Object.assign({}, scatter.data);
      d.alpha = alphas.slice();
      // fully remove alpha-0 points by shrinking them to nothing (see setSize)
      var s = sizes ? sizes.slice() : d.size.slice();
      for(var i=0;i<alphas.length;i++){
        if(alphas[i] === 0) s[i] = 0;
        else if(!sizes) s[i] = sized(i, d, baseSize);
      }
      d.size = s;
      scatter.data = d;
    });
  }
  function select(i){
    later(function(){ scatter.selected.indices = [i]; });
  }
  var shownAspect = 1.0;
  function fit(){
    // The page occupies x 0..1 and y 0..ar in data space. To show it undistorted
    // the DATA aspect must equal the FRAME's pixel aspect, so letterbox along
    // whichever axis has room instead of stretching the image to the ranges.
    if(!viewer) return;
    var W = viewer.inner_width || 0, H = viewer.inner_height || 0;
    if(!W || !H) return;
    var P = H / W, ar = shownAspect;
    if(ar > P){                       // page relatively taller: fit its height
      var half = (ar / P) / 2;
      viewer.x_range.start = 0.5 - half; viewer.x_range.end = 0.5 + half;
      viewer.y_range.start = 0;          viewer.y_range.end = ar;
    } else {                          // fit its width, centre it vertically
      viewer.x_range.start = 0;          viewer.x_range.end = 1;
      viewer.y_range.start = ar / 2 - P / 2;
      viewer.y_range.end   = ar / 2 + P / 2;
    }
  }
  function showImage(uri, w, h){
    later(function(){
      if(!page) return;
      shownAspect = (h && w) ? h / w : 1.0;
      page.data = {url:[uri], x:[0], y:[shownAspect], w:[1], h:[shownAspect]};
      fit();
    });
  }
  function onTap(cb){
    later(function(){
      // BokehJS signals live on the property, not the model
      scatter.selected.properties.indices.change.connect(function(){
        var idx = scatter.selected.indices;
        if(idx && idx.length) cb(idx[idx.length - 1]);
      });
    });
  }
  return {setColors:setColors, setAlphas:setAlphas, showImage:showImage,
          onTap:onTap, select:select, setLabels:setLabels, showLabels:showLabels,
          setSize:setSize, setTheme:setTheme, markNeighbors:markNeighbors,
          clearNeighbors:clearNeighbors, showHull:showHull, clearHull:clearHull,
          centerOn:centerOn, flash:flash, pulse:pulse};
})();
""".replace("__THEME__", theme_js)
