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


def build(coords, names, hands, colors):
    """Return ``(script, map_div, view_div, css, js)`` ready to drop into a page."""
    from bokeh.embed import components
    from bokeh.models import ColumnDataSource, HoverTool, Range1d, TapTool
    from bokeh.plotting import figure
    from bokeh.resources import INLINE
    from bokeh.themes import built_in_themes

    xs = [float(v) for v in coords[:, 0]]
    ys = [float(v) for v in coords[:, 1]]
    src = ColumnDataSource(dict(
        x=xs, y=ys, name=list(names),
        hand=[h or "not attributed" for h in hands],
        color=list(colors), alpha=[0.85] * len(xs), size=[9.0] * len(xs)),
        name="scatter")

    p = figure(name="map", sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,lasso_select,tap,reset,save",
               active_scroll="wheel_zoom", toolbar_location="above",
               x_axis_location=None, y_axis_location=None,
               background_fill_color="#12131a", border_fill_color="#12131a",
               outline_line_color="#2a2c39")
    p.grid.grid_line_color = "#20222e"
    r = p.scatter("x", "y", source=src, size="size", color="color",
                  fill_alpha="alpha", line_color="#0008", line_width=0.5)
    p.add_tools(HoverTool(renderers=[r], tooltips=[("charter", "@name"),
                                                   ("hand", "@hand")]))
    p.select(TapTool)                      # tap selection drives the viewer

    # the viewer: an image in data space, so zoom/pan behave like the map
    # every column must start empty: a url=[] beside x=[0] trips a BokehUserWarning
    img = ColumnDataSource(dict(url=[], x=[], y=[], w=[], h=[]), name="page")
    v = figure(name="viewer", sizing_mode="stretch_both",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom", toolbar_location="above",
               x_axis_location=None, y_axis_location=None, match_aspect=True,
               background_fill_color="#12131a", border_fill_color="#12131a",
               outline_line_color="#2a2c39")
    v.grid.grid_line_color = None
    v.x_range = Range1d(0, 1)
    v.y_range = Range1d(0, 1)
    v.image_url(url="url", x="x", y="y", w="w", h="h", source=img,
                anchor="top_left")

    script, divs = components({"map": p, "view": v},
                              theme=built_in_themes["dark_minimal"])
    css = "\n".join(INLINE.css_raw)
    js = "\n".join(INLINE.js_raw)
    return script, divs["map"], divs["view"], css, js


def glue_js() -> str:
    """`window.MOLE`: recolour, highlight and load a page, from ordinary JS.

    Bokeh models are looked up by name once the document exists — the review
    panel is plain HTML and must not have to know anything about Bokeh beyond
    these three calls.
    """
    return r"""
window.MOLE = (function(){
  var scatter=null, page=null, viewer=null, ready=false, queue=[];
  function grab(){
    if(!window.Bokeh || !Bokeh.documents || !Bokeh.documents.length) return false;
    var doc = Bokeh.documents[0];
    scatter = doc.get_model_by_name('scatter');
    page    = doc.get_model_by_name('page');
    viewer  = doc.get_model_by_name('viewer');
    if(!scatter) return false;
    ready = true;
    while(queue.length) queue.shift()();
    return true;
  }
  (function wait(n){ if(grab()||n>200) return; setTimeout(function(){wait(n+1)}, 50); })(0);
  function later(fn){ ready ? fn() : queue.push(fn); }

  function setColors(cols){
    later(function(){
      // replace the whole data object: mutating a column in place does not
      // always trigger a redraw in BokehJS
      var d = Object.assign({}, scatter.data);
      d.color = cols.slice();
      scatter.data = d;
    });
  }
  function setAlphas(alphas, sizes){
    later(function(){
      var d = Object.assign({}, scatter.data);
      d.alpha = alphas.slice();
      if(sizes) d.size = sizes.slice();
      scatter.data = d;
    });
  }
  function select(i){
    later(function(){ scatter.selected.indices = [i]; });
  }
  function showImage(uri, w, h){
    later(function(){
      if(!page) return;
      var ar = (h && w) ? h / w : 1.0;
      page.data = {url:[uri], x:[0], y:[ar], w:[1], h:[ar]};
      page.change.emit();
      if(viewer){
        viewer.x_range.start = 0; viewer.x_range.end = 1;
        viewer.y_range.start = 0; viewer.y_range.end = ar;
      }
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
          onTap:onTap, select:select};
})();
"""
