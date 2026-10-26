"""Design-system layer: global styles and small HTML building blocks.

Streamlit's theme (.streamlit/config.toml) sets colours and fonts. This module
adds what the theme cannot express (panel surfaces, metric strips, editorial
headers) as CSS plus escaped HTML snippets. It formats values it is given and
never computes statistics. Every user-provided string (column and file names)
goes through ``esc`` before it is placed in markup.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import streamlit as st

Tone = Literal["ok", "review", "na"]

CSS = """
<style>
:root{--dl-ground:#FAF9F6;--dl-sunken:#F2F1EC;--dl-surface:#FFFFFF;--dl-line:#ECEAE3;--dl-border:#E3E1D9;
--dl-strong:#CFCCC1;--dl-ink:#0F1D33;--dl-ink2:#3A4659;--dl-muted:#5B6474;--dl-emerald:#0B7A57;
--dl-emerald-dark:#0A6447;--dl-tint:#E6F2EC;--dl-amber:#8A5300;--dl-amber-bg:#FBF2DF;--dl-amber-line:#EBD3A1;
--dl-mono:'JetBrains Mono',ui-monospace,monospace;--dl-serif:Newsreader,Georgia,serif}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:2.25rem;padding-bottom:4rem}
[data-testid="stSidebarContent"]{padding-top:.5rem}
[data-testid="stMetricValue"]{font-family:var(--dl-mono);letter-spacing:-.02em}
[data-testid="stExpander"] details{background:var(--dl-surface)}
[class*="st-key-panel"]{background:var(--dl-surface)}
[class*="st-key-upload-zone"]{background:var(--dl-surface);border:1.5px dashed #8FBFA9 !important;border-radius:8px}
[class*="st-key-sample-zone"]{background:var(--dl-sunken)}
[class*="st-key-stale-results"]{opacity:.45}
[class*="st-key-nav-"] button{justify-content:flex-start;text-align:left;border-color:transparent;background:transparent;
  min-height:2.75rem;padding-left:.75rem}
[class*="st-key-nav-"] button>div,[class*="st-key-nav-"] button span[data-has-shortcut]{justify-content:flex-start !important;width:100%}
[class*="st-key-nav-"] [data-testid="stTooltipHoverTarget"]{justify-content:stretch !important}
[class*="st-key-nav-"] button:hover{background:rgba(255,255,255,.6);border-color:var(--dl-border)}
[class*="st-key-nav-current-"] button{background:var(--dl-surface);border-color:var(--dl-border);font-weight:600;color:var(--dl-ink)}
[class*="st-key-nav-current-"] [data-testid="stIconMaterial"],
[class*="st-key-nav-complete-"] [data-testid="stIconMaterial"]{color:var(--dl-emerald)}
[class*="st-key-nav-locked-"] button{color:var(--dl-muted)}
.dl-eyebrow{font-family:var(--dl-mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dl-muted)}
.dl-mono{font-family:var(--dl-mono);font-variant-numeric:tabular-nums}
.dl-brand{display:flex;align-items:center;gap:10px;padding:4px 4px 8px}
.dl-brand b{font-family:var(--dl-serif);font-size:23px;font-weight:500;letter-spacing:-.01em;display:block;line-height:1.1}
.dl-brand span{font-family:var(--dl-mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dl-muted)}
.dl-card{background:var(--dl-surface);border:1px solid var(--dl-border);border-radius:6px;padding:12px 14px}
.dl-card-dashed{border:1px dashed var(--dl-strong);border-radius:6px;padding:12px 14px}
.dl-chip{display:inline-block;font-family:var(--dl-mono);font-size:11px;border-radius:999px;padding:2px 8px;
  background:var(--dl-sunken);color:var(--dl-ink2)}
.dl-chip-amber{background:var(--dl-amber-bg);color:var(--dl-amber);border:1px solid var(--dl-amber-line)}
.dl-crumb{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:13px;color:var(--dl-muted);
  border-bottom:1px solid var(--dl-border);padding-bottom:12px;margin-bottom:8px}
.dl-crumb b{color:var(--dl-ink);font-weight:500}
.dl-head{display:flex;flex-direction:column;gap:8px;max-width:720px}
.dl-head h1{font-family:var(--dl-serif);font-weight:400;font-size:46px;line-height:1.06;letter-spacing:-.015em;margin:0;padding:0;color:var(--dl-ink)}
.dl-head p{margin:0;font-size:15.5px;line-height:1.6;color:var(--dl-ink2)}
.dl-strip{display:flex;flex-wrap:wrap;background:var(--dl-surface);border:1px solid var(--dl-border);border-radius:6px}
.dl-strip>div{flex:1 1 170px;padding:18px 20px;display:flex;flex-direction:column;gap:6px;border-right:1px solid var(--dl-line)}
.dl-strip>div:last-child{border-right:0}
.dl-strip .v{font-family:var(--dl-mono);font-size:30px;font-weight:500;letter-spacing:-.02em;line-height:1.1}
.dl-strip .v.big{font-size:38px;letter-spacing:-.03em}
.dl-strip .c{font-size:12.5px;color:var(--dl-ink2)}
.dl-bar{display:block;height:4px;background:var(--dl-line);border-radius:2px;position:relative}
.dl-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:var(--dl-emerald)}
.dl-sub{background:var(--dl-ground);border:1px solid var(--dl-border);border-top:0;border-radius:0 0 6px 6px;
  padding:10px 20px;display:flex;flex-wrap:wrap;gap:6px 26px;font-size:13px;color:var(--dl-ink2);margin-top:-6px}
.dl-sub .dl-mono{color:var(--dl-ink)}
.dl-status{font-size:12.5px;font-weight:500;display:inline-flex;align-items:center;gap:6px}
.dl-status.ok{color:var(--dl-emerald-dark)}.dl-status.review{color:var(--dl-amber)}.dl-status.na{color:var(--dl-muted)}
.dl-strip>div.review{background:#FDF8EE}
.dl-eq{font-family:var(--dl-serif);font-size:26px;line-height:1.45;margin:4px 0 0;word-break:break-word}
.dl-eq .dl-mono{font-size:.8em}.dl-eq .pos{color:var(--dl-emerald-dark)}
.dl-note{font-size:14.5px;line-height:1.6;color:var(--dl-ink2);border-top:1px solid var(--dl-line);padding-top:12px;margin-top:12px}
.dl-hero h1{overflow-wrap:break-word;font-family:var(--dl-serif);font-weight:400;font-size:56px;line-height:1.04;letter-spacing:-.02em;margin:10px 0 14px;padding:0;max-width:880px}
.dl-hero p{font-size:17px;line-height:1.6;color:var(--dl-ink2);max-width:680px;margin:0}
.dl-steps{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:24px}
.dl-steps li{flex:1 1 210px;border-top:2px solid var(--dl-strong);padding-top:14px;display:flex;flex-direction:column;gap:6px}
.dl-steps li:first-child{border-top-color:var(--dl-ink)}
.dl-steps .n{font-family:var(--dl-mono);font-size:12px;color:var(--dl-emerald);font-weight:500}
.dl-steps b{font-size:15.5px}.dl-steps span{font-size:13.5px;line-height:1.6;color:var(--dl-ink2)}
.dl-caps{display:flex;flex-wrap:wrap;gap:20px}
.dl-caps article{flex:1 1 280px;background:var(--dl-surface);border:1px solid var(--dl-border);border-radius:6px;overflow:hidden}
.dl-caps .fig{background:var(--dl-ground);border-bottom:1px solid var(--dl-line);padding:18px 18px 10px}
.dl-caps .body{padding:18px 20px 22px;display:flex;flex-direction:column;gap:8px}
.dl-caps h3{font-family:var(--dl-serif);font-weight:400;font-size:21px;line-height:1.2;margin:0;padding:0}
.dl-caps ul{margin:4px 0 0;padding-left:18px;font-size:13.5px;line-height:1.7;color:var(--dl-ink2)}
.dl-aside{background:var(--dl-sunken);border:1px solid var(--dl-border);border-radius:6px;padding:20px 22px}
.dl-aside h3{font-family:var(--dl-serif);font-weight:400;font-size:22px;margin:6px 0 10px;padding:0}
.dl-aside ul{margin:0;padding-left:18px;font-size:13.5px;line-height:1.6;color:var(--dl-ink2)}
.dl-aside li{margin-bottom:6px}
.dl-empty{border:1px solid var(--dl-border);background:var(--dl-surface);border-radius:6px;padding:40px 32px}
.dl-empty h3{font-family:var(--dl-serif);font-weight:400;font-size:24px;margin:10px 0 6px;padding:0}
.dl-empty p{margin:0;font-size:14px;color:var(--dl-ink2)}
.dl-paper{background:var(--dl-surface);border:1px solid var(--dl-border);border-radius:4px;padding:44px 52px}
.dl-paper header{border-bottom:1px solid var(--dl-ink);padding-bottom:22px;margin-bottom:26px}
.dl-paper h2{font-family:var(--dl-serif);font-weight:400;font-size:36px;line-height:1.1;margin:8px 0 10px;padding:0}
.dl-paper h3{font-family:var(--dl-serif);font-weight:400;font-size:23px;margin:26px 0 12px;padding:0}
.dl-paper .lede{font-family:var(--dl-serif);font-style:italic;font-size:18px;color:var(--dl-ink2);margin:0}
.dl-paper dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px 24px;margin:16px 0 0;font-size:12.5px}
.dl-paper dt{color:var(--dl-muted)}.dl-paper dd{margin:0;font-family:var(--dl-mono)}
.dl-paper ol.find{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:14px}
.dl-paper ol.find li{display:flex;gap:14px;font-size:14.5px;line-height:1.6;color:var(--dl-ink2)}
.dl-paper ol.find .n{font-family:var(--dl-mono);font-size:13px;color:var(--dl-emerald-dark);padding-top:2px}
.dl-paper ul{padding-left:18px;font-size:14.5px;line-height:1.6;color:var(--dl-ink2)}
.dl-paper li{margin-bottom:6px}
.dl-paper table{width:100%;border-collapse:collapse;font-size:13.5px}
.dl-paper th{text-align:left;font-weight:500;font-size:12px;color:var(--dl-muted);padding:8px 10px;border-bottom:1px solid var(--dl-ink)}
.dl-paper td{padding:9px 10px;border-bottom:1px solid var(--dl-line)}
.dl-paper tr:last-child td{border-bottom:1px solid var(--dl-ink)}
.dl-paper .num{text-align:right;font-family:var(--dl-mono);font-size:13px;white-space:nowrap}
.dl-paper footer{border-top:1px solid var(--dl-line);margin-top:26px;padding-top:12px;font-size:12px;color:var(--dl-muted)}
@media (max-width:820px){
  .dl-head h1{font-size:34px}.dl-hero h1{font-size:38px}.dl-paper{padding:26px 20px}.dl-eq{font-size:20px}
  [data-testid="stMainBlockContainer"]{padding-left:1rem;padding-right:1rem}
}
@media (max-width:480px){.dl-hero h1{font-size:30px}.dl-head h1{font-size:30px}}
</style>
"""

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_rule(cls: str, svg: str, width: str, height: str) -> str:
    """CSS rule that paints ``svg`` as the background of ``.cls``.

    st.html sanitises away inline <svg> and data-URI <img> elements but keeps
    <style> blocks, so icons and small illustrations are drawn from CSS.
    """
    if "xmlns=" not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    data = base64.b64encode(" ".join(svg.split()).encode("utf-8")).decode("ascii")
    return (f".{cls}{{display:inline-block;flex:none;vertical-align:middle;width:{width};height:{height};"
            f'background:url("data:image/svg+xml;base64,{data}") center/contain no-repeat}}')


def _icon_svg(paths: str, color: str) -> str:
    return (f'<svg viewBox="0 0 16 16" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


ICON_CSS = "".join([
    svg_rule("dl-i-ok", _icon_svg('<circle cx="8" cy="8" r="6.2"/><path d="M5.5 8.2l1.8 1.8 3.3-3.6"/>', "#0A6447"),
             "14px", "14px"),
    svg_rule("dl-i-review", _icon_svg('<path d="M8 2.5 14 13H2z"/><path d="M8 6.5v3M8 11.3h.01"/>', "#8A5300"),
             "14px", "14px"),
    svg_rule("dl-i-na", _icon_svg('<circle cx="8" cy="8" r="6"/><path d="M5 8h6"/>', "#5B6474"), "14px", "14px"),
    svg_rule("dl-i-upload", _icon_svg('<path d="M8 11V3M4.5 6.5 8 3l3.5 3.5M3 13h10"/>', "#0B7A57"), "22px", "22px"),
    svg_rule(
        "dl-logo",
        '<svg viewBox="0 0 28 28"><circle cx="12" cy="12" r="9" fill="none" stroke="#0F1D33" stroke-width="1.8"/>'
        '<path d="M18.6 18.6 L25 25" stroke="#0F1D33" stroke-width="2.4" stroke-linecap="round"/>'
        '<path d="M6.5 15.5 L17.5 8.5" stroke="#0B7A57" stroke-width="1.8" stroke-linecap="round"/>'
        '<circle cx="8.5" cy="11.5" r="1.3" fill="#0F1D33"/><circle cx="12.5" cy="13" r="1.3" fill="#0F1D33"/>'
        '<circle cx="15" cy="8.6" r="1.3" fill="#0F1D33"/></svg>',
        "30px", "30px",
    ),
])


def icon(name: str) -> str:
    return f'<span class="dl-i-{name}" aria-hidden="true"></span>'


ICON_OK, ICON_REVIEW, ICON_NA, ICON_UPLOAD = icon("ok"), icon("review"), icon("na"), icon("upload")
STATUS = {"ok": (ICON_OK, "Tidak ada temuan"), "review": (ICON_REVIEW, "Perlu ditinjau"), "na": (ICON_NA, "Tidak tersedia")}
LOGO = '<span class="dl-logo" aria-hidden="true"></span>'


def inject_css() -> None:
    st.html(CSS.replace("</style>", ICON_CSS + "</style>"))


def breadcrumb(parts: Sequence[str], right: str = "") -> None:
    trail = ' <span aria-hidden="true">/</span> '.join(
        f"<b>{esc(p)}</b>" if i == len(parts) - 1 else esc(p) for i, p in enumerate(parts)
    )
    st.html(f'<nav class="dl-crumb" aria-label="Breadcrumb"><span>{trail}</span><span>{esc(right)}</span></nav>')


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.html(
        f'<div class="dl-head"><span class="dl-eyebrow">{esc(eyebrow)}</span>'
        f"<h1>{esc(title)}</h1><p>{esc(description)}</p></div>"
    )


@dataclass(frozen=True)
class Stat:
    label: str
    value: str
    caption: str = ""
    bar: float | None = None      # 0–1, drawn as a thin bar under the value
    tone: Tone | None = None      # adds a status line instead of a caption colour
    value_color: str | None = None


def stat_strip(items: Sequence[Stat], big: bool = False, footer: Sequence[str] = ()) -> None:
    """One bordered panel divided into cells; optional secondary row underneath.

    ``footer`` items are pre-escaped HTML fragments (built with ``esc``).
    """
    cells = []
    for s in items:
        color = f' style="color:{s.value_color}"' if s.value_color else ""
        bar = f'<span class="dl-bar"><i style="width:{max(0.0, min(1.0, s.bar)) * 100:.1f}%"></i></span>' if s.bar is not None else ""
        status = ""
        if s.tone:
            icon, _ = STATUS[s.tone]
            status = f'<span class="dl-status {s.tone}">{icon}{esc(s.caption)}</span>'
        caption = status or (f'<span class="c">{esc(s.caption)}</span>' if s.caption else "")
        cls = f' class="{s.tone}"' if s.tone == "review" else ""
        cells.append(
            f'<div{cls}><span class="dl-eyebrow">{esc(s.label)}</span>'
            f'<span class="v{" big" if big else ""}"{color}>{esc(s.value)}</span>{bar}{caption}</div>'
        )
    out = f'<section class="dl-strip">{"".join(cells)}</section>'
    if footer:
        out += f'<div class="dl-sub">{"".join(f"<span>{f}</span>" for f in footer)}</div>'
    st.html(out)


@dataclass(frozen=True)
class Check:
    title: str
    detail: str
    tone: Tone


def check_strip(checks: Sequence[Check]) -> None:
    cells = []
    for c in checks:
        icon, label = STATUS[c.tone]
        cls = ' class="review"' if c.tone == "review" else ""
        cells.append(
            f'<div{cls}><span class="dl-status {c.tone}">{icon}{label}</span>'
            f'<span style="font-size:14.5px;font-weight:600">{esc(c.title)}</span>'
            f'<span class="c">{esc(c.detail)}</span></div>'
        )
    st.html(f'<section class="dl-strip" aria-label="Ringkasan asumsi">{"".join(cells)}</section>')


def panel_title(title: str, caption: str = "") -> None:
    cap = f'<span style="font-size:12.5px;color:var(--dl-muted)">{esc(caption)}</span>' if caption else ""
    st.html(
        f'<div style="display:flex;flex-direction:column;gap:2px"><span style="font-size:15px;font-weight:600">'
        f"{esc(title)}</span>{cap}</div>"
    )


def empty_state(title: str, body: str, eyebrow: str = "") -> None:
    eb = f'<span class="dl-eyebrow">{esc(eyebrow)}</span>' if eyebrow else ""
    st.html(f'<div class="dl-empty">{eb}<h3>{esc(title)}</h3><p>{esc(body)}</p></div>')


def aside(eyebrow: str, title: str, items: Sequence[str]) -> None:
    lis = "".join(f"<li>{esc(i)}</li>" for i in items)
    st.html(f'<aside class="dl-aside"><span class="dl-eyebrow">{esc(eyebrow)}</span><h3>{esc(title)}</h3><ul>{lis}</ul></aside>')
