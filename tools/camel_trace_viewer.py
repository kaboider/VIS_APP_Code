#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for a CAMEL single-agent trace."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _short(value: object, limit: int = 150) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_run(source: Path) -> tuple[Path, dict]:
    if source.is_file():
        trace_path = source
        run_dir = source.parent.parent
    else:
        run_dir = source
        trace_path = run_dir / "logs" / "camel_single_trace.jsonl"
    if not trace_path.is_file():
        raise SystemExit(f"trace not found: {trace_path}")

    events = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            events.append({"event": "parse_error", "line": line_number, "error": str(exc)})

    start = next((event for event in events if event.get("event") == "start"), {})
    finish = next((event for event in reversed(events) if event.get("event") == "finish"), {})
    summary_doc = _read_json(run_dir / "logs" / "summary.json")
    summary = summary_doc.get("summary") or finish.get("summary", {}).get("summary", {})
    meta = _read_json(run_dir / "meta.json")

    steps = []
    tools = []
    model_calls = []
    for event in events:
        kind = event.get("event")
        if kind == "model_call":
            model_calls.append(event)
        if kind != "agent_step":
            continue
        response = event.get("response") or {}
        info = response.get("info") or {}
        usage = info.get("usage") or {}
        step_tools = []
        for index, tool in enumerate(info.get("tool_calls") or [], 1):
            args = tool.get("args") or {}
            result = tool.get("result")
            result_text = str(result or "")
            failed = result_text.lower().startswith(("error", "traceback")) or "rejected" in result_text.lower()
            label = args.get("id") or args.get("filename") or args.get("title") or tool.get("tool_name") or "tool"
            command = args.get("command") or args.get("filename") or args.get("title") or ""
            item = {
                "number": len(tools) + 1,
                "step": event.get("step"),
                "name": tool.get("tool_name") or "tool",
                "label": str(label),
                "preview": _short(command or result_text),
                "args": args,
                "result": result,
                "tool_call_id": tool.get("tool_call_id"),
                "failed": failed,
            }
            tools.append(item)
            step_tools.append(item["number"])
        steps.append({
            "step": event.get("step"),
            "input": event.get("input", ""),
            "time": event.get("time"),
            "usage": usage,
            "tool_numbers": step_tools,
            "terminated": response.get("terminated"),
            "termination_reasons": info.get("termination_reasons") or [],
            "final": ((response.get("msgs") or [{}])[0].get("content") or ""),
        })

    first_time = start.get("time")
    last_time = finish.get("time")
    duration = summary.get("elapsed_s")
    if duration is None and first_time and last_time:
        duration = round(last_time - first_time, 1)
    usage = {"prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0, "cached_tokens": 0}
    usage_sources = model_calls if model_calls else steps
    for source in usage_sources:
        source_usage = source.get("usage") or {}
        for key in usage:
            usage[key] += int(source_usage.get(key) or 0)

    if model_calls:
        model_call_display = str(len(model_calls))
        model_call_note = "exactly logged"
    elif tools:
        model_call_display = f"{max(2, len(tools))}–{len(tools) + len(steps)}"
        model_call_note = "estimated from tool loop"
    else:
        model_call_display = str(len(steps))
        model_call_note = "no tool loop"

    data = {
        "run_id": run_dir.name,
        "model": summary.get("model") or meta.get("model") or start.get("model") or "unknown",
        "task": meta.get("task") or summary.get("task") or "unknown",
        "variant": meta.get("variant") or "",
        "duration": duration,
        "is_error": summary.get("is_error"),
        "steps": steps,
        "tools": tools,
        "usage": usage,
        "model_calls": model_calls,
        "model_call_display": model_call_display,
        "model_call_note": model_call_note,
        "result": summary.get("result") or (steps[-1]["final"] if steps else ""),
        "trace_file": str(trace_path),
    }
    return run_dir, data


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"CAMEL trace · {data['run_id']}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f5f6f8;--surface:#fff;--surface2:#f0f2f5;--text:#17191d;--muted:#667085;--border:#dde1e7;--accent:#5b55d6;--accent-soft:#ebeafd;--danger:#c43d4b;--danger-soft:#fcebee;--ok:#247a52;--ok-soft:#e7f6ee;--terminal:#137b78;--file:#9a5d17;--shadow:0 12px 35px rgba(17,24,39,.08)}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111318;--surface:#191c22;--surface2:#22262e;--text:#eef0f4;--muted:#a3aab8;--border:#303641;--accent:#9893ff;--accent-soft:#292750;--danger:#ff8e9a;--danger-soft:#42232a;--ok:#72d8a7;--ok-soft:#18392b;--terminal:#65cbc7;--file:#efb86f;--shadow:0 12px 35px rgba(0,0,0,.28)}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} button{{font:inherit}} .app{{max-width:1440px;margin:auto;padding:28px}} .top{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}} .top>div{{min-width:0}} h1{{font-size:22px;margin:0 0 5px;font-weight:650;overflow-wrap:anywhere}} .sub{{color:var(--muted)}} .status{{padding:6px 10px;border-radius:999px;background:var(--ok-soft);color:var(--ok);font-weight:650}} .status.fail{{background:var(--danger-soft);color:var(--danger)}} .stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}} .stat,.panel{{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow)}} .stat{{padding:16px}} .stat span{{display:block;color:var(--muted);font-size:12px}} .stat strong{{display:block;font-size:23px;margin:4px 0 1px}} .stat small{{color:var(--muted)}} .tokenbar{{height:8px;background:var(--surface2);border-radius:99px;overflow:hidden;margin-top:10px;display:flex}} .tokenbar i{{display:block;background:var(--accent)}} .tokenbar i:last-child{{background:var(--terminal)}} .layout{{display:grid;grid-template-columns:minmax(420px,1.15fr) minmax(340px,.85fr);gap:16px;align-items:start}} .panel{{overflow:hidden}} .panelhead{{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}} h2{{font-size:15px;margin:0;font-weight:650}} .filters{{display:flex;gap:5px;flex-wrap:wrap}} .filter{{border:1px solid var(--border);background:transparent;color:var(--muted);padding:5px 9px;border-radius:7px;cursor:pointer}} .filter.active{{background:var(--accent-soft);border-color:var(--accent);color:var(--text)}} .timeline{{padding:14px 16px 18px}} .step{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:3px 0 9px}} .tool{{position:relative;width:100%;text-align:left;border:0;border-left:2px solid var(--border);background:transparent;color:var(--text);padding:10px 10px 10px 19px;cursor:pointer}} .tool:before{{content:"";position:absolute;width:9px;height:9px;border-radius:50%;left:-6px;top:16px;background:var(--accent);border:2px solid var(--surface)}} .tool.failed:before{{background:var(--danger)}} .tool:hover,.tool.selected{{background:var(--surface2)}} .tooltop{{display:flex;align-items:center;gap:8px}} .num{{color:var(--muted);font-variant-numeric:tabular-nums}} .kind{{font-weight:650}} .kind.terminal{{color:var(--terminal)}} .kind.file{{color:var(--file)}} .failedtag{{color:var(--danger);font-size:12px;margin-left:auto}} .preview{{display:block;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:3px}} .detail{{padding:16px;min-height:420px}} .detail h3{{font-size:17px;margin:0 0 4px}} .meta{{color:var(--muted);margin-bottom:16px;word-break:break-all}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin:15px 0 5px}} pre{{margin:0;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:310px;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}} .empty{{color:var(--muted);padding:40px 10px;text-align:center}} .final{{margin-top:16px}} .final .body{{padding:16px;white-space:pre-wrap}} @media(max-width:850px){{.app{{padding:16px}}.stats{{grid-template-columns:1fr 1fr}}.layout{{grid-template-columns:1fr}}.detail{{min-height:0}}}} @media(max-width:480px){{.stats{{grid-template-columns:1fr}}.top{{display:block}}.status{{display:inline-block;margin-top:10px}}}}
</style>
</head>
<body>
<main class="app">
  <header class="top"><div><h1 id="title"></h1><div class="sub" id="subtitle"></div></div><span class="status" id="status"></span></header>
  <section class="stats" aria-label="Run metrics">
    <div class="stat"><span>Outer steps</span><strong id="steps"></strong><small>agent.step calls</small></div>
    <div class="stat"><span>Model requests</span><strong id="calls"></strong><small id="callnote"></small></div>
    <div class="stat"><span>Tool calls</span><strong id="tools"></strong><small id="failures"></small></div>
    <div class="stat"><span>Total tokens</span><strong id="tokens"></strong><small id="tokenlabels"></small><div class="tokenbar" aria-label="Prompt and completion token split"><i id="promptbar"></i><i id="completionbar"></i></div></div>
  </section>
  <section class="layout">
    <div class="panel"><div class="panelhead"><h2>Tool-call timeline</h2><div class="filters" role="group" aria-label="Filter tool calls"><button class="filter active" data-filter="all">All</button><button class="filter" data-filter="terminal">Terminal</button><button class="filter" data-filter="file">Files</button><button class="filter" data-filter="failed">Failed</button></div></div><div class="timeline" id="timeline"></div></div>
    <aside class="panel"><div class="panelhead"><h2>Selected call</h2></div><div class="detail" id="detail"><div class="empty">Select a tool call to inspect its arguments and result.</div></div></aside>
  </section>
  <section class="panel final"><div class="panelhead"><h2>Final response</h2></div><div class="body" id="final"></div></section>
</main>
<script>
const DATA={payload};
const $=id=>document.getElementById(id), fmt=n=>new Intl.NumberFormat().format(n||0);
$('title').textContent=DATA.run_id;
$('subtitle').textContent=[DATA.task,DATA.variant,DATA.model,DATA.duration!=null?DATA.duration+'s':null].filter(Boolean).join(' · ');
$('status').textContent=DATA.is_error===false?'completed':DATA.is_error===true?'error':'unknown';
if(DATA.is_error!==false)$('status').classList.add('fail');
$('steps').textContent=DATA.steps.length; $('calls').textContent=DATA.model_call_display; $('callnote').textContent=DATA.model_call_note;
$('tools').textContent=DATA.tools.length; const failed=DATA.tools.filter(t=>t.failed).length; $('failures').textContent=failed?failed+' failed':'all completed';
$('tokens').textContent=fmt(DATA.usage.total_tokens); $('tokenlabels').textContent=fmt(DATA.usage.prompt_tokens)+' prompt · '+fmt(DATA.usage.completion_tokens)+' completion · '+fmt(DATA.usage.cached_tokens)+' cached';
const total=DATA.usage.total_tokens||1; $('promptbar').style.width=(DATA.usage.prompt_tokens/total*100)+'%'; $('completionbar').style.width=(DATA.usage.completion_tokens/total*100)+'%';
$('final').textContent=DATA.result||'No final response recorded.';
function category(t){{if(t.failed)return 'failed';if(t.name.includes('shell')||t.name.includes('terminal'))return 'terminal';return 'file'}}
function draw(filter='all'){{const root=$('timeline');root.textContent='';let lastStep=null;const visible=DATA.tools.filter(t=>filter==='all'||category(t)===filter||(filter==='file'&&category(t)==='file'));if(!visible.length){{root.innerHTML='<div class="empty">No matching calls.</div>';return}} visible.forEach(t=>{{if(t.step!==lastStep){{const s=document.createElement('div');s.className='step';s.textContent='Agent step '+t.step;root.appendChild(s);lastStep=t.step}}const b=document.createElement('button');b.className='tool'+(t.failed?' failed':'');b.dataset.number=t.number;b.innerHTML='<span class="tooltop"><span class="num">#'+t.number+'</span><span class="kind '+category(t)+'"></span>'+(t.failed?'<span class="failedtag">failed</span>':'')+'</span><span class="preview"></span>';b.querySelector('.kind').textContent=t.name;b.querySelector('.preview').textContent=t.label+(t.preview?' · '+t.preview:'');b.addEventListener('click',()=>select(t,b));root.appendChild(b)}})}}
function select(t,b){{document.querySelectorAll('.tool').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');const d=$('detail');d.textContent='';const h=document.createElement('h3');h.textContent='#'+t.number+' '+t.name;const m=document.createElement('div');m.className='meta';m.textContent='step '+t.step+(t.tool_call_id?' · '+t.tool_call_id:'');d.append(h,m);[['Arguments',t.args],['Result',t.result]].forEach(([label,value])=>{{const l=document.createElement('div');l.className='label';l.textContent=label;const p=document.createElement('pre');p.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);d.append(l,p)}})}}
document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));b.classList.add('active');draw(b.dataset.filter)}}));draw();
</script>
</body>
</html>"""


def build_viewer(source: str | Path, output: str | Path | None = None) -> Path:
    run_dir, data = load_run(Path(source).expanduser().resolve())
    destination = Path(output).expanduser().resolve() if output else run_dir / "logs" / "camel_trace_viewer.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(data), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="CAMEL run directory or camel_single_trace.jsonl")
    parser.add_argument("--output", help="Output HTML path (default: <run>/logs/camel_trace_viewer.html)")
    args = parser.parse_args()
    print(build_viewer(args.source, args.output))


if __name__ == "__main__":
    main()
