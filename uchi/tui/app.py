import re
import threading

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, ProgressBar, Label, Static
from textual.containers import Horizontal, Vertical
from textual import work
from textual.binding import Binding



class UchiApp(App):
    # ── Cyberpunk / Tokyo Night palette ──────────────────────────────────────
    #   bg-deep    #1a1b26   midnight purple   (screen / chat area)
    #   bg-panel   #16161e   near-black        (sidebars, dog strip)
    #   bg-subtle  #1f2335   inky blue         (borders, dividers)
    #   cyan       #7dcfff   neon cyan         (primary accent)
    #   purple     #bb9af7   soft purple       (secondary accent)
    #   green      #9ece6a   terminal green    (success / reinforce)
    #   red        #f7768e   neon pink-red     (error / prune)
    #   amber      #e0af68   warm amber        (warning / hallucination)
    #   orange     #ff9e64   bright orange     (user prompt)
    #   text-dim   #565f89   muted lavender    (secondary text)
    #   text-base  #a9b1d6   soft lavender     (normal text)
    #   text-hi    #c0caf5   bright lavender   (bold / headings)

    CSS = """
    Screen {
        background: #1a1b26;
        layout: vertical;
    }

    Header {
        background: #16161e;
        color: #7dcfff;
        text-style: bold;
    }

    Footer {
        background: #16161e;
        color: #565f89;
    }

    /* ── Main split ── */
    #main-layout {
        height: 1fr;
        layout: horizontal;
    }

    /* ── Chat panel (left) ── */
    #chat-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        background: #1a1b26;
        color: #a9b1d6;
        padding: 0 1;
    }

    .rl-bar-section {
        height: auto;
        padding: 0 1;
        background: #16161e;
    }

    #rl-label {
        color: #7dcfff;
        text-align: center;
        height: 1;
    }

    #rl-progress {
        height: 1;
    }

    /* Think section — auto-collapses to label bar when idle */
    #think-section {
        height: 1;
        layout: vertical;
        background: #16161e;
    }

    #think-section.active {
        height: 14;
    }

    #think-section.pinned {
        height: 14;
    }

    #think-label {
        height: 1;
        background: #1f2335;
        color: #e0af68;
        text-align: center;
        text-style: bold;
    }

    #think-log {
        height: 1fr;
        background: #16161e;
        color: #565f89;
        padding: 0 1;
    }

    /* ── Stats sidebar (right) ── */
    #side-panel {
        width: 26;
        height: 100%;
        layout: vertical;
        background: #16161e;
        padding: 1 1;
    }

    #stats-panel {
        height: auto;
        background: #16161e;
        color: #565f89;
    }

    #glass-brain-panel {
        height: 1fr;
        background: #16161e;
        color: #565f89;
        margin-top: 1;
    }

    /* ── Telemetry strip — above input, full width ── */
    #telemetry-strip {
        height: 3;
        background: #16161e;
        border-top: solid #1f2335;
        padding: 0 2;
        layout: horizontal;
    }

    .telemetry-item {
        color: #7dcfff;
        height: 1;
        padding-right: 4;
    }

    .telemetry-val {
        color: #9ece6a;
        text-style: bold;
    }

    /* ── Input box — prominent command prompt ── */
    #input-box {
        margin: 1 2;
        border: heavy #7dcfff;
        background: #1a1b26;
        color: #c0caf5;
        padding: 0 1;
    }

    #input-box:focus {
        border: heavy #bb9af7;
    }

    /* ── Utilities ── */
    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "maybe_quit",      "Quit"),
        Binding("ctrl+s", "save",             "Save Brain"),
        Binding("ctrl+r", "reload_skills",    "Reload Skills"),
        Binding("ctrl+t", "toggle_think",     "Toggle Think"),
        Binding("up",     "history_prev",     "History ↑", show=False),
        Binding("down",   "history_next",     "History ↓", show=False),
    ]

    TITLE = "Uchi"
    SUB_TITLE = "Grounded answers, honest abstentions — runs on your machine"

    def __init__(self, brain_path, preload_path, **kwargs):
        super().__init__(**kwargs)
        self.router         = None
        self.brain_path     = brain_path
        self.preload_path   = preload_path
        self._predicting    = False
        self._cancel_event  = threading.Event()
        self._history: list[str] = []
        self._history_idx: int   = 0
        import time
        self.last_activity  = time.time()
        self.rl_process     = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", markup=True, highlight=True)
                with Vertical(classes="rl-bar-section hidden", id="rl-bar-section"):
                    yield Label("Bootstrapping...", id="rl-label")
                    yield ProgressBar(id="rl-progress", show_eta=False)
                with Vertical(id="think-section"):
                    yield Label("🧠 What I'm thinking  (ctrl+t to keep open)", id="think-label")
                    yield RichLog(id="think-log", markup=True, highlight=False)
            with Vertical(id="side-panel"):
                yield Static(self._stats_text(), id="stats-panel")
                yield Static(self._glass_brain_text(), id="glass-brain-panel")
        with Horizontal(id="telemetry-strip"):
            yield Static("Multi-step reasoning: [bold #9ece6a]ready[/bold #9ece6a]", classes="telemetry-item")
            yield Static("Fact-checking: [bold #9ece6a]ready[/bold #9ece6a]", classes="telemetry-item")
            yield Static("Code sandbox: [bold #9ece6a]ready[/bold #9ece6a]", classes="telemetry-item")
        yield Input(placeholder="Getting Uchi ready...", id="input-box", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold #7aa2f7]◆ Uchi v0.3.0[/bold #7aa2f7]  [dim]— FLUX proposes, Uchi verifies[/dim]")
        log.write("[dim]I ground every answer in what I actually know, and say so honestly when I don't.[/dim]")
        log.write("[dim]Type [bold]/help[/bold] for commands, or just start chatting.[/dim]\n")
        self.initialize_brain()
        self.set_interval(10.0, self._tick_stats)
        self.set_interval(1.0, self._tick_glass_brain)

    def on_input_changed(self, event: Input.Changed) -> None:
        import time
        self.last_activity = time.time()



    def _tick_stats(self) -> None:
        self.query_one("#stats-panel", Static).update(self._stats_text())

    def _tick_glass_brain(self) -> None:
        """0.4.0 Item 14 — refreshed faster than the 10s stats tick so the
        tool-call trace feels live while a request is in flight."""
        self.query_one("#glass-brain-panel", Static).update(self._glass_brain_text())

    def _glass_brain_text(self) -> str:
        """0.4.0 Item 14 — live tool-call trace / goal state / pending
        HitL yield, refreshed on the same tick as the stats panel."""
        if self.router is None:
            return "[bold #bb9af7]─ Glass Brain ─[/bold #bb9af7]\n[dim]starting up...[/dim]"
        from uchi.tui.glass_brain import render_glass_brain
        try:
            return render_glass_brain(self.router)
        except Exception:
            return "[bold #bb9af7]─ Glass Brain ─[/bold #bb9af7]\n[dim]unavailable[/dim]"

    def _stats_text(self) -> str:
        lines = ["[bold #7aa2f7]─ At a glance ─[/bold #7aa2f7]"]
        if self.router is not None:
            try:
                n_skills = len(self.router.skills.list_skills()) if hasattr(self.router, "skills") else 0
                n_facts = len(getattr(self.router.index, "passages", []) or [])
                lines.append(f"[bold #9ece6a]{n_skills} skills[/bold #9ece6a]")
                lines.append(f"[bold #9ece6a]{n_facts:,} facts known[/bold #9ece6a]")
                if self.router.proposer:
                    lines.append("[bold #9ece6a]Reasoning: on[/bold #9ece6a]")
                else:
                    lines.append("[bold #e0af68]Reasoning: extractive only[/bold #e0af68]")
            except Exception:
                pass
        else:
            lines.append("[dim]starting up...[/dim]")
        import os
        bp = getattr(self, "brain_path", "brain.uchi")
        if os.path.exists(bp):
            mb = os.path.getsize(bp) / 1024 / 1024
            lines.append(f"Saved brain  {mb:.1f}MB")
        lines.append("\n[dim]^s save  ^r reload skills[/dim]")
        return "\n".join(lines)

    # ── Brain init ────────────────────────────────────────────────────────────

    @work(thread=True)
    def initialize_brain(self) -> None:
        self.call_from_thread(self.write_log, "[dim]Waking up...[/dim]")

        from uchi.simple import Core

        try:
            router = Core()
            self.router = router
            self.call_from_thread(self.on_brain_ready)
        except Exception as e:
            self.call_from_thread(self.write_log, f"[bold #f7768e]Couldn't start up:[/bold #f7768e] {e}")

    def write_log(self, msg: str) -> None:
        self.query_one("#chat-log", RichLog).write(msg)

    def write_think(self, msg: str) -> None:
        self.query_one("#think-log", RichLog).write(msg)

    def on_brain_ready(self) -> None:
        n_skills = len(self.router.skills.list_skills())
        n_facts = len(getattr(self.router.index, "passages", []) or [])
        knowledge_note = (
            f"{n_facts:,} general-knowledge facts pre-loaded" if n_facts
            else "no prior knowledge yet — teach me with /learn"
        )
        self.write_log(
            f"[bold #9ece6a]Ready![/bold #9ece6a] "
            f"{n_skills} skills available, {knowledge_note}. "
            f"Type [bold]/help[/bold] any time."
        )
        ib = self.query_one(Input)
        ib.disabled = False
        ib.placeholder = "Ask me anything, or try /classify, /forecast, ..."
        ib.focus()
        self._tick_stats()
        self._tick_stats()

    def _update_progress(self, current: int, total: int, label: str = "Working") -> None:
        section = self.query_one("#rl-bar-section")
        bar     = self.query_one("#rl-progress", ProgressBar)
        lbl     = self.query_one("#rl-label",    Label)
        if section.has_class("hidden"):
            section.remove_class("hidden")
            bar.total = total
        lbl.update(f"{label}... {current}/{total}")
        bar.progress = current

    def _hide_progress(self) -> None:
        self.query_one("#rl-bar-section").add_class("hidden")

    # ── Think pane expand / collapse ─────────────────────────────────────────

    def _expand_think(self) -> None:
        self.query_one("#think-section").add_class("active")

    def _collapse_think(self) -> None:
        section = self.query_one("#think-section")
        if not section.has_class("pinned"):
            section.remove_class("active")

    # ── Command history ───────────────────────────────────────────────────────

    def _push_history(self, cmd: str) -> None:
        if cmd and (not self._history or self._history[-1] != cmd):
            self._history.append(cmd)
        self._history_idx = len(self._history)

    def action_history_prev(self) -> None:
        ib = self.query_one(Input)
        if not ib.has_focus or not self._history:
            return
        self._history_idx = max(0, self._history_idx - 1)
        ib.value = self._history[self._history_idx]
        ib.cursor_position = len(ib.value)

    def action_history_next(self) -> None:
        ib = self.query_one(Input)
        if not ib.has_focus or not self._history:
            return
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            ib.value = self._history[self._history_idx]
            ib.cursor_position = len(ib.value)
        else:
            self._history_idx = len(self._history)
            ib.value = ""

    # ── Input handling ────────────────────────────────────────────────────────

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        cmd = message.value.strip()
        ib  = self.query_one(Input)
        ib.value = ""
        log = self.query_one("#chat-log", RichLog)

        if not cmd:
            return

        self._push_history(cmd)
        log.write(f"\n[bold #ff9e64]uchi>[/bold #ff9e64] {cmd}")

        # ── built-in TUI commands ─────────────────────────────────────────────
        if cmd.lower() in ("/quit", "/exit"):
            self.action_quit()
            return

        if cmd.lower() == "/help":
            self._show_help()
            return

        if cmd.lower() == "/save":
            self.action_save()
            return

        if cmd.startswith("/load "):
            from uchi.cli import ingest_file
            ingest_file(self.router, cmd.split(" ", 1)[1].strip())
            log.write("[bold #9ece6a][+] File ingested.[/bold #9ece6a]")
            return

        if cmd.startswith("/learn "):
            target = cmd.split(" ", 1)[1].strip()
            ib.disabled = True
            ib.placeholder = f"Learning from {target[:40]}..."
            self.learn_from(target)
            return



        # ── skill dispatch (/name args) ───────────────────────────────────────
        if cmd.startswith("/") and self.router is not None:
            parts      = cmd[1:].split(None, 1)
            skill_name = parts[0].lower()
            skill_args = parts[1] if len(parts) > 1 else ""
            if self.router.skills.has(skill_name):
                ib.disabled = True
                ib.placeholder = f"Running /{skill_name}... (ctrl+c to cancel)"
                self.run_skill(skill_name, skill_args)
                return
            else:
                log.write(f"[yellow]Unknown command '/{skill_name}'. Type /help for list.[/yellow]")
                return

        # ── normal chat ───────────────────────────────────────────────────────
        ib.disabled = True
        ib.placeholder = "ODUSP predicting... (ctrl+c to cancel)"
        self.process_command(cmd)

    # ── Workers ───────────────────────────────────────────────────────────────

    # Internal-dialogue event tags: every step of Uchi's live thought process
    # (status updates, FLUX's actual reasoning trace, the Devil's Advocate
    # critique, verification outcomes) surfaces here with a clear label — this
    # IS the "glass box" view into what's happening, kept out of the main
    # conversation so that stays clean and readable.
    _THINK_TAGS = {
        "thinking":      ("💭", "Working",   "#7aa2f7"),
        "reasoning":     ("🧠", "Reasoning", "#bb9af7"),
        "critique":      ("⚖️", "Critique",  "#e0af68"),
        "reinforce":     ("✓",  "Verified",  "#9ece6a"),
        "prune":         ("✗",  "Rejected",  "#f7768e"),
        "hallucination": ("⚠",  "Flagged",   "#e0af68"),
    }

    def _make_callback(self):
        cancel = self._cancel_event

        def on_event(event_type, msg):
            if cancel.is_set():
                raise InterruptedError("generation cancelled")
            icon, label, colour = self._THINK_TAGS.get(event_type, ("•", event_type, "#a9b1d6"))
            self.call_from_thread(self.write_think, f"[{colour}]{icon} {label}:[/{colour}] {msg}")
            if event_type == "thinking":
                m = re.search(r"rollout\s+(\d+)/(\d+)", msg)
                if m:
                    n, total = int(m.group(1)), int(m.group(2))
                    self.call_from_thread(self._update_progress, n, total, "Working")

        return on_event

    def _begin_predict(self) -> None:
        self._predicting = True
        self._cancel_event.clear()

    def _end_predict(self) -> None:
        self._predicting = False
        self.call_from_thread(self._hide_progress)
        self.call_from_thread(self._collapse_think)

    @work(thread=True)
    def process_command(self, cmd: str) -> None:
        self.call_from_thread(self._begin_predict)
        self.call_from_thread(self._reset_think_log, cmd)
        try:
            reply = self.router.ask(cmd, callback=self._make_callback())
        except InterruptedError:
            self.call_from_thread(self.write_log, "[yellow]Generation cancelled.[/yellow]")
            self.call_from_thread(self._restore_input)
            return
        except Exception as e:
            self.call_from_thread(self.write_log, f"[bold #f7768e]Error:[/bold #f7768e] {e}")
            self.call_from_thread(self._restore_input)
            return
        finally:
            self._end_predict()
        self.call_from_thread(self.display_reply, cmd, reply)

    @work(thread=True)
    def run_skill(self, name: str, args: str) -> None:
        self.call_from_thread(self._begin_predict)
        self.call_from_thread(self._reset_think_log, f"/{name} {args}".strip())
        try:
            reply = self.router.skills.dispatch(name, args, callback=self._make_callback())
        except InterruptedError:
            self.call_from_thread(self.write_log, "[yellow]Generation cancelled.[/yellow]")
            self.call_from_thread(self._restore_input)
            return
        except Exception as e:
            self.call_from_thread(self.write_log, f"[bold #f7768e]Error:[/bold #f7768e] {e}")
            self.call_from_thread(self._restore_input)
            return
        finally:
            self._end_predict()
        self.call_from_thread(self.display_reply, f"/{name} {args}".strip(), reply)

    @work(thread=True)
    def learn_from(self, target: str) -> None:
        raw_text     = target
        source_label = target[:60]

        if target.startswith(("http://", "https://")):
            try:
                import requests as _req
                from bs4 import BeautifulSoup
                resp = _req.get(target, timeout=15, headers={"User-Agent": "Uchi/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                raw_text = soup.get_text(separator=" ", strip=True)
            except Exception as exc:
                self.call_from_thread(
                    self.write_log,
                    f"[bold #f7768e][-] Failed to fetch URL: {exc}[/bold #f7768e]",
                )
                self.call_from_thread(self._restore_input)
                return

        if not raw_text.strip():
            self.call_from_thread(self.write_log, "[yellow]No usable text found.[/yellow]")
            self.call_from_thread(self._restore_input)
            return

        self.router.learn(raw_text)

        self.call_from_thread(
            self.write_log,
            f"[bold #9ece6a][+] Successfully ingested knowledge from: {source_label}[/bold #9ece6a]",
        )
        self.call_from_thread(self._restore_input)

    def _restore_input(self) -> None:
        ib = self.query_one(Input)
        ib.disabled = False
        ib.placeholder = "Ask me anything, or try /classify, /forecast, ..."
        ib.focus()

    # ── Display helpers ───────────────────────────────────────────────────────

    def display_reply(self, cmd: str, reply_text: str) -> None:
        from rich.markdown import Markdown
        from uchi.response_normalizer import normalize
        reply_text = normalize(reply_text)
        log = self.query_one("#chat-log", RichLog)
        log.write("\n[#7dcfff][bold]Uchi:[/bold][/#7dcfff]")
        log.write(Markdown(reply_text))

        ib = self.query_one(Input)

        ib.placeholder = "Ask me anything, or try /classify, /forecast, ..."

        ib.disabled = False
        ib.focus()

    def _show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("\n[bold #7aa2f7]── Commands ──────────────────────────[/bold #7aa2f7]")
        log.write("  [bold]/help[/bold]               Show this menu")
        log.write("  [bold]/save[/bold]               Save what I've learned to disk")
        log.write("  [bold]/load[/bold] [italic]<path>[/italic]        Teach me from a file")
        log.write("  [bold]/learn[/bold] [italic]<url|text>[/italic]   Teach me from a URL or raw text")
        log.write("  [bold]/quit[/bold]               Exit")
        log.write("  [bold]↑/↓[/bold]                 Cycle command history")
        log.write("  [bold]ctrl+c[/bold]              Cancel generation (or quit if idle)")
        log.write("  [bold]ctrl+t[/bold]              Pin/unpin the thinking pane")
        log.write("\n[bold #7dcfff]── Skills (/name args) ────────────────────────[/bold #7dcfff]")
        if self.router is not None:
            for s in self.router.skills.list_skills():
                log.write(
                    f"  [bold]/{s.name:<12}[/bold] "
                    f"[dim]{s.args_hint:<18}[/dim] "
                    f"{s.description}"
                )
            log.write(
                "\n[dim]Drop .md files in [bold]~/.uchi/skills/[/bold] to add your own.[/dim]"
            )

    def _reset_think_log(self, query: str) -> None:
        self._expand_think()
        tlog  = self.query_one("#think-log", RichLog)
        tlog.clear()
        short = query[:60] + ("…" if len(query) > 60 else "")
        tlog.write(f"[bold #e0af68]◈ {short}[/bold #e0af68]")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_maybe_quit(self) -> None:
        if self._predicting:
            self._cancel_event.set()
            self.write_log("[yellow]Cancelling generation...[/yellow]")
        else:
            self.action_quit()

    def action_toggle_think(self) -> None:
        """Pin think pane open, or unpin (returns to auto-collapse)."""
        section = self.query_one("#think-section")
        section.toggle_class("pinned")
        if section.has_class("pinned"):
            section.add_class("active")

    def action_save(self) -> None:
        if self.router:
            from uchi.cli import save_brain
            save_brain(self.router, self.brain_path)
            self.write_log("[bold #9ece6a][+] Brain saved.[/bold #9ece6a]")

    def action_quit(self) -> None:
        if self.router:
            pass
        self.exit()

    def action_reload_skills(self) -> None:
        if self.router and hasattr(self.router, "skills"):
            self.router.skills.reload()
            n = len(self.router.skills.list_skills())
            self.write_log(f"[bold #9ece6a][+] Skills reloaded — {n} loaded.[/bold #9ece6a]")

