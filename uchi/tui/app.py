import re
import threading

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, ProgressBar, Label, Static
from textual.containers import Horizontal, Vertical
from textual import work
from textual.binding import Binding

# ── ASCII dog animation ───────────────────────────────────────────────────────

_DOG_FRAMES = [
    # 0 — sitting, tail low
    "  /\\_____/\\\n"
    " ( ^o   o^ )\n"
    " (   ---   )\n"
    "  \\_______/ ~\n"
    "    |   |\n"
    "   _|___|_",

    # 1 — sitting, tail up
    "  /\\_____/\\\n"
    " ( ^o   o^ )\n"
    " (   ---   )\n"
    "  \\_______/~~\n"
    "    |   |\n"
    "   _|___|_",

    # 2 — blink
    "  /\\_____/\\\n"
    " ( ^-   o^ )\n"
    " (   ---   )\n"
    "  \\_______/ ~\n"
    "    |   |\n"
    "   _|___|_",

    # 3 — happy ears up
    "  /\\_____/\\\n"
    " ( *o   o* )\n"
    " (   www   )\n"
    "  \\_______/~~\n"
    "    |   |\n"
    "   _|___|_",

    # 4 — running left
    "  /\\_____/\\\n"
    " ( ^o   o^ )\n"
    " (   ---   )~\n"
    "   \\ ___ /\n"
    "  // | |  \\\n"
    " //  |    \\",

    # 5 — running right
    "  /\\_____/\\\n"
    " ( ^o   o^ )\n"
    " (   ---   )~~\n"
    "   \\ ___ /\n"
    "   / | |  \\\\\n"
    "  /  |     \\\\",

    # 6 — sleepy
    "  /\\_____/\\\n"
    " ( -u   u- )\n"
    " ( z  z  z )\n"
    "  \\_______/\n"
    "   |     |\n"
    "  [_______|",
]

_PRED_FRAMES = [4, 5, 4, 5, 3, 1, 0, 3]
_IDLE_FRAMES = [0, 1, 0, 2, 0, 1]

_PRED_MOODS = ["predicting...", "searching...", "on it!", "thinking...", "*nose twitching*", "processing..."]
_IDLE_MOODS = ["*wags tail*", "ready to help!", "*sniffs*", "at your service!"]

_DOG_SLEEP_SECS = 30


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
        height: 1fr;
        background: #16161e;
        color: #565f89;
    }

    /* ── Dog strip — above input, full width ── */
    #dog-strip {
        height: 8;
        background: #16161e;
        border-top: solid #1f2335;
        padding: 0 2;
    }

    #dog-widget {
        width: 20;
        height: 8;
        color: #7dcfff;
        content-align: center middle;
        text-align: center;
    }

    #dog-info {
        height: 8;
        padding: 1 2;
        layout: vertical;
        content-align: left middle;
    }

    #dog-title {
        color: #bb9af7;
        text-style: bold;
        height: 1;
    }

    #dog-mood {
        color: #ff9e64;
        height: 1;
    }

    #dog-desc {
        color: #565f89;
        height: 1;
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

    TITLE = "Uchi ODUSP"
    SUB_TITLE = "Omni-modal Deterministic Universal Sequence Predictor"

    def __init__(self, brain_path, preload_path, **kwargs):
        super().__init__(**kwargs)
        self.router         = None
        self.brain_path     = brain_path
        self.preload_path   = preload_path
        self.active_learning_word  = None
        self.active_learning_cmd   = None
        self.active_teaching_query = None
        self.active_hole_context   = None
        # Dog animation indices per state
        self._dog_pred_idx  = 0
        self._dog_idle_idx  = 0
        # Generation state
        self._predicting    = False
        self._cancel_event  = threading.Event()
        # Command history
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
                    yield Label("◈ thinking  (ctrl+t to pin)", id="think-label")
                    yield RichLog(id="think-log", markup=True, highlight=False)
            with Vertical(id="side-panel"):
                yield Static(self._stats_text(), id="stats-panel")
        with Horizontal(id="dog-strip"):
            yield Static(_DOG_FRAMES[0], id="dog-widget")
            with Vertical(id="dog-info"):
                yield Static("◈  UCHI  ODUSP", id="dog-title")
                yield Static("*wags tail*",    id="dog-mood")
                yield Static("Omni-modal Deterministic Sequence Predictor", id="dog-desc")
        yield Input(placeholder="Initializing ODUSP...", id="input-box", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold #7dcfff]╔══════════════════════════════════════════╗[/bold #7dcfff]")
        log.write("[bold #7dcfff]║    Uchi ODUSP  ◈  v0.3.0                ║[/bold #7dcfff]")
        log.write("[bold #7dcfff]║    Omni-modal Deterministic Predictor    ║[/bold #7dcfff]")
        log.write("[bold #7dcfff]╚══════════════════════════════════════════╝[/bold #7dcfff]")
        log.write("[dim]Type [bold]/help[/bold] for commands and skills, or start chatting.[/dim]\n")
        self.initialize_brain()
        self.set_interval(2.0,  self._tick_dog)
        self.set_interval(10.0, self._tick_stats)

    def on_input_changed(self, event: Input.Changed) -> None:
        import time
        self.last_activity = time.time()

    # ── Dog animation — stateful ──────────────────────────────────────────────

    def _tick_dog(self) -> None:
        import time
        if self._predicting:
            self._dog_pred_idx = (self._dog_pred_idx + 1) % len(_PRED_FRAMES)
            frame = _PRED_FRAMES[self._dog_pred_idx]
            mood  = _PRED_MOODS[self._dog_pred_idx % len(_PRED_MOODS)]
        elif time.time() - self.last_activity >= _DOG_SLEEP_SECS:
            frame = 6
            mood  = "z z z..."
        else:
            self._dog_idle_idx = (self._dog_idle_idx + 1) % len(_IDLE_FRAMES)
            frame = _IDLE_FRAMES[self._dog_idle_idx]
            mood  = _IDLE_MOODS[self._dog_idle_idx % len(_IDLE_MOODS)]
        self.query_one("#dog-widget", Static).update(_DOG_FRAMES[frame])
        self.query_one("#dog-mood",   Static).update(mood)

    def _tick_stats(self) -> None:
        self.query_one("#stats-panel", Static).update(self._stats_text())

    def _stats_text(self) -> str:
        lines = ["[bold #7dcfff]─ Stats ─[/bold #7dcfff]"]
        if self.router is not None:
            try:
                mem = len(self.router.memory.cpu_mem.records) if hasattr(self.router.memory, "cpu_mem") else 0
                lines.append(f"Mem   {mem}")
                lines.append(f"μ     {self.router.baseline.mean:.3f}")
                lines.append(f"σ     {self.router.baseline.std:.3f}")
                n_skills = len(self.router.skills.list_skills()) if hasattr(self.router, "skills") else 0
                lines.append(f"Skills {n_skills}")
                pool = getattr(self.router, "specialist_pool", None)
                if pool:
                    experts = " ".join(
                        e for e in ("code", "math", "convo")
                        if pool.has_specialist(e)
                    )
                    lines.append(f"[dim]{experts or 'none'}[/dim]")
                    
                import uchi.telemetry as _tel
                tel_data = _tel.dump_all()
                lines.append("\n[bold #bb9af7]─ Telemetry ─[/bold #bb9af7]")
                
                # L2 Norm
                l2 = tel_data.get("latent_space", {}).get("vector_l2_norm", "N/A")
                lines.append(f"L2: [dim]{l2}[/dim]")
                
                # MoE Experts
                # (Assuming Claude recorded MoE experts, if not, this will just gracefully fallback to N/A)
                moe = tel_data.get("latent_space", {}).get("active_experts", "N/A")
                lines.append(f"MoE: [dim]{moe}[/dim]")
                
                # FAISS Fallback
                faiss = tel_data.get("tokenizer", {}).get("faiss_fallback_count", 0)
                lines.append(f"FAISS: [dim]{faiss}[/dim]")
                
                # MCTS Batch Util
                batch = tel_data.get("mcts", {}).get("gpu_batch_utilization", "N/A")
                lines.append(f"Batch: [dim]{batch}[/dim]")
                
                # Repetition Penalty
                rep = tel_data.get("mcts", {}).get("repetition_penalty_applied", 0)
                lines.append(f"RepPen: [dim]{'Active' if rep > 0 else 'Inactive'}[/dim]")
                
            except Exception as e:
                lines.append(f"[dim]tel_err: {e}[/dim]")
        else:
            lines.append("[dim]loading...[/dim]")
        import os
        bp = getattr(self, "brain_path", "brain.uchi")
        if os.path.exists(bp):
            mb = os.path.getsize(bp) / 1024 / 1024
            lines.append(f"Brain  {mb:.1f}MB")
        lines.append("\n[dim]^s save  ^r skills[/dim]")
        return "\n".join(lines)

    # ── Brain init ────────────────────────────────────────────────────────────

    @work(thread=True)
    def initialize_brain(self) -> None:
        self.call_from_thread(self.write_log, f"[dim][*] Loading brain from [bold]{self.brain_path}[/bold]...[/dim]")

        from uchi.cli import load_brain, preload_context
        from uchi.omni_router import OmniRouter
        from uchi.node_compressor import NodeCompressor

        import builtins
        _orig_print = builtins.print
        def _ui_print(*args, **kwargs):
            self.call_from_thread(self.write_log, "[dim]" + " ".join(str(a) for a in args) + "[/dim]")
        builtins.print = _ui_print

        try:
            router = load_brain(self.brain_path)
            if router is None:
                self.call_from_thread(self.write_log, "[yellow][-] No brain found — cold start.[/yellow]")

                def _prog(cur, total):
                    self.call_from_thread(self._update_progress, cur, total, "Bootstrapping persona")

                router = OmniRouter(use_bpe=False, memory_window=5, progress_callback=_prog)
                self.call_from_thread(self._hide_progress)

                compressor = NodeCompressor()
                self.call_from_thread(self.write_log, "[dim][*] Compressing persona memory...[/dim]")
                pruned = compressor.compress_pass(router.predictor._pred._root, router.predictor._pred._cred_max_base)
                self.call_from_thread(self.write_log, f"[dim][+] Compressed {pruned} nodes.[/dim]")

            if self.preload_path:
                preload_context(router, self.preload_path)

            self.router = router
            self.call_from_thread(self.on_brain_ready)
        except Exception as e:
            self.call_from_thread(self.write_log, f"[bold #f7768e]Init error:[/bold #f7768e] {e}")
        finally:
            builtins.print = _orig_print

    def write_log(self, msg: str) -> None:
        self.query_one("#chat-log", RichLog).write(msg)

    def write_think(self, msg: str) -> None:
        self.query_one("#think-log", RichLog).write(msg)

    def on_brain_ready(self) -> None:
        n_skills = len(self.router.skills.list_skills())
        self.write_log(
            f"[bold #9ece6a][+] Brain ready.[/bold #9ece6a] "
            f"{n_skills} skills loaded. Type [bold]/help[/bold] to list them."
        )
        ib = self.query_one(Input)
        ib.disabled = False
        ib.placeholder = "Chat with Uchi, or /skill args..."
        ib.focus()
        self._tick_stats()
        self.router.start_background_jobs()

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

        # ── stateful input modes ──────────────────────────────────────────────
        if self.active_hole_context is not None:
            orig_cmd, code_with_holes, hole_desc = self.active_hole_context
            from uchi.code_engine import CodeEngine
            filled = CodeEngine.fill_hole(code_with_holes, hole_desc, cmd)
            seq = (
                ["<|user|>"] + orig_cmd.split()
                + ["<|assistant|>"] + filled.split()
                + ["<|end|>"]
            )
            self.router.stream(seq)
            log.write(f"[bold #9ece6a][+] Hole filled![/bold #9ece6a] Learned pattern for: [italic]{hole_desc}[/italic]")
            log.write(f"\n[#7dcfff][bold]Uchi:[/bold][/#7dcfff] {filled}")
            self.active_hole_context = None
            ib.disabled = False
            ib.placeholder = "Chat with Uchi, or /skill args..."
            ib.focus()
            return

        if self.active_teaching_query is not None:
            seq = (
                ["<|user|>"] + self.active_teaching_query.split()
                + ["<|assistant|>"] + cmd.split()
                + ["<|end|>"]
            )
            self.router.stream(seq)
            log.write(f"[bold #9ece6a][+] Taught:[/bold #9ece6a] '{self.active_teaching_query}' → '{cmd}'")
            self.active_teaching_query = None
            ib.disabled = False
            ib.placeholder = "Chat with Uchi, or /skill args..."
            ib.focus()
            return

        if self.active_learning_word is not None:
            self.router.tokenizer.ontology.add_mapping(self.active_learning_word, cmd)
            log.write(f"[#9ece6a][+] Learned:[/#9ece6a] '{self.active_learning_word}' → '{cmd}'")
            self.active_learning_word = None
            ib.disabled = True
            ib.placeholder = "ODUSP predicting..."
            self.process_command(self.active_learning_cmd)
            self.active_learning_cmd = None
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

    def _make_callback(self):
        cancel = self._cancel_event

        def on_event(event_type, msg):
            if cancel.is_set():
                raise InterruptedError("generation cancelled")
            if event_type == "thinking":
                self.call_from_thread(self.write_think, msg)
                m = re.search(r"rollout\s+(\d+)/(\d+)", msg)
                if m:
                    n, total = int(m.group(1)), int(m.group(2))
                    self.call_from_thread(self._update_progress, n, total, "Predicting")
            else:
                colours = {"reinforce": "#9ece6a", "prune": "#f7768e", "hallucination": "#e0af68"}
                colour  = colours.get(event_type, "white")
                self.call_from_thread(self.write_log, f"[{colour}]{msg}[/{colour}]")

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
            reply = self.router.chat(cmd, callback=self._make_callback())
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

        tokens = self.router.tokenizer.tokenize(raw_text.split(), is_inference=False)
        self.router.stream(tokens)

        from uchi.cli import save_brain
        save_brain(self.router, getattr(self, "brain_path", "brain.uchi"))

        self.call_from_thread(
            self.write_log,
            f"[bold #9ece6a][+] Ingested {len(tokens)} tokens from: {source_label}[/bold #9ece6a]",
        )
        self.call_from_thread(self._restore_input)

    def _restore_input(self) -> None:
        ib = self.query_one(Input)
        ib.disabled = False
        ib.placeholder = "Chat with Uchi, or /skill args..."
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

        from uchi.code_engine import CodeEngine
        holes = CodeEngine.extract_holes(reply_text)
        if holes:
            self.active_hole_context = (cmd, reply_text, holes[0])
            log.write("\n[bold #e0af68][?] Hole detected — fill in the implementation:[/bold #e0af68]")
            log.write(f"[#e0af68]    {holes[0]}[/#e0af68]")
            ib.placeholder = f"Fill: {holes[0][:38]}"
            ib.disabled = False
            ib.focus()
            return

        if reply_text == "I do not have enough context to accurately predict a response to that yet. How should I respond?":
            self.active_teaching_query = cmd
            ib.placeholder = f"Teach response to: '{cmd[:28]}'"
        else:
            ib.placeholder = "Chat with Uchi, or /skill args..."

        ib.disabled = False
        ib.focus()

    def _show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("\n[bold #7dcfff]── Built-in Commands ──────────────────────────[/bold #7dcfff]")
        log.write("  [bold]/help[/bold]               Show this menu")
        log.write("  [bold]/save[/bold]               Serialize brain to disk")
        log.write("  [bold]/load[/bold] [italic]<path>[/italic]        Ingest a file into the brain")
        log.write("  [bold]/learn[/bold] [italic]<url|text>[/italic]   Fetch a URL or ingest raw text permanently")
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
            from uchi.cli import save_brain
            self.router.stop_background_jobs()
            save_brain(self.router, self.brain_path)
        self.exit()

    def action_reload_skills(self) -> None:
        if self.router and hasattr(self.router, "skills"):
            self.router.skills.reload()
            n = len(self.router.skills.list_skills())
            self.write_log(f"[bold #9ece6a][+] Skills reloaded — {n} loaded.[/bold #9ece6a]")

    def prompt_active_learning(self, word: str) -> None:
        self.active_learning_word = word
        self.write_log(
            f"\n[#7dcfff][bold]Uchi:[/bold][/#7dcfff] "
            f"I'm unfamiliar with '{word}'. What's a synonym?"
        )
        ib = self.query_one(Input)
        ib.placeholder = f"Synonym for '{word}'> "
        ib.disabled = False
        ib.focus()
