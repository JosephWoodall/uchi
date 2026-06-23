from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, ProgressBar, Label, Static
from textual.containers import Horizontal, Vertical
from textual import work

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

_SEQUENCE = [0, 1, 0, 2, 0, 1, 4, 5, 0, 1, 0, 3, 0, 1, 0, 2, 1, 0, 6, 0, 1]

_MOODS = [
    "*wags tail*",
    "ready to help!",
    "thinking...",
    "learning...",
    "*sniffs*",
    "at your service!",
    "predicting...",
    "*tail wagging*",
    "processing...",
    "on it!",
]


class UchiApp(App):
    CSS = """
    Screen {
        background: #0d1117;
    }

    Header {
        background: #161b22;
        color: #58a6ff;
        text-style: bold;
    }

    Footer {
        background: #161b22;
        color: #8b949e;
    }

    #main-layout {
        height: 1fr;
        layout: horizontal;
    }

    #chat-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        border: round #30363d;
        background: #0d1117;
        color: #c9d1d9;
    }

    .rl-bar-section {
        height: auto;
        padding: 0 1;
    }

    #rl-label {
        color: #58a6ff;
        text-align: center;
        height: 1;
    }

    #rl-progress {
        height: 1;
    }

    #side-panel {
        width: 30;
        height: 100%;
        layout: vertical;
        padding: 0 1;
    }

    #dog-widget {
        height: 14;
        border: round #58a6ff;
        background: #161b22;
        content-align: center middle;
        color: #79c0ff;
        text-align: center;
        padding: 1 0;
    }

    #dog-mood {
        height: 1;
        text-align: center;
        color: #f0883e;
        padding: 0 1;
    }

    #stats-panel {
        height: 1fr;
        border: round #30363d;
        background: #161b22;
        padding: 1 1;
        color: #8b949e;
    }

    #input-box {
        dock: bottom;
        border: round #58a6ff;
        background: #0d1117;
        color: #c9d1d9;
        margin: 0 0 0 0;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+s", "save", "Save Brain"),
        ("ctrl+r", "reload_skills", "Reload Skills"),
    ]

    TITLE = "Uchi ODUSP"
    SUB_TITLE = "Omni-modal Deterministic Universal Sequence Predictor"

    def __init__(self, brain_path, preload_path, **kwargs):
        super().__init__(**kwargs)
        self.router = None
        self.brain_path = brain_path
        self.preload_path = preload_path
        self.active_learning_word = None
        self.active_learning_cmd = None
        self.active_teaching_query = None
        self.active_hole_context = None
        self._dog_idx = 0
        self._mood_idx = 0
        import time
        self.last_activity = time.time()
        self.rl_process = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", markup=True, highlight=True)
                with Vertical(classes="rl-bar-section hidden", id="rl-bar-section"):
                    yield Label("Bootstrapping...", id="rl-label")
                    yield ProgressBar(id="rl-progress", show_eta=False)
            with Vertical(id="side-panel"):
                yield Static(_DOG_FRAMES[0], id="dog-widget")
                yield Static("*wags tail*", id="dog-mood")
                yield Static(self._stats_text(), id="stats-panel")
        yield Input(placeholder="Initializing ODUSP...", id="input-box", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold #58a6ff]╔══════════════════════════════════════════╗[/bold #58a6ff]")
        log.write("[bold #58a6ff]║    Uchi ODUSP  —  v0.3.0                ║[/bold #58a6ff]")
        log.write("[bold #58a6ff]║    Omni-modal Deterministic Predictor    ║[/bold #58a6ff]")
        log.write("[bold #58a6ff]╚══════════════════════════════════════════╝[/bold #58a6ff]")
        log.write("[dim]Type [bold]/help[/bold] for commands and skills, or start chatting.[/dim]\n")
        self.initialize_brain()
        self.set_interval(2.0, self._tick_dog)
        self.set_interval(10.0, self._tick_stats)

    def on_input_changed(self, event: Input.Changed) -> None:
        import time
        self.last_activity = time.time()

    # ── Dog animation ─────────────────────────────────────────────────────────

    def _tick_dog(self) -> None:
        self._dog_idx = (self._dog_idx + 1) % len(_SEQUENCE)
        frame_idx = _SEQUENCE[self._dog_idx]
        self._mood_idx = (self._mood_idx + 1) % len(_MOODS)
        self.query_one("#dog-widget", Static).update(_DOG_FRAMES[frame_idx])
        self.query_one("#dog-mood", Static).update(_MOODS[self._mood_idx])

    def _tick_stats(self) -> None:
        self.query_one("#stats-panel", Static).update(self._stats_text())

    def _stats_text(self) -> str:
        lines = ["[bold #58a6ff]─ Stats ─[/bold #58a6ff]"]
        if self.router is not None:
            try:
                mem = len(self.router.memory.cpu_mem.records) if hasattr(self.router.memory, "cpu_mem") else 0
                lines.append(f"Memory:  {mem} records")
                lines.append(f"SSM μ:   {self.router.baseline.mean:.3f}")
                lines.append(f"SSM σ:   {self.router.baseline.std:.3f}")
                n_skills = len(self.router.skills.list_skills()) if hasattr(self.router, "skills") else 0
                lines.append(f"Skills:  {n_skills} loaded")
                has_code = self.router.specialist_pool.has_specialist("code") if hasattr(self.router, "specialist_pool") else False
                lines.append(f"Experts: {'code ' if has_code else ''}{'math ' if self.router.specialist_pool.has_specialist('math') else ''}{'convo' if self.router.specialist_pool.has_specialist('convo') else ''}")
            except Exception:
                pass
        else:
            lines.append("[dim]loading...[/dim]")
        import os
        bp = getattr(self, "brain_path", "brain.uchi")
        if os.path.exists(bp):
            mb = os.path.getsize(bp) / 1024 / 1024
            lines.append(f"Brain:   {mb:.1f} MB")
        lines.append("\n[dim]ctrl+s  save[/dim]")
        lines.append("[dim]ctrl+r  reload skills[/dim]")
        return "\n".join(lines)

    # ── Brain init ────────────────────────────────────────────────────────────

    @work(thread=True)
    def initialize_brain(self) -> None:
        self.call_from_thread(self.write_log, f"[dim][*] Loading brain from [bold]{self.brain_path}[/bold]...[/dim]")

        from uchi.cli import load_brain, preload_context, save_brain, ingest_file
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
                    self.call_from_thread(self._update_progress, cur, total)

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
            self.call_from_thread(self.write_log, f"[bold red]Init error:[/bold red] {e}")
        finally:
            builtins.print = _orig_print

    def write_log(self, msg: str) -> None:
        self.query_one(RichLog).write(msg)

    def on_brain_ready(self) -> None:
        n_skills = len(self.router.skills.list_skills())
        self.write_log(
            f"[bold #3fb950][+] Brain ready.[/bold #3fb950] "
            f"{n_skills} skills loaded. Type [bold]/help[/bold] to list them."
        )
        ib = self.query_one(Input)
        ib.disabled = False
        ib.placeholder = "Chat or /skill args..."
        ib.focus()
        self._tick_stats()
        # Start background RL daemon via the router (works for both TUI and API)
        self.router.start_background_jobs()

    def _update_progress(self, current: int, total: int) -> None:
        section = self.query_one("#rl-bar-section")
        bar = self.query_one("#rl-progress", ProgressBar)
        lbl = self.query_one("#rl-label", Label)
        if section.has_class("hidden"):
            section.remove_class("hidden")
            bar.total = total
        lbl.update(f"Bootstrapping persona... {current}/{total}")
        bar.progress = current

    def _hide_progress(self) -> None:
        self.query_one("#rl-bar-section").add_class("hidden")

    # ── Input handling ────────────────────────────────────────────────────────

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        cmd = message.value.strip()
        ib = self.query_one(Input)
        ib.value = ""
        log = self.query_one(RichLog)

        if not cmd:
            return

        log.write(f"\n[bold #f0883e]uchi>[/bold #f0883e] {cmd}")

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
            log.write("[bold #3fb950][+] File ingested.[/bold #3fb950]")
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
            log.write(f"[bold #3fb950][+] Hole filled![/bold #3fb950] Learned pattern for: [italic]{hole_desc}[/italic]")
            log.write(f"\n[#79c0ff][bold]Uchi:[/bold][/#79c0ff] {filled}")
            self.active_hole_context = None
            ib.disabled = False
            ib.placeholder = "Chat or /skill args..."
            ib.focus()
            return

        if self.active_teaching_query is not None:
            seq = (
                ["<|user|>"] + self.active_teaching_query.split()
                + ["<|assistant|>"] + cmd.split()
                + ["<|end|>"]
            )
            self.router.stream(seq)
            log.write(f"[bold #3fb950][+] Taught:[/bold #3fb950] '{self.active_teaching_query}' → '{cmd}'")
            self.active_teaching_query = None
            ib.disabled = False
            ib.placeholder = "Chat or /skill args..."
            ib.focus()
            return

        if self.active_learning_word is not None:
            self.router.tokenizer.ontology.add_mapping(self.active_learning_word, cmd)
            log.write(f"[#3fb950][+] Learned:[/#3fb950] '{self.active_learning_word}' → '{cmd}'")
            self.active_learning_word = None
            ib.disabled = True
            ib.placeholder = "ODUSP predicting..."
            self.process_command(self.active_learning_cmd)
            self.active_learning_cmd = None
            return

        # ── skill dispatch (/name args) ───────────────────────────────────────
        if cmd.startswith("/") and self.router is not None:
            parts = cmd[1:].split(None, 1)
            skill_name = parts[0].lower()
            skill_args = parts[1] if len(parts) > 1 else ""
            if self.router.skills.has(skill_name):
                ib.disabled = True
                ib.placeholder = f"Running /{skill_name}..."
                self.run_skill(skill_name, skill_args)
                return
            else:
                log.write(f"[yellow]Unknown command '/{skill_name}'. Type /help for list.[/yellow]")
                return

        # ── normal chat ───────────────────────────────────────────────────────
        ib.disabled = True
        ib.placeholder = "ODUSP predicting..."
        self.process_command(cmd)

    # ── Workers ───────────────────────────────────────────────────────────────

    @work(thread=True)
    def process_command(self, cmd: str) -> None:
        def on_event(event_type, msg):
            colours = {"reinforce": "#3fb950", "prune": "#f85149", "hallucination": "#d29922"}
            colour = colours.get(event_type, "white")
            self.call_from_thread(self.write_log, f"[{colour}]{msg}[/{colour}]")

        reply = self.router.chat(cmd, callback=on_event)
        self.call_from_thread(self.display_reply, cmd, reply)

    @work(thread=True)
    def run_skill(self, name: str, args: str) -> None:
        def on_event(event_type, msg):
            colours = {"reinforce": "#3fb950", "prune": "#f85149", "hallucination": "#d29922"}
            colour = colours.get(event_type, "white")
            self.call_from_thread(self.write_log, f"[{colour}]{msg}[/{colour}]")

        reply = self.router.skills.dispatch(name, args, callback=on_event)
        self.call_from_thread(self.display_reply, f"/{name} {args}".strip(), reply)

    # ── Display helpers ───────────────────────────────────────────────────────

    def display_reply(self, cmd: str, reply_text: str) -> None:
        log = self.query_one(RichLog)
        log.write(f"\n[#79c0ff][bold]Uchi:[/bold][/#79c0ff] {reply_text}")

        ib = self.query_one(Input)

        # Phase 3: hole detection
        from uchi.code_engine import CodeEngine
        holes = CodeEngine.extract_holes(reply_text)
        if holes:
            self.active_hole_context = (cmd, reply_text, holes[0])
            log.write(f"\n[bold #d29922][?] Hole detected — fill in the implementation:[/bold #d29922]")
            log.write(f"[#d29922]    {holes[0]}[/#d29922]")
            ib.placeholder = f"Fill: {holes[0][:38]}"
            ib.disabled = False
            ib.focus()
            return

        # Active teaching trigger
        if reply_text == "I do not have enough context to accurately predict a response to that yet. How should I respond?":
            self.active_teaching_query = cmd
            ib.placeholder = f"Teach response to: '{cmd[:28]}'"
        else:
            ib.placeholder = "Chat or /skill args..."

        ib.disabled = False
        ib.focus()

    def _show_help(self) -> None:
        log = self.query_one(RichLog)
        log.write("\n[bold #58a6ff]── Built-in Commands ──────────────────────────[/bold #58a6ff]")
        log.write("  [bold]/help[/bold]               Show this menu")
        log.write("  [bold]/save[/bold]               Serialize brain to disk")
        log.write("  [bold]/load[/bold] [italic]<path>[/italic]        Ingest a file into the brain")
        log.write("  [bold]/quit[/bold]               Exit")
        log.write("\n[bold #58a6ff]── Skills (/name args) ────────────────────────[/bold #58a6ff]")
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

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_save(self) -> None:
        if self.router:
            from uchi.cli import save_brain
            save_brain(self.router, self.brain_path)
            self.write_log("[bold #3fb950][+] Brain saved.[/bold #3fb950]")

    def action_quit(self) -> None:
        if self.router:
            from uchi.cli import save_brain
            save_brain(self.router, self.brain_path)
        if hasattr(self, "rl_process") and self.rl_process:
            self.rl_process.terminate()
        self.exit()

    def action_reload_skills(self) -> None:
        if self.router and hasattr(self.router, "skills"):
            self.router.skills.reload()
            n = len(self.router.skills.list_skills())
            self.write_log(f"[bold #3fb950][+] Skills reloaded — {n} loaded.[/bold #3fb950]")

    def prompt_active_learning(self, word: str) -> None:
        self.active_learning_word = word
        self.write_log(
            f"\n[#79c0ff][bold]Uchi:[/bold][/#79c0ff] "
            f"I'm unfamiliar with '{word}'. What's a synonym?"
        )
        ib = self.query_one(Input)
        ib.placeholder = f"Synonym for '{word}'> "
        ib.disabled = False
        ib.focus()
