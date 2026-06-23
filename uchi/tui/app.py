from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, ProgressBar, Label
from textual import work

class UchiApp(App):
    CSS = """
    #chat-log {
        height: 1fr;
        border: solid cyan;
        background: $surface;
    }
    #input-box {
        dock: bottom;
        margin: 1 0;
    }
    .hidden {
        display: none;
    }
    #rl-label {
        color: cyan;
        text-align: center;
        padding-top: 1;
    }
    """
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit Uchi"),
        ("ctrl+s", "save", "Save Brain"),
    ]

    def __init__(self, brain_path, preload_path, **kwargs):
        super().__init__(**kwargs)
        self.router = None
        self.brain_path = brain_path
        self.preload_path = preload_path
        self.active_learning_word = None
        self.active_learning_cmd = None
        self.active_teaching_query = None
        import time
        self.last_activity = time.time()
        self.rl_process = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", markup=True)
        yield Label("Bootstrapping RL Persona...", id="rl-label", classes="hidden")
        yield ProgressBar(id="rl-progress", show_eta=False, classes="hidden")
        yield Input(placeholder="Initializing ODUSP...", id="input-box", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold cyan]===============================================================[/bold cyan]")
        log.write("[bold cyan] Uchi v0.2.0 - Omni-modal Deterministic Sequence Predictor[/bold cyan]")
        log.write("[bold cyan]===============================================================[/bold cyan]")
        log.write("Type '/help' for a list of commands, or start typing to stream.\n")
        self.initialize_brain()
        
    def on_input_changed(self, event: Input.Changed) -> None:
        import time
        self.last_activity = time.time()

    @work(thread=True)
    def initialize_brain(self) -> None:
        self.call_from_thread(self.write_log, f"[*] Loading persistent brain state from {self.brain_path}...")
        
        # We handle imports here to prevent circular dependencies with CLI
        from uchi.cli import load_brain, preload_context, save_brain, ingest_file
        from uchi.omni_router import OmniRouter
        from uchi.node_compressor import NodeCompressor

        # Redirect standard output to the RichLog natively to prevent stderr / multiprocessing issues
        import builtins
        original_print = builtins.print
        def ui_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            self.call_from_thread(self.write_log, msg)
        builtins.print = ui_print
        
        try:
            router = load_brain(self.brain_path)
            if router is None:
                self.call_from_thread(self.write_log, "[-] Failed to load brain, creating new instance...")
                
                def progress_cb(current, total):
                    self.call_from_thread(self._update_progress, current, total)
                    
                router = OmniRouter(use_bpe=False, memory_window=5, progress_callback=progress_cb)
                self.call_from_thread(self._hide_progress)
                
                compressor = NodeCompressor()
                self.call_from_thread(self.write_log, "[*] Compressing hyper-reinforced persona memory...")
                pruned = compressor.compress_pass(router.predictor._pred._root, router.predictor._pred._cred_max_base)
                self.call_from_thread(self.write_log, f"[+] Compressed {pruned} foundational concept nodes.")
                
            if self.preload_path:
                preload_context(router, self.preload_path)
            
            self.router = router
            self.call_from_thread(self.on_brain_ready)
        except Exception as e:
            self.call_from_thread(self.write_log, f"[bold red]Initialization Error:[/bold red] {e}")
        finally:
            builtins.print = original_print

    def write_log(self, msg: str) -> None:
        self.query_one(RichLog).write(msg)

    def on_brain_ready(self) -> None:
        input_box = self.query_one(Input)
        input_box.disabled = False
        input_box.placeholder = "Type your message..."
        input_box.focus()
        
        # Start the Continuous RL Background Daemon
        import subprocess
        import os
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "background_rl.py")
        if os.path.exists(script_path):
            self.rl_process = subprocess.Popen(["python", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.query_one(RichLog).write("[cyan][*] Continuous RL Background Daemon Started.[/cyan]")
        
    def _update_progress(self, current: int, total: int) -> None:
        bar = self.query_one("#rl-progress", ProgressBar)
        lbl = self.query_one("#rl-label", Label)
        if bar.has_class("hidden"):
            bar.remove_class("hidden")
            lbl.remove_class("hidden")
            bar.total = total
        bar.progress = current

    def _hide_progress(self) -> None:
        bar = self.query_one("#rl-progress", ProgressBar)
        lbl = self.query_one("#rl-label", Label)
        bar.add_class("hidden")
        lbl.add_class("hidden")

        
    def action_save(self) -> None:
        if self.router:
            from uchi.cli import save_brain
            save_brain(self.router, self.brain_path)
            self.query_one(RichLog).write("[green]Brain saved.[/green]")
        
    def action_quit(self) -> None:
        if self.router:
            from uchi.cli import save_brain
            save_brain(self.router, self.brain_path)
        if hasattr(self, 'rl_process') and self.rl_process:
            self.rl_process.terminate()
        self.exit()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        cmd = message.value.strip()
        input_box = self.query_one(Input)
        input_box.value = ""
        log = self.query_one(RichLog)
        
        if not cmd:
            return
            
        log.write(f"\n[yellow]uchi>[/yellow] {cmd}")
        
        if cmd.lower() in ["/quit", "/exit"]:
            self.action_quit()
            return
            
        if cmd.lower() == "/help":
            log.write("[bold yellow]Available Commands:[/bold yellow]")
            log.write("  /help             Show this help menu")
            log.write("  /load <file>      Dynamically stream a new file")
            log.write("  /save             Force serialize brain")
            log.write("  /quit             Exit session")
            return
            
        if cmd.startswith("/load "):
            from uchi.cli import ingest_file
            ingest_file(self.router, cmd.split(" ", 1)[1])
            log.write("[green]File ingested.[/green]")
            return
            
        if cmd.startswith("/save"):
            self.action_save()
            return

        if self.active_teaching_query is not None:
            ans_tokens = cmd.split()
            # Reinforce the graph with the answer!
            full_sequence = ["<|user|>"] + self.active_teaching_query.split() + ["<|assistant|>"] + ans_tokens + ["<|end|>"]
            self.router.stream(full_sequence)
            log.write(f"[green][+] Active Teaching: Reinforced response for '{self.active_teaching_query}'[/green]")
            self.active_teaching_query = None
            
            input_box.disabled = False
            input_box.placeholder = "Type your message..."
            input_box.focus()
            return
            
        if self.active_learning_word is not None:
            ans = cmd
            word = self.active_learning_word
            self.router.tokenizer.ontology.add_mapping(word, ans)
            log.write(f"[green][+] Learned: '{word}' maps to '{ans}'. Re-evaluating...[/green]")
            self.active_learning_word = None
            
            input_box.disabled = True
            input_box.placeholder = "ODUSP is predicting..."
            self.process_command(self.active_learning_cmd)
            self.active_learning_cmd = None
            return


        input_box.disabled = True
        input_box.placeholder = "ODUSP is predicting..."
        self.process_command(cmd)

    @work(thread=True)
    def process_command(self, cmd: str) -> None:
        def on_rl_event(event_type, msg):
            if event_type == "reinforce":
                self.call_from_thread(self.query_one(RichLog).write, f"[green]{msg}[/green]")
            elif event_type == "prune":
                self.call_from_thread(self.query_one(RichLog).write, f"[red]{msg}[/red]")
            elif event_type == "hallucination":
                self.call_from_thread(self.query_one(RichLog).write, f"[yellow]{msg}[/yellow]")
                
        reply_text = self.router.chat(cmd, callback=on_rl_event)
        self.call_from_thread(self.display_reply, cmd, reply_text)

    def prompt_active_learning(self, word: str) -> None:
        self.active_learning_word = word
        log = self.query_one(RichLog)
        log.write(f"\n[cyan][bold]ODUSP (Reply):[/bold][/cyan] I am unfamiliar with the word '{word}'. What is a synonym for it?")
        input_box = self.query_one(Input)
        input_box.placeholder = f"uchi (teaching '{word}')> "
        input_box.disabled = False
        input_box.focus()

    def display_reply(self, cmd: str, reply_text: str) -> None:
        if reply_text == "I do not have enough context to accurately predict a response to that yet. How should I respond?":
            self.active_teaching_query = cmd
            
        log = self.query_one(RichLog)
        log.write(f"\n[cyan][bold]ODUSP (Reply):[/bold][/cyan] {reply_text}")
        input_box = self.query_one(Input)
        
        if self.active_teaching_query is not None:
            input_box.placeholder = f"uchi (teaching response to '{cmd}')> "
        else:
            input_box.placeholder = "Type your message..."
            
        input_box.disabled = False
        input_box.focus()
