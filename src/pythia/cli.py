"""CLI entry point for Pythia."""
from __future__ import annotations

import typer

app = typer.Typer(name="pythia", help="Pythia — self-hosted AI search engine")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="API server host"),
    port: int = typer.Option(8900, help="API server port"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
) -> None:
    """Start the Pythia API server."""
    import uvicorn
    from pythia.config import load_config
    from pythia.server.app import create_app

    cfg = load_config(config)
    application = create_app(cfg)
    uvicorn.run(application, host=host, port=port)


@app.command()
def search(
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    no_auto_start: bool = typer.Option(False, help="Disable automatic service startup"),
    host: str = typer.Option("", help="Override API server host"),
    port: str = typer.Option("", help="Override API server port"),
) -> None:
    """Launch the Pythia TUI with automatic service startup."""
    from pythia.config import load_config
    from pythia.tui.app import run_tui

    cfg = load_config(config)
    if model:
        cfg.ollama.model = model
    
    auto_start = not no_auto_start
    host_override = host if host else None
    port_override = int(port) if port else None
    
    run_tui(cfg, auto_start=auto_start, host=host_override, port=port_override)


def main() -> None:
    app()
