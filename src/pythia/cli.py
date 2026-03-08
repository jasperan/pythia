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
) -> None:
    """Launch the Pythia TUI."""
    from pythia.config import load_config
    from pythia.tui.app import run_tui

    cfg = load_config(config)
    if model:
        cfg.ollama.model = model
    run_tui(cfg)


def main() -> None:
    app()
