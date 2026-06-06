import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference.domain_router import DomainRouter, DOMAINS

console = Console()

TEST_CASES = [
    ("let's play chess", "GAME"),
    ("go to the kitchen", "NAVIGATION"),
    ("explain photosynthesis to me", "STUDY"),
    ("tell me a story about a dragon", "STORY"),
    ("you're adorable little robot", "PET"),
    ("what's your favorite color", "CHAT"),
    ("turn up the volume", "SYSTEM"),
    ("quiz me on math", "STUDY"),
    ("stop moving", "NAVIGATION"),
    ("give me a hint", "GAME"),
    ("continue the story", "STORY"),
    ("restart yourself", "SYSTEM"),
    ("don't be sad little buddy", "PET"),
    ("tell me a joke", "CHAT"),
    ("follow me to the bedroom", "NAVIGATION"),
    ("start a game of trivia", "GAME"),
    ("what is the water cycle", "STUDY"),
    ("make up a story about a pirate", "STORY"),
    ("give me a hug", "PET"),
    ("how are you today", "CHAT"),
    ("mute the sound", "SYSTEM"),
    ("walk to the garden", "NAVIGATION"),
    ("i want to play word guess", "GAME"),
    ("tell me a bedtime story", "STORY"),
    ("do a spin for me", "PET"),
    ("change language to french", "SYSTEM"),
    ("what's the definition of gravity", "STUDY"),
    ("you're my best friend robot", "PET"),
    ("take me to the living room", "NAVIGATION"),
    ("what's my score", "GAME"),
]


def show_result(text, result):
    method = result["method"]
    tag = "sigmoid" if method == "sigmoid" else "kmeans (fallback)"
    color = "green" if method == "sigmoid" else "yellow"

    console.print()
    console.print(f"  [bold]method:[/bold]  [{color}]{tag}[/{color}]")

    if len(result["domains"]) > 1:
        console.print(f"  [bold]domains:[/bold] [bold cyan]{', '.join(result['domains'])}[/bold cyan]")
    else:
        console.print(
            f"  [bold]domain:[/bold]  [bold cyan]{result['top_domain']}[/bold cyan] "
            f"({result['top_score']:.4f})"
        )

    console.print()
    tbl = Table(title="sigmoid scores", show_header=True, header_style="bold", border_style="dim", pad_edge=False)
    tbl.add_column("domain", width=14)
    tbl.add_column("score", justify="right", width=8)
    tbl.add_column("thresh", justify="right", width=8)
    tbl.add_column("", justify="center", width=4)

    for d in DOMAINS:
        sc = result["sigmoid_scores"][d]
        th = result["sigmoid_thresholds"][d]
        hit = sc >= th
        st = "bold green" if hit else "dim"
        tbl.add_row(d, f"[{st}]{sc:.4f}[/{st}]", f"{th:.3f}",
                     "[bold green]✓[/bold green]" if hit else "[dim]✗[/dim]")
    console.print(tbl)

    if result["kmeans_scores"]:
        console.print()
        kt = Table(title="kmeans fallback", show_header=True, header_style="bold", border_style="dim", pad_edge=False)
        kt.add_column("domain", width=14)
        kt.add_column("cos sim", justify="right", width=8)
        kt.add_column("softmax", justify="right", width=8)
        kt.add_column("", justify="center", width=4)

        for d in DOMAINS:
            top = d == result["top_domain"]
            st = "bold yellow" if top else "dim"
            mk = "[bold yellow]←[/bold yellow]" if top else ""
            kt.add_row(f"[{st}]{d}[/{st}]",
                        f"[{st}]{result['kmeans_sims'][d]:.4f}[/{st}]",
                        f"[{st}]{result['kmeans_scores'][d]:.4f}[/{st}]", mk)
        console.print(kt)


def interactive(router):
    console.print(Panel(
        "[bold]wini_intent_pkg — Domain Router[/bold]\n"
        "[dim]type an utterance, or 'quit' to exit[/dim]",
        border_style="cyan", padding=(1, 2),
    ))

    while True:
        try:
            console.print()
            text = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            console.print("[dim]bye[/dim]")
            break

        show_result(text, router.route(text))


def batch(router):
    console.print(Panel(
        f"[bold]batch test — {len(TEST_CASES)} cases[/bold]",
        border_style="cyan", padding=(1, 2),
    ))

    correct = 0
    per_domain = {d: {"tp": 0, "total": 0} for d in DOMAINS}

    tbl = Table(title="results", show_header=True, header_style="bold", border_style="cyan")
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("utterance", width=40)
    tbl.add_column("expected", width=12)
    tbl.add_column("predicted", width=12)
    tbl.add_column("method", width=8)
    tbl.add_column("score", justify="right", width=8)
    tbl.add_column("", justify="center", width=3)

    for i, (text, expected) in enumerate(TEST_CASES, 1):
        r = router.route(text)
        pred = r["top_domain"]
        ok = pred == expected

        if ok:
            correct += 1
        per_domain[expected]["total"] += 1
        if ok:
            per_domain[expected]["tp"] += 1

        tbl.add_row(
            str(i), text, expected,
            f"[{'green' if ok else 'red'}]{pred}[/{'green' if ok else 'red'}]",
            r["method"], f"{r['top_score']:.4f}",
            "[bold green]✓[/bold green]" if ok else "[bold red]✗[/bold red]",
        )

    console.print(tbl)
    console.print()

    acc = correct / len(TEST_CASES)
    c = "green" if acc >= 0.9 else "yellow" if acc >= 0.7 else "red"
    console.print(f"  [bold]overall:[/bold] [{c}]{correct}/{len(TEST_CASES)} ({acc:.1%})[/{c}]")

    console.print()
    for d in DOMAINS:
        s = per_domain[d]
        if s["total"] > 0:
            da = s["tp"] / s["total"]
            console.print(f"  {d:<14} {s['tp']}/{s['total']} ({da:.0%})")


def main():
    parser = argparse.ArgumentParser(description="domain router demo")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    console.print("[dim]loading...[/dim]")
    router = DomainRouter(args.models_dir)
    console.print("[dim]ready[/dim]")

    if args.batch:
        batch(router)
    else:
        interactive(router)


if __name__ == "__main__":
    main()
