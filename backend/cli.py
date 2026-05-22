import typer, asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

from app.services.ingestion_service import ingest_document
from app.services.chunking_service import chunk__document
from app.services.embedding_service import store_embeddings
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.generation_service import generate_answer
from app.services.reranker_service import rerank_chunks
    
app = typer.Typer(help="Personal Learning Assistant CLI")
console = Console()

@app.command()
def chat(path_file:str):
    path = Path(path_file)
    if not path.exists():
        console.print(f"[red]File not found: {path_file}[/red]")
        raise typer.Exit(code=1)

    console.print("[bold cyan]Processing document...[/bold cyan]")
    _total_chunks = 0
    _files_to_process = []

    if path.is_dir():
        _files_to_process = [
            file for file in path.iterdir() if file.is_file() and file.suffix in [".pdf", ".txt", ".docx"]
        ]
    else:
        _files_to_process = [path]

    for file in _files_to_process:
        doc = ingest_document(str(file))
        chunks = chunk__document(doc['text'], doc['source'])
        store_count = store_embeddings(chunks)
        _total_chunks += store_count
    console.print(f"[green]Indexed {_total_chunks} chunks from {path_file} successfully![/green]")

    asyncio.run(user_chat())
async def user_chat():
    while True:
        question = console.input("\n[bold blue]Ask a question (or type 'exit' to quit): [/bold blue]")
        if question.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break

        with console.status("[bold cyan]Generating answer...[/bold cyan]"):
            chunks = retrieve_relevant_chunks(question, top_k=3)
            reranked_chunks = rerank_chunks(question, chunks, top_k=3)
            result = await generate_answer(question, reranked_chunks)
        
        console.print(
            Panel(result["answer"], title="[bold green]Answer[/bold green]", border_style="green")
        )

        table = Table(title="Citations", show_header=True, header_style="bold yellow")
        table.add_column("Source Id")
        table.add_column("Document Source")
        table.add_column("Chunk Index")

        for citation in result["citations"]:
            table.add_row(
                str(citation["source_id"]),
                citation["source"],
                str(citation["chunk_index"])
            )
        console.print(table)

if __name__ == "__main__":
    app()