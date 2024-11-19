import sys
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from g4f.client import Client
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append(str(Path(__file__).resolve().parents[1]))

import nest_asyncio
nest_asyncio.apply()

from history_manager import HistoryManager

from Agent_tool import (
    APISearcher,
    Osinteraction,
    CodeGenerationSystem,
    WebSearchAgent,
    generate_fine_tuned_prompts,
    format_prompt
)
from rag_pipeline import Ragpipeline
import matplotlib
matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class TextGenerationPipeline:
    def __init__(self, model: str, max_history_size: int = 1000):
        self.client = Client()
        self.model = model
        self.history_manager = HistoryManager(max_size=max_history_size)
        self.api_searcher = APISearcher()
        self.code_generation_system = CodeGenerationSystem()
        self.web_search_agent = WebSearchAgent()


    @retry(stop=stop_after_attempt(30), wait=wait_random_exponential(min=1, max=120))
    def generate_text(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        try:
            messages = context or []
            messages.append({"role": "user", "content": query})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            generated_text = response.choices[0].message.content
            self.history_manager.add(query, generated_text)
            return generated_text
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise

    def generate_with_history(self, query: str, use_history: bool = True) -> str:
        context = []
        if use_history:
            history_df = self.history_manager.get_history_dataframe()
            recent_history = history_df.tail(5)  # Use last 5 interactions
            for _, row in recent_history.iterrows():
                context.append({"role": "user", "content": row['key']})
                context.append({"role": "assistant", "content": row['value']})

        return self.generate_text(query, context)

    def batch_generate(self, queries: List[str]) -> List[str]:
        return [self.generate_text(query) for query in queries]

    def get_generation_history(self) -> pd.DataFrame:
        return self.history_manager.get_history_dataframe()

    def export_history(self, filename: str) -> None:
        self.history_manager.export_history(filename)

    def import_history(self, filename: str) -> None:
        self.history_manager.import_history(filename)

    def run_api_search(self) -> None:
        self.api_searcher.run()

    def generate_and_execute_code(self, prompt: str, timeout: int) -> Dict[str, Any]:
        return self.code_generation_system.generate_and_execute_code(prompt, timeout)

    def web_search_and_synthesize(self, query: str) -> str:
        return self.web_search_agent.search_and_synthesize(query)

    def generate_fine_tuned_prompts(self, user_query: str) -> List[str]:
        return generate_fine_tuned_prompts(user_query)

    def format_prompt(self, prompt: str) -> str:
        return format_prompt(prompt)


async def interactive_loop(pipeline: TextGenerationPipeline):
    session = PromptSession(
        style=PromptStyle.from_dict({
            'prompt': '#00ff00 bold',
        })
    )

    command_completer = WordCompleter([
        'generate', 'history', 'export', 'import', 'help', 'exit',
        'api_search', 'os_interact', 'code_gen', 'web_search', 'selfprompting',
    ])

    console.print(Panel.fit(" 🤖 Welcome to the Advanced AI Assistant CLI! 🤖 by Hemanth Ai ", 
                            title="AI Assistant", 
                            border_style="cyan"))

    while True:
        try:
            user_input = await session.prompt_async(
                "\n💬 Enter a command: ",
                completer=command_completer
            )

            if user_input.lower() == 'exit':
                console.print("  👋🏼   Goodbye!  ")
                break

            elif user_input.lower() == 'generate':
                query = await session.prompt_async("Enter your query: ")
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    response = pipeline.generate_with_history(query)
                console.print(Panel(response, title="AI Response", border_style="cyan"))

            elif user_input.lower() == 'history':
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                     history = pipeline.get_generation_history()
                     table = Table(title="Generation History")
                     table.add_column("Query", style="cyan")
                     table.add_column("Response", style="magenta")
                     for _, row in history.iterrows():
                         table.add_row(row['key'], row['value'])
                     console.print(table)

            elif user_input.lower() == 'export':
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    filename = await session.prompt_async("Enter filename to export: ")
                    pipeline.export_history(filename)
                    console.print("History exported successfully!")

            elif user_input.lower() == 'import':
                filename = await session.prompt_async("Enter filename to import: ")
                pipeline.import_history(filename)
                console.print("History imported successfully!")

            elif user_input.lower() == 'api_search':
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    pipeline.run_api_search()
                console.print("API search completed.")


            elif user_input.lower() == 'os_interact':
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    Osinteraction(Hemanth=True)
                console.print("OS interaction completed.")

            elif user_input.lower() == 'code_gen':
                prompt = await session.prompt_async("Enter code generation prompt: ")
                timeout = 120  # Default timeout
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    result = pipeline.generate_and_execute_code(prompt, timeout)
                console.print(Panel(f"Output: {result['output']}\nError: {result['error']}\nCode:\n{result['code']}", 
                                    title="Code Generation Result", 
                                    border_style="cyan"))

            elif user_input.lower() == 'web_search':
                query = await session.prompt_async("Enter web search query: ")
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    result = pipeline.web_search_and_synthesize(query)
                console.print(Panel(result, title="Web Search Result", border_style="cyan"))


            elif user_input.lower() == 'selfprompting':
                query = await session.prompt_async("Enter query for self prompting : ")
                with console.status(f"[bold blue]🧠 Hemanth Thinking...[/bold blue]"):
                    fine_tuned_prompts = pipeline.generate_fine_tuned_prompts(query)
                for i, prompt in enumerate(fine_tuned_prompts, 1):
                    console.print(f"Prompt {i}: {prompt}")

            elif user_input.lower() == 'help':
                help_table = Table(title="Available Commands")
                help_table.add_column("Command", style="cyan")
                help_table.add_column("Description", style="magenta")
                help_table.add_row("generate", "Generate text based on a query")
                help_table.add_row("history", "Show generation history")
                help_table.add_row("export", "Export history to a file")
                help_table.add_row("import", "Import history from a file")
                help_table.add_row("api_search", "Perform API search")
                help_table.add_row("os_interact", "Perform OS interaction")
                help_table.add_row("code_gen", "Generate and execute code")
                help_table.add_row("web_search", "Perform web search and synthesis")
                help_table.add_row("selfprompting", "Rewrite prompt user prompt")
                help_table.add_row("help", "Show this help message")
                help_table.add_row("exit", "Exit the program")
                console.print(help_table)

            else:
                console.print("[bold red]Invalid command. Type 'help' for available commands.[/bold red]")

        except KeyboardInterrupt:
            console.print( "\n[bold yellow]Interrupted. Exiting...[/bold yellow]")
        except Exception as e:
            logger.error( f"[bold red]An unexpected error occurred: {str(e)}[/bold red]") 
            console.print( f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")


async def main():
    pipeline = TextGenerationPipeline(model="gpt-4o-mini")
    console.print("[bold magenta]Welcome to Advanced AI Assistan[/bold magenta]t")
    
    try:
        await interactive_loop(pipeline)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        console.print("An unexpected error occurred. Please check the logs.")
    finally:
        console.print("Saving final state...")
        pipeline.export_history("final_state.json")
        console.print("Program terminated successfully.")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())