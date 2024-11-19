import sys
from pathlib import Path
from typing import List
import asyncio
import logging
from rich.console import Console
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.panel import Panel
from rich.table import Table
import nest_asyncio
nest_asyncio.apply()

sys.path.append(str(Path(__file__).resolve().parents[1]))
from Agent_tool import (
    APISearcher,
    Osinteraction,
    CodeGenerationSystem,
    WebSearchAgent,
    generate_fine_tuned_prompts,
    format_prompt,
    process_document ,
    build_vector_database, 
    format_prompt
)
from history_manager import HistoryManager
from text_generation import ChatAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


console = Console()
cmd_feature:List[str]=[
        "help",]

stp_cmd:List[str]=["quit","kill","stop","q","cancel","exit"]
generation_cmd:List[str]=['generate',"q","talking_model","text_generation","text_gen"]

history_manager = HistoryManager(max_size=1000)
def generate_with_history( query: str, use_history: bool = True) -> str:
        context = []
        if use_history:
            history_df =history_manager.get_history_dataframe()
            recent_history = history_df.tail(5)  # Use last 5 interactions
            for _, row in recent_history.iterrows():
                context.append({"role": "user", "content": row['key']})
                context.append({"role": "assistant", "content": row['value']})

        return generate_text(query, context)


async def interactive_loop():
    session = PromptSession(
        style=PromptStyle.from_dict({
            'prompt': '#00ff00 bold',
        })
    )

    command_completer = WordCompleter(cmd_feature)
    console.print(Panel.fit(" 🤖 Welcome to the Advanced AI Assistant CLI! 🤖 by Hemanth Ai ", 
                            title="AI Assistant", 
                            border_style="cyan"))

    while True:
        try:
            user_input = await session.prompt_async(
                "\n💬 Enter a command: ",
                completer=command_completer
            )

            if user_input.lower() == stp_cmd:
                console.print(" 👋🏼 Goodbye!  ")
                break

            elif user_input.lower() == generation_cmd:
                query = await session.prompt_async("Enter your query: ")
                with console.status("🧠 Hemanth Thinking..."):
                    gen_agent= ChatAgent()
                    response = gen_agent.process_query(query)
                console.print(Panel(response, title="AI Response", border_style="cyan"))
            


        except KeyboardInterrupt:
            console.print("\nOperation cancelled by user.")

            
        