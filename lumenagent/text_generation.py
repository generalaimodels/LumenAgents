import re
from typing import List, Dict, Any
from datetime import datetime
import emoji
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from tabulate import tabulate
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from g4f.client import Client

class ChatAgent:
    def __init__(self) -> None:
        self.client = Client()
        self.console = Console()
        self.prompt_session = PromptSession(
            style=Style.from_dict({
                'prompt': 'ansigreen bold',
            })
        )
        self.conversation_history: List[Dict[str, str]] = []

    def process_query(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.0-mini",
                messages=self.conversation_history + [{"role": "user", "content": query}],
            )
            return response.choices[0].message.content
        except Exception as e:
            self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
            return "I apologize, but I encountered an error while processing your request."

    def format_response(self, response: str) -> str:
        response_with_emoji = emoji.emojize(response, language='alias')
        
        table_pattern = r'\|.*\|'
        if re.search(table_pattern, response_with_emoji):
            lines = response_with_emoji.split('\n')
            table_data = [line.split('|')[1:-1] for line in lines if re.match(table_pattern, line)]
            headers = table_data[0]
            data = table_data[1:]
            table = tabulate(data, headers, tablefmt="grid")
            response_with_emoji = re.sub(table_pattern, table, response_with_emoji, flags=re.MULTILINE)
        
        return response_with_emoji

    def visualize_data(self, data: List[Dict[str, Any]]) -> None:
        try:
            df = pd.DataFrame(data)
            fig: Figure = px.line(df, x='response', y='query', title='Data Visualization')
            fig.show()
        except Exception as e:
            self.console.print(f"[bold red]Failed to visualize data: {str(e)}[/bold red]")

    def run(self) -> None:
        while True:
            try:
                query = self.prompt_session.prompt("You: ")
                if query.lower() in ['exit', 'quit', 'bye']:
                    self.console.print("[bold green]Goodbye![/bold green]")
                    break

                response = self.process_query(query)
                formatted_response = self.format_response(response)
                self.console.print(f"[bold blue]Assistant:[/bold blue] {formatted_response}")

                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": response})

                if "visualize" in query.lower():
                    sample_data = [
                        {"date": "2023-01-01", "value": 100},
                        {"date": "2023-01-02", "value": 150},
                        {"date": "2023-01-03", "value": 120},
                        {"date": "2023-01-04", "value": 200},
                    ]
                    self.visualize_data(sample_data)

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted. Exiting...[/bold yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")

# if __name__ == "__main__":
#     agent = ChatAgent()
#     agent.run()