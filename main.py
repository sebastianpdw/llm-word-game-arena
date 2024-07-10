import argparse
import csv
import os
import time
from typing import List, Dict, Optional

from loguru import logger as lg
from ollama import chat


def swap_roles(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Swaps the roles of 'user' and 'assistant' in the given messages.

    Args:
        messages: A list of message dictionaries with 'role' and 'content'.

    Returns:
        A list of messages with swapped roles.
    """
    messages_swapped = []
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if role == "user":
            role_reversed = "assistant"
        elif role == "assistant":
            role_reversed = "user"
        else:
            role_reversed = role
        message_reversed = {
            "role": role_reversed,
            "content": content
        }
        messages_swapped.append(message_reversed)
    return messages_swapped


def run_experiment(
        experiment_id: int,
        model_a: str,
        model_b: str,
        starting_animal: str = "Giraffe",
        max_turns: int = 200,
        csv_file: str = './data/game_results.csv',
        fieldnames: Optional[List[str]] = None
) -> None:
    """Runs a single experiment between two models.

    Args:
        experiment_id: The ID of the experiment.
        model_a: The name of model A.
        model_b: The name of model B.
        starting_animal: The animal to start the game with.
        max_turns: The maximum number of turns in the game.
        csv_file: The file to save the results.
        fieldnames: The field names for the CSV file.
    """
    if fieldnames is None:
        fieldnames = ['experiment_number', 'winner', 'reason']

    def save_winner_to_csv(experiment_number: int, model_winner: str, reason_winner: str) -> None:
        """Saves the winner of the experiment to the CSV file.

        Args:
            experiment_number: The number of the experiment.
            model_winner: The winner model.
            reason_winner: The reason for winning.
        """
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'experiment_number': experiment_number, 'winner': model_winner, 'reason': reason_winner})

    messages = [
        {
            "role": "system",
            "content": """
                You are playing a game of word-snake.
                In this game, you need to respond with an animal name that starts with the last letter of the previous animal mentioned.
                You are not allowed to repeat any animal that has already been mentioned.
                If the other player breaks a rule, respond with: 'Disqualified [reason].'
                If you can't think of a valid animal name, respond with: 'I forfeit the game.'
                Otherwise, Respond only with the animal name. Do not include any other text.
                """
        },
        {
            "role": "user",
            "content": starting_animal
        }
    ]

    lg.debug(f"{experiment_id}-0-A:  {starting_animal}")
    for i in range(0, max_turns + 1):

        if i % 2 == 0:
            current_model_name = "A"
            response = chat(model_a, messages, stream=False)
            response_content = response["message"]["content"]
            response_role = "assistant"
        else:
            current_model_name = "B"
            swapped_messages = swap_roles(messages)
            response = chat(model_b, swapped_messages, stream=False)
            response_content = response["message"]["content"]
            response_role = "user"
        other_model_name = "B" if current_model_name == "A" else "A"

        response_content = response_content.strip()

        messages.append(
            {
                "role": response_role,
                "content": response_content
            }
        )

        lg.debug(f"{experiment_id}-{i + 1}-{current_model_name}:  {response_content}")

        if "forfeit" in response_content.lower():
            winner = "Model B" if current_model_name == "A" else "Model A"
            reason = f"{current_model_name} forfeited"
            save_winner_to_csv(experiment_id, winner, reason)
            lg.info(f"Experiment {experiment_id}: Winner: {winner} - {reason}")
            break
        elif "disqualified" in response_content.lower():
            winner = "Model B" if current_model_name == "B" else "Model A"
            reason = f"{other_model_name} disqualified: {response_content}"
            save_winner_to_csv(experiment_id, winner, reason)
            lg.info(f"Experiment {experiment_id}: Winner: {winner} - {reason}")
            break
        elif len(response_content) > 30:
            winner = "Model B" if current_model_name == "A" else "Model A"
            reason = f"{current_model_name} response too long (so not an animal)"
            save_winner_to_csv(experiment_id, winner, reason)
            lg.info(f"Experiment {experiment_id}: Winner: {winner} - {reason}")
            break

    else:
        lg.info(f"Experiment {experiment_id}: Game ended without a winner")
        save_winner_to_csv(experiment_id, "No winner", "No conclusion")


def run_experiments(
        model_a: str,
        model_b: str,
        num_experiments: int = 100,
        max_turns: int = 200,
        csv_file: str = './data/game_results.csv'
) -> None:
    """Runs multiple experiments between two models.

    Args:
        model_a: The name of model A.
        model_b: The name of model B.
        num_experiments: The number of experiments to run.
        max_turns: The maximum number of turns in each game.
        csv_file: The file to save the results.
    """
    fieldnames = ['experiment_number', 'winner', 'reason']

    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    for experiment in range(1, num_experiments + 1):
        csv_time = os.path.basename(os.path.dirname(csv_file))
        log_path = os.path.join("logs", csv_time, f"experiment-{experiment}.log")
        lg.add(log_path)
        lg.info(f"Running experiment {experiment}")
        lg.debug(f"Models: {model_a} vs {model_b}")
        lg.debug(f"Max turns: {max_turns}")
        lg.debug(f"CSV file: {csv_file}")

        run_experiment(
            experiment_id=experiment,
            model_a=model_a,
            model_b=model_b,
            max_turns=max_turns,
            csv_file=csv_file,
            fieldnames=fieldnames
        )

    print("END OF ALL GAMES REACHED")


if __name__ == '__main__':
    # DEFAULT_MODEL_A = "llama3:8b-instruct-fp16
    # DEFAULT_MODEL_B = "llama3:8b-instruct-fp16"
    # DEFAULT_MODEL_A = "llama3:8b"
    # DEFAULT_MODEL_B = "gemma2:9b"
    # DEFAULT_MODEL_A = "llama3:8b-instruct-fp16"
    # DEFAULT_MODEL_B = "gemma2:9b-instruct-fp16"
    DEFAULT_MODEL_A = "llama3:8b-instruct-q8_0"
    DEFAULT_MODEL_B = "gemma2:9b-instruct-q8_0"
    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--model_a', type=str, help='Model A', default=DEFAULT_MODEL_A)
    parser.add_argument('--model_b', type=str, help='Model B', default=DEFAULT_MODEL_B)
    parser.add_argument('--num_experiments', type=int, help='Number of experiments', default=100)
    parser.add_argument('--max_turns', type=int, help='Max turns per experiment', default=200)

    # include unix timestamp in the csv file name
    csv_filepath = f'data/{int(time.time())}/game_results.csv'

    run_experiments(**vars(parser.parse_args()), csv_file=csv_filepath)
