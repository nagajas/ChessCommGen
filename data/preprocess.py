import chess
import chess.engine
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Initialize Stockfish engine
stockfish_path = "../engine/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
print("Engine loaded")

# Read the game commentary file
with open("./scrap/saved_files/game_commentary.txt", "r") as f:
    lines = f.readlines()

def determine_phase(board):
    moves = len(board.move_stack)
    if moves <= 15:
        return "Opening"
    elif len(list(board.legal_moves)) <= 10 or board.is_endgame():
        return "Endgame"
    else:
        return "Middlegame"

# Initialize data storage
data = []
games = 0
start = 0

print("Analyzing games...")

try:
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("1.") and not line.startswith("1..."): 
            games += 1
            board = chess.Board()
            history = []

        game_segments = line.strip().split("<move>")
        
        for gc in tqdm(game_segments, desc=f"Game {games}", position=0, leave=True):
            moves_commentary = gc.split("<sep>")
            if len(moves_commentary) == 1:
                continue

            pgn_moves, commentary = moves_commentary[0].strip(), moves_commentary[1].strip()
            move_list = pgn_moves.split(" ")
            eval_before, eval_after, delta_eval, top_k_best_moves = None, None, None, []

            for idx, move in enumerate(move_list):
                if move[0] in "0123456789": 
                    continue
                try:
                    # Push the move and track history
                    chess_move = board.push_san(move)
                    history.append(move)

                    if idx == len(move_list) - 1: 
                        start +=1
                        if start < 10000:
                            continue
                        # Evaluate before and after the move
                        if len(board.move_stack) > 1:
                            board.pop()  
                            try:
                                eval_before = engine.analyse(board, chess.engine.Limit(time=0.5))["score"].relative
                                eval_before = eval_before.score() if not eval_before.is_mate() else "Mate"
                            except Exception as e:
                                eval_before = None
                            board.push(chess_move)

                        try:
                            eval_after = engine.analyse(board, chess.engine.Limit(time=0.5))["score"].relative
                            eval_after = eval_after.score() if not eval_after.is_mate() else "Mate"
                        except Exception as e:
                            eval_after = None

                        # Change in evaluation
                        if isinstance(eval_before, int) and isinstance(eval_after, int):
                            delta_eval = eval_after - eval_before

                        # Top K best moves
                        try:
                            analysis_result = engine.analyse(board, chess.engine.Limit(time=0.5))
                            if "pv" in analysis_result:
                                best_moves = analysis_result["pv"][:3]
                                top_k_best_moves = [move.uci() for move in best_moves]
                            else:
                                top_k_best_moves = []
                        except Exception as e:
                            top_k_best_moves = []

                        # Append data
                        data.append({
                            "Move": move,
                            "History (PGN)": " ".join(history[:-1]),
                            "Commentary": commentary,
                            "Eval Before": eval_before,
                            "Eval After": eval_after,
                            "Delta": delta_eval,
                            "Top K Best Moves": ", ".join(top_k_best_moves)
                        })

                        # Stop if we reach 10,000 entries
                        if len(data) >= 30000:
                            raise StopIteration

                except ValueError:
                    # Ignore invalid moves and continue
                    break

except StopIteration:
    pass
    #print("Reached 10,000 entries. Stopping analysis.")

# Finalizing
print("Number of games analyzed: ", games)
engine.quit()

# Create DataFrame
df = pd.DataFrame(data)

# Split into train, validation, and test sets
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save to CSV
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Files generated: train.csv, val.csv, test.csv")
