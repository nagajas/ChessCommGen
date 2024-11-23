import chess
import chess.engine
import pandas as pd
from sklearn.model_selection import train_test_split

stockfish_path = "/home/hari/Desktop/Chess commentary/LoveDaleNLP/data"

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

with open("commentary.txt", "r") as f:
    lines = f.readlines()

def determine_phase(board):
    moves = len(board.move_stack)
    if moves <= 15:
        return "Opening"
    elif len(list(board.legal_moves)) <= 10 or board.is_endgame():
        return "Endgame"
    else:
        return "Middlegame"

data = []
for line in lines:
    moves_commentary = line.strip().split("<sep>")
    pgn_moves, commentary = moves_commentary[0].strip(), moves_commentary[1].strip()

    board = chess.Board()
    move_list = pgn_moves.split()
    history = []
    eval_before, eval_after, delta_eval, top_k_best_moves = None, None, None, []

    for move in move_list:
        try:
            chess_move = board.push_san(move)
            history.append(move)
        except ValueError:
            print(f"Invalid move: {move}")
            break

    if len(board.move_stack) > 1:
        board.pop()
        eval_before = engine.analyse(board, chess.engine.Limit(time=0.5))["score"].relative
        eval_before = eval_before.score() if not eval_before.is_mate() else "Mate"
        board.push(chess_move)

    eval_after = engine.analyse(board, chess.engine.Limit(time=0.5))["score"].relative
    eval_after = eval_after.score() if not eval_after.is_mate() else "Mate"

    if isinstance(eval_before, int) and isinstance(eval_after, int):
        delta_eval = eval_after - eval_before

    best_moves = engine.analyse(board, chess.engine.Limit(time=0.5))["pv"][:5]
    top_k_best_moves = [str(move) for move in best_moves]

    phase = determine_phase(board)

    data.append({
        "Move": move_list[-1],
        "History (PGN)": " ".join(history),
        "Commentary": commentary,
        "Eval Before": eval_before,
        "Eval After": eval_after,
        "Delta": delta_eval,
        "Top K Best Moves": ", ".join(top_k_best_moves),
        "Phase": phase
    })

df = pd.DataFrame(data)

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

engine.quit()

print("Files generated: train.csv, val.csv, test.csv")
