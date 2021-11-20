# # chess stuff
import os, random
import chess
import chess.pgn
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers

class super_simple_bot:

    def __init__(self, username, games_dir):
        self.username = username
        self.game_filenames = []
        for filename in os.listdir(games_dir):
            self.game_filenames.append(games_dir+filename)

    def find_a_random_move(self, board):
        legal_moves = []
        for move in board.legal_moves:
            legal_moves.append(move)
        return legal_moves[random.randrange(len(legal_moves))]

    def playing_as(self, game):
        if game.headers["Black"] == self.username:
            return chess.BLACK
        else:
            return chess.WHITE

    def find_my_most_played_move(self, position):
        position_fen = position.fen()
        candidate_move_dict = {}

        for file in self.game_filenames:
            pgn = open(file)
            game = chess.pgn.read_game(pgn)
            while game is not None:
                my_colour = self.playing_as(game)
                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == my_colour:
                        if board.fen() == position_fen:
                            if move in candidate_move_dict:
                                candidate_move_dict[move] += 1
                            else:
                                candidate_move_dict[move] = 1
                    board.push(move)
                game = chess.pgn.read_game(pgn)
            pgn.close()
        if candidate_move_dict:
            chosen_move = max(candidate_move_dict, key = candidate_move_dict.get)
            return chosen_move
        else:
            return None

class smart_bot(super_simple_bot):

    def __init__(self, username, games_dir):
        self.username = username
        self.game_filenames = []
        for filename in os.listdir(games_dir):
            self.game_filenames.append(games_dir+filename)
        self.model = None

    def get_board_state(self, board):
        state = []
        for square in chess.SQUARES:
            state.append(str(board.piece_at(square)))
        return state

    def get_board_masks(self, move):
        from_mask = np.zeros(64)
        to_mask = np.zeros(64)
        from_mask[move.from_square] = 1
        to_mask[move.to_square]=1
        return from_mask, to_mask

    def get_feature_names(self, target_name="my_move"):
        move_from_square_names = ['from_' + square for square in chess.SQUARE_NAMES]
        move_to_square_names = ['to_' + square for square in chess.SQUARE_NAMES]
        return chess.SQUARE_NAMES + move_from_square_names + move_to_square_names + list(["colour_to_move"]) + list([target_name])

    def save_moves(self, game_data, filename, append=False):
        if append:
            df = pd.DataFrame(data = game_data)
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df = pd.DataFrame(data = [], columns = self.get_feature_names())
            df.to_csv(filename, index=False)

    def label_my_played_games(self, labelled_moves_filename):
        if os.path.isfile(labelled_moves_filename):
            print("Label file exists - delete " + labelled_moves_filename + " if you want the games relabelled.")
            return
        self.save_moves(None, labelled_moves_filename, append=False)

        for file in self.game_filenames:
            pgn = open(file)
            print(file)
            game = chess.pgn.read_game(pgn)
            while game is not None:
                game_data = []
                my_colour = self.playing_as(game)
                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == my_colour:
                        board_state = self.get_board_state(board)
                        for legal_move in board.legal_moves:
                            from_mask, to_mask = self.get_board_masks(legal_move)
                            game_data.append(np.concatenate((board_state, from_mask, to_mask, list([board.turn]), list([legal_move==move]))))
                    board.push(move)
                self.save_moves(game_data, labelled_moves_filename, append=True)
                game = chess.pgn.read_game(pgn)
                        
            pgn.close()
    
    def load_model(self, model_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded.")

    def train_model(self, labelled_moves_filename, model_filename, nrows=None):
        if os.path.isfile(model_filename+"/saved_model.pb"):
            print("Model exists")
            self.load_model(model_filename)
            # os.remove(model_filename)
            return

        dataframe = pd.read_csv(labelled_moves_filename, nrows = nrows)
        train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

        def df_to_dataset(dataframe, shuffle=True, batch_size=32):
            df = dataframe.copy()
            labels = df.pop('my_move')
            df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
            ds = tf.data.Dataset.from_tensor_slices((dict(df),labels))
            if shuffle:
                ds= ds.shuffle(buffer_size = len(dataframe))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
            return ds

        def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
            if dtype == "string":
                index = layers.StringLookup(max_tokens=max_tokens)
            else:
                index = layers.IntegerLookup(max_tokens=max_tokens)

            feature_ds = dataset.map(lambda x, y:x[name])
            
            index.adapt(feature_ds)

            encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

            return lambda feature: encoder(index(feature))

        batch_size = 256
        train_ds = df_to_dataset(train, batch_size=batch_size)
        val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
        test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

        all_inputs = []
        encoded_features = []

        [(headers, label_batch)] = train_ds.take(1)

        for header in list(headers.keys())[:64]:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
            encoded_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype="string", max_tokens=5)
            encoded_categorical_col = encoded_layer(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)

        for header in list(headers.keys())[64:-1]:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype="float64")
            encoded_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype="float64", max_tokens=5)
            encoded_categorical_col = encoded_layer(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)

        all_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        output = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(all_inputs, output)
        self.model.compile(optimizer="adam", loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
        self.model.fit(train_ds, epochs=10, validation_data=val_ds)

        loss, accuracy = self.model.evaluate(test_ds)
        print("Accuracy", accuracy)
        self.model.save(model_filename)

    def predict_move(self, board):
        if self.model == None:
            print("Load a model first.")
            return

        board_state = self.get_board_state(board)
        feature_names = self.get_feature_names()[:-1]

        from_mask = []
        to_mask = []
        board_states = []
        colour_to_move = []

        for i in range(len(board_state)):
            from_mask.append([])
            to_mask.append([])
            board_states.append([])

        for move in board.legal_moves:
            from_, to_ = self.get_board_masks(move)
            colour_to_move.append(board.turn)
            for i in range(len(from_)):
                from_mask[i].append(from_[i])
                to_mask[i].append(to_[i])
                board_states[i].append(board_state[i])

        predict_dict = {}
        for i in range(len(board_state)):
            predict_dict[feature_names[i]] = tf.convert_to_tensor(board_states[i])

        for i in range(len(board_state), len(board_state) + len(from_mask)):
            predict_dict[feature_names[i]] = tf.convert_to_tensor(from_mask[i - len(board_state)])

        for i in range(len(board_state) + len(from_mask), len(board_state) + len(from_mask) + len(to_mask)):
            predict_dict[feature_names[i]] = tf.convert_to_tensor(to_mask[i - len(board_state) - len(from_mask)])

        predict_dict["colour_to_move"] = tf.convert_to_tensor(colour_to_move)

        predictions = self.model.predict(predict_dict)

        predictions_dict = {}
        for i in range(len(predictions)):
            predictions_dict[list(board.legal_moves)[i]] = tf.nn.sigmoid(predictions[i])

        return max(predictions_dict, key = predictions_dict.get)

chessbot = smart_bot('dylanjsw', 'played_games/')
chessbot.load_model("chessbot_model")

# web stuff
from http.server import BaseHTTPRequestHandler, HTTPServer

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        web_fen = self.rfile.read(content_length).decode('utf-8')
        web_board = chess.Board(fen=web_fen)
        web_board.push(chessbot.predict_move(web_board))

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(web_board.fen().encode("utf-8"))
    
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

with HTTPServer(('', 8000), handler) as server:
    print("Server is up")
    server.serve_forever()