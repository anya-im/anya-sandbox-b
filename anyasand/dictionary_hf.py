import numpy as np
import torch
import os
import sqlite3
import csv
import struct
import yaml
import jaconv
import argparse
import logging
import datetime
from tqdm import tqdm
from logging import basicConfig, getLogger, INFO
from transformers import AutoModel, AutoTokenizer
from bitnet import replace_linears_in_hf
from anyasand.dictionary import Dictionary

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter,)


class DictionaryHf(Dictionary):
    def __init__(self, db_path, pre_trained_model="line-corporation/line-distilbert-base-japanese"):
        super().__init__(db_path)
        self._tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(pre_trained_model)
        replace_linears_in_hf(self._hf_model)

        self._vec_size = 768

    def get_sword(self, wid):
        return np.concatenate([self._vector(str(wid)), self.vec_eye(wid)])

    def _vector(self, idx):
        with torch.no_grad():
            token = self._tokenizer(self._words[idx]["name"], return_tensors="pt")
            token_len = len(token["input_ids"].squeeze()) - 1
            outputs = self._hf_model(**token)
            vec = torch.zeros_like(outputs["last_hidden_state"].squeeze()[0]).detach()
            for i in range(token_len):
                if i == 0:
                    continue
                vec += outputs["last_hidden_state"].squeeze()[i]
            vec = torch.where(vec > 0., 1., 0.)
        return vec.numpy()


def _count_csv_rows(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count


class DicBuilder:
    _dic_csv = [
        #"small_lex.csv",
        #"core_lex.csv",
        #"notcore_lex.csv"
        "sudachi_dict.csv"
    ]

    def __init__(self, db_path="./anya-dic.db", sudachidic_path="./sudachidic",
                 pre_trained_model="line-corporation/line-distilbert-base-japanese", word_vec_size=768):
        self._tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(pre_trained_model)
        replace_linears_in_hf(self._hf_model)

        self._db_path = db_path
        self._sudachidic_path = sudachidic_path
        self._word_vec_size = word_vec_size

        self._logger = getLogger("ANYA-DicBuilder")

    def __call__(self):
        # db initialize
        if os.path.isfile(self._db_path):
            os.remove(self._db_path)

        pos_id_def = os.path.dirname(__file__) + "/data/sudachidic.yml"
        with open(pos_id_def, 'r') as f:
            pos_ids = yaml.safe_load(f)["pos-ids"]

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        cur.execute("CREATE TABLE positions(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);")
        for pos_name in pos_ids:
            cur.execute('INSERT INTO positions(name) values(?);', (pos_name,))

        cur.execute("""
            CREATE TABLE words(
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT, 
            read TEXT, 
            pos INTEGER,
            cost INTEGER,
            vec BLOB);
            """)

        cur.execute("""
            CREATE TABLE vector(
            id INTEGER, 
            vec BLOB);
            """)

        # add BOS
        print(datetime.datetime.today())
        print(" add BOS")
        cur.execute("SELECT id FROM positions WHERE name = ?", ("BOS.*.*.*",))
        pos_id = cur.fetchone()[0]
        vec_np = self._bos_vector()
        vec = struct.pack('=%df' % vec_np.size, *vec_np)
        cur.execute("INSERT INTO words(name, read, pos, cost, vec) values(?, ?, ?, ?, ?);",
                    ("_BOS", "_BOS", pos_id, 0., vec))

        for csv_file in self._dic_csv:
            file_path = self._sudachidic_path + "/" + csv_file
            print(datetime.datetime.today())
            print(" reading... %s" % csv_file)

            row_count = _count_csv_rows(file_path)
            bar = tqdm(total=row_count)
            print(row_count)
            with open(self._sudachidic_path + "/" + csv_file) as f:
                reader = csv.reader(f, delimiter=",")
                try:
                    for row in reader:
                        pos_name = row[5] + "." + row[6] + "." + row[7] + "." + row[8]
                        cur.execute("SELECT id FROM positions WHERE name = ?", (pos_name,))
                        pos_id = cur.fetchone()[0]
                        vec_np = self._vector(row[0])
                        #vec_np = np.zeros(self._word_vec_size, dtype=float)
                        vec = struct.pack('=%df' % vec_np.size, *vec_np)
                        cur.execute("INSERT INTO words(name, read, pos, cost, vec) values(?, ?, ?, ?, ?);",
                                    (row[0], jaconv.kata2hira(row[11]), pos_id, 0, vec))
                        bar.update(1)
                except UnicodeDecodeError:
                    pass

        print(datetime.datetime.today())
        print("end!!")

        cur.execute('CREATE INDEX words_idx ON words(id, name, read);')
        cur.execute('CREATE INDEX vec_idx ON vector(id);')
        cur.execute('commit;')
        cur.close()
        conn.close()

    def _vector(self, in_str):
        with torch.no_grad():
            token = self._tokenizer(in_str, return_tensors="pt")
            token_len = len(token["input_ids"].squeeze()) - 1
            outputs = self._hf_model(**token)
            vec = torch.zeros_like(outputs["last_hidden_state"].squeeze()[0]).detach()
            for i in range(token_len):
                if i == 0:
                    continue
                vec += outputs["last_hidden_state"].squeeze()[i]
            vec = torch.where(vec > 0., 1., 0.).half()
        return vec.numpy()

    def _bos_vector(self):
        with torch.no_grad():
            token = self._tokenizer("は", return_tensors="pt")
            outputs = self._hf_model(**token)
            vec = outputs["last_hidden_state"].squeeze()[0]
            vec = torch.where(vec > 0., 1., 0.).half()
        return vec.numpy()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--db_path', help='dictionary DB path', default="./anya-dic.db")
    args = arg_parser.parse_args()
    builder = DicBuilder(args.db_path)
    builder()


if __name__ == "__main__":
    main()
