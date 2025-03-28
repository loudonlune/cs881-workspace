
from curses.ascii import isalnum

import jinja2
import nltk
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity, masi_distance, jaccard_distance, edit_distance

nltk.download('wordnet')

import numpy
import os
import pandas
from typing import Iterable

from tool.llm.base import LLMBackend

class CheckThatTask2Data(object):
    dev_ds: pandas.DataFrame
    test_ds: pandas.DataFrame
    train_ds: pandas.DataFrame

    def __init__(self, repository: os.PathLike):
        base_path: os.PathLike = os.path.join(repository, "task2", "data")

        self.dev_ds = pandas.read_csv(os.path.join(base_path, "dev", "dev-eng.csv"))
        self.test_ds = pandas.read_csv(os.path.join(base_path, "test", "test-eng.csv"))
        self.train_ds = pandas.read_csv(os.path.join(base_path, "train", "train-eng.csv"))

class CheckThatTask2(object):
    """
    Little class to manage state for running the CheckThat! task 2 stuff.
    """
    prompt_template: str | None = None
    eval_frame: pandas.DataFrame | None = None
    data: CheckThatTask2Data
    backend: LLMBackend
    query_tmpl: jinja2.Template
    profile_name: str
    OUTPUT_COLUMN_EMPTY: str = '---'
    trainargs: dict = {}
    _cache_path: os.PathLike
    
    def __init__(self, profile_name: str, backend: LLMBackend, query_tmpl: jinja2.Template, repository_path: os.PathLike = os.path.join(os.curdir, "checkthat_data"), cache_path: os.PathLike = os.path.join(os.curdir, '.checkthat_cache')):
        # Fail here if the path to the CheckThat! data repository is not found.
        if not os.path.isdir(repository_path):
            print("ERROR: You must clone the CheckThat! data repository first! Use the provided makefile.")
            raise FileNotFoundError(repository_path)
        
        # Read the data from CSV files in the Git repository.
        self.data = CheckThatTask2Data(repository_path)
        
        if not os.path.isdir(cache_path):
            try:
                os.mkdir(cache_path)
            except Exception as e:
                print("ERROR: Failed to create cache directory.")
                raise e
        
        # Set paths.
        self._cache_path = cache_path
        self.profile_name = profile_name
        self.backend = backend
        self.query_tmpl = query_tmpl

        # Determine where in the cache directory to store the eval table.
        self.eval_file: os.PathLike = os.path.join(self._cache_path, f"{self.profile_name}-eval.csv")

        # Initialize backend.
        self.backend.initialize()

    def delete_eval_table_file(self):
        if os.path.isfile(self.eval_file):
            os.remove(self.eval_file)
        else:
            print("Note: Eval file did not exist. Didn't change anything...")
    
    def initialize_eval_table(self):
        if not os.path.isfile(self.eval_file):
            queries = []
            results = []
            refopts = []

            for _, row in self.data.dev_ds.iterrows():
                queries.append(self.query_tmpl.render(
                    train_rows=self.data.train_ds,
                    input=row['post'],
                ))
                results.append(CheckThatTask2.OUTPUT_COLUMN_EMPTY)
                refopts.append(row['normalized claim'])

            self.eval_frame = pandas.DataFrame({"input": queries, "output": results, "reference": refopts})
            print(f'init: Created new eval sheet with {len(queries)} test prompts.')
            self.save_eval_table()
        else:
            print('init: Eval sheet already exists.')
            self.eval_frame = pandas.read_csv(self.eval_file)

    def save_eval_table(self):
        self.eval_frame[['input', 'output', 'reference']].to_csv(self.eval_file)

    def train(self):
        """
        If we wanted to further train/refine an LLM using datasets, this function
        serves as a place to do that.

        Arbitrary kwargs may be passed in for parameterization of this process.
        """
        self.backend.train(self.data, **self.trainargs)
    
    def fill_eval_table(self):
        rows_to_fill = []        

        for key, row in self.eval_frame.iterrows():
            opt_value: float | str = row['output']

            # Empty cells in pandas data frames are a bit of a headache.
            if type(opt_value) is float or (type(opt_value) is str and (opt_value in ["", CheckThatTask2.OUTPUT_COLUMN_EMPTY])):
                rows_to_fill.append(key)

        print(f'fill-eval-table: Querying {len(rows_to_fill)}. This may take a long time.')        
        for c in rows_to_fill:
            row: pandas.Series = self.eval_frame.loc[c]
            query_result = self.backend.query(row['input'])
            self.eval_frame.at[c, 'output'] = query_result

            # Save the file each time we get a response to memoize them eagerly.
            self.save_eval_table()

    def calculate_meteor_score_avg(self) -> float:
        tokenizer = nltk.tokenize.NLTKWordTokenizer()
        def tokenize_custom(input: str) -> Iterable[str]:
            return filter(lambda tok: all(map(lambda c: isalnum(c), tok)), tokenizer.tokenize(input))

        meteors = [
            nltk.meteor(references=[tokenize_custom(row['reference'])], hypothesis=tokenize_custom(row['output']))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str
        ]

        print(f"calculate_meteor_score_avg: Calculated individual meteor score for {len(meteors)} rows.")

        if len(meteors) > 0:
            return round(numpy.average(meteors), 4)
        else:
            return 0.0
    def interval_distances(self):
        tokenizer = nltk.tokenize.NLTKWordTokenizer()
        def tokenize_custom(input: str) -> Iterable[str]:
            return filter(lambda tok: all(map(lambda c: isalnum(c), tok)), tokenizer.tokenize(input))
        jaccrad_d = [
            jaccard_distance(label1=[tokenize_custom(row["reference"])], label2=tokenize_custom(row["output"]))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str]
        jaro = [
            jaro_similarity(s1=[tokenize_custom(row["reference"])], s2=tokenize_custom(row["output"]))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str
            ]
        jaro_winlket = [
            jaro_winkler_similarity(s1=[tokenize_custom(row["reference"])], s2=tokenize_custom(row["output"]))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str
            ]
        masi_dist =[
            masi_distance(label1=[tokenize_custom(row["reference"])], label2=tokenize_custom(row["output"]))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str
            ]
        edit_d = [
            edit_distance(s1=[tokenize_custom(row["reference"])], s2=tokenize_custom(row["output"]))
            for _, row in self.eval_frame.iterrows()
            if type(row['output']) is str
        ]
        avg_jaccard_distance = round(numpy.average(jaccrad_d))
        avg_jaro_distance = round(numpy.average(jaro))
        avg_jaro_winklet = round(numpy.average(jaro_winlket))
        avg_masi_distance = round(numpy.average(masi_dist))
        avg_edit_d = round(numpy.average(edit_d))
        return avg_jaccard_distance, avg_jaro_distance, avg_jaro_winklet, avg_masi_distance, avg_edit_d

