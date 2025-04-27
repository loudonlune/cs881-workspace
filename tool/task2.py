
from curses.ascii import isalnum
from tabnanny import check

import jinja2
import nltk
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity, masi_distance, jaccard_distance, edit_distance
from sentence_transformers import SentenceTransformer
import torch

nltk.download('wordnet')

import numpy
import os
import pandas
import datasets
from typing import Any, Iterable, Optional, List, Tuple, Callable

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

class CategorizeData(object):
    backend: LLMBackend
    data: CheckThatTask2Data
    OUTPUT_COLUMN_EMPTY: str = '---'
    _cache_path: os.PathLike

    category_df: Optional[pandas.DataFrame] = None

    def __init__(self, backend, repository_path: os.PathLike = os.path.join(os.curdir, "checkthat_data"), cache_path: os.PathLike = os.path.join(os.curdir, '.checkthat_cache')):
        
        if not os.path.isdir(repository_path):
            print("ERROR: You must clone the CheckThat! data repository first! Use the provided makefile.")
            raise FileNotFoundError(repository_path)
        self.data = CheckThatTask2Data(repository_path)
        if not os.path.isdir(cache_path):
                try:
                    os.mkdir(cache_path)
                except Exception as e:
                    print("ERROR: Failed to create cache directory.")
                    raise e
        self.cat_file: os.PathLike = os.path.join(self._cache_path, f"category.csv")

    def delete_cat_table_file(self):
        if os.path.isfile(self.cat_file):
            os.remove(self.cat_file)
        else:
            print("Note: cat file did not exist. Didn't change anything...")   
    def initialize_cat_table(self):
            if not os.path.isfile(self.cat_file):
                queries = []
                results = []

                for _, row in self.data.dev_ds.iterrows():
                    queries.append(CheckThatTask2.OUTPUT_COLUMN_EMPTY)
                    results.append(row['category'])

                self.eval_frame = pandas.DataFrame({"input": queries, "output": results})
                print(f'init: Created new eval sheet with {len(queries)} test prompts.')
                self.save_cat_table()
            else:
                print('init: Eval sheet already exists.')
                self.cat_frame = pandas.read_csv(self.cat_file)

    def save_cat_table(self):
        self.cat_frame[['input', 'output',]].to_csv(self.cat_file)

    def delete_cat_table_file(self):
        if os.path.isfile(self.eval_file):
            os.remove(self.eval_file)
        else:
            print("Note: Eval file did not exist. Didn't change anything...")
    def fill_cat_table(self):
        rows_to_fill = []        

        for key, row in self.cat_frame.iterrows():
            opt_value: float | str = row['output']

            # Empty cells in pandas data frames are a bit of a headache.
            if type(opt_value) is float or (type(opt_value) is str and (opt_value in ["", CheckThatTask2.OUTPUT_COLUMN_EMPTY])):
                rows_to_fill.append(key)

        print(f'fill-eval-table: Querying {len(rows_to_fill)}. This may take a long time.')
        count = 0
        for c in rows_to_fill:
            row: pandas.Series = self.eval_frame.loc[c]
            query_result = self.backend.query(row['input'])
            query_result = query_result.split(":")
            len_query = len(query_result)
            self.eval_frame.at[c, 'output'] = query_result[len_query-1]
            count += 1
            print(f'{rows_to_fill - count} rows remaining')

            # Save the file each time we get a response to memoize them eagerly.
            self.save_cat_table()       

class MetricStatistics(object):
    metric: str
    entries: List[float]
    average: float
    dist: Tuple[float, float, float, float, float]

    def __init__(self,
                 # Name of metric.
                 metric_name: str,
                 # Eval frame.
                 frame: pandas.DataFrame,
                 # Function to tokenize source and test.
                 tok_fn: Callable[[str, str], Tuple[Any, Any]],
                 # Function to calculate metric from tokenized value.
                 fun: Callable[[Any, Any], float]):
        # Compute the metric.
        print(f"MetricStatistics ctor: Computing metric {metric_name}...")

        metrics = [
            fun(*tok_fn(row['reference'], row['output']))
            for _, row in frame.iterrows()
            if type(row['output']) is str and row['output'] != CheckThatTask2.OUTPUT_COLUMN_EMPTY
        ]

        self.completeness = round((float(len(metrics)) / float(len(frame))) * 100.0, 2)
        self.entries = metrics
        self.metric = metric_name
        self.average = round(numpy.average(metrics), 4)
        self.dist = (
            round(numpy.min(metrics), 4),
            round(numpy.percentile(metrics, 25.0), 4),
            round(numpy.percentile(metrics, 50.0), 4),
            round(numpy.percentile(metrics, 75.0), 4),
            round(numpy.max(metrics), 4)
        )
        print("MetricStatistics ctor: Complete!")

    def print(self):
        head = "=" * 20, f"Metric: {self.metric} (Completeness: {self.completeness}%)", "=" * 20
        vmin, lq, median, hq, vmax = self.dist

        print(head)

        print(f"Average: {self.average}")
        print(f"Minimum:        {vmin}")
        print(f"Lower Quartile: {lq}")
        print(f"Median:         {median}")
        print(f"Upper Quartile: {hq}")
        print(f"Maximum:        {vmax}")

        print("=" * len(head))

class CheckThatTask2(object):
    """
    Little class to manage state for running the CheckThat! task 2 stuff.
    """
    prompt_template: Optional[str] = None
    eval_frame: Optional[pandas.DataFrame] = None
    train_data: Optional[datasets.Dataset] = None
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

    def delete_eval_table_file(self):
        if os.path.isfile(self.eval_file):
            os.remove(self.eval_file)
        else:
            print("Note: Eval file did not exist. Didn't change anything...")
    
    def initialize_train_data_from_train_ds(self):
        """
        Create training dataset from the given train dataset.
        """
        queries = []
        results = []

        for _, row in self.data.train_ds.iterrows():
            queries.append(
                self.query_tmpl.render(input=row['post'])
            )
            results.append(
                row['normalized claim']
            )
        
        self.train_data = datasets.Dataset.from_dict(
            {
                "input": queries,
                "output": results,
            }
        )

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

    def train(self, cap: Optional[int] = None):
        """
        If we wanted to further train/refine an LLM using datasets, this function
        serves as a place to do that.

        Arbitrary kwargs may be passed in for parameterization of this process.
        """
        self.backend.train(self.train_data, cap=cap, **self.trainargs)
    
    def fill_eval_table(self):
        rows_to_fill = []        

        for key, row in self.eval_frame.iterrows():
            opt_value: float | str = row['output']

            # Empty cells in pandas data frames are a bit of a headache.
            if type(opt_value) is float or (type(opt_value) is str and (opt_value in ["", CheckThatTask2.OUTPUT_COLUMN_EMPTY])):
                rows_to_fill.append(key)

        print(f'fill-eval-table: Querying {len(rows_to_fill)}. This may take a long time.')
        count = 0
        for c in rows_to_fill:
            row: pandas.Series = self.eval_frame.loc[c]
            query_result = self.backend.query(row['input'])
            self.eval_frame.at[c, 'output'] = query_result
            count += 1
            print(f'{len(rows_to_fill) - count} rows remaining')

            # Save the file each time we get a response to memoize them eagerly.
            self.save_eval_table()

    def calculate_statistics(self) -> List[MetricStatistics]:
        tokenizer = nltk.tokenize.NLTKWordTokenizer()
        sentence_encoder: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        def tokenize_custom(input: str) -> Iterable[str]:
            return filter(lambda tok: all(map(lambda c: isalnum(c), tok)), tokenizer.tokenize(input))
        
        def nltk_tokenize(ref: str, test: str) -> Tuple[List[str], List[str]]:
            return (tokenize_custom(ref), tokenize_custom(test))

        def calculate_meteor_score(ref: List[str], test: List[str]) -> float:
            return nltk.meteor(references=[ref], hypothesis=test)
        
        def jaccard(ref: List[str], test: List[str]) -> float:
            return jaccard_distance(label1=set(ref), label2=set(test))
        
        def masi(ref: List[str], test: List[str]) -> float:
            return masi_distance(label1=set(ref), label2=set(test))

        def sentence_encode(ref: str, test: str) -> Tuple[torch.Tensor, torch.Tensor]:
            reft, testt = sentence_encoder.encode([ref, test], convert_to_tensor=True)
            return (reft, testt)
        
        def cosine_similarity(ref: torch.Tensor, test: torch.Tensor) -> float:
            return torch.nn.functional.cosine_similarity(ref, test, dim=0).item()

        metrics = [
            ("METEOR",              nltk_tokenize, calculate_meteor_score),
            ("Jaccard Distance",    nltk_tokenize, jaccard),
            ("Masi Distance",       nltk_tokenize, masi),
            ("Cosine Similarity of Sentence Embeddings", sentence_encode, cosine_similarity)
        ]

        return [
            MetricStatistics(name, self.eval_frame, tfn, sfn)
            for name, tfn, sfn in metrics
        ]

