from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter, defaultdict
from nltk import ngrams
import numpy as np
import pandas as pd
import re


class NGramClassifier(BaseEstimator, ClassifierMixin):
    """
    Character n-gram classifier.

    Parameters
    ----------
    n : int, default=3
        Length of the character n-grams.
    top_k : int, default=300
        Number of top n-grams to keep for each class.
    distance : str, default='rank'
        Distance metric: 'rank', 'cosine_tfidf', or 'match_rate'.
    pad_char : str, default='^'
        Character to prepend to each word.
    end_char : str, default='$'
        Character to append to each word.

    Attributes
    ----------
    encoder_ : LabelEncoder
        Fitted label encoder.
    vocabulary_ : list[str]
        Sorted list of n-grams across all classes.
    class_profiles_ : dict[str, list[str]]
        Top n-grams per class.
    class_rank_vectors_ : dict[str, np.ndarray]
        Rank vectors for each class (if distance='rank').
    vectorizer_ : TfidfVectorizer
        Fitted TF-IDF vectorizer (if distance='cosine_tfidf').
    class_tfidf_vectors_ : dict[str, np.ndarray]
        Mean TF-IDF vectors per class.
    class_ngram_sets_ : dict[str, set[str]]
        N-gram sets per class (if distance='match_rate').

    Methods
    -------
    fit(X, y)
        Fit the classifier to data.
    predict(X)
        Predict class labels for X.
    predict_proba(X)
        Predict class probabilities for X.
    _extract(word)
        Extract n-grams from a word.
    _name_to_rank_vector(name)
        Convert a name to its rank vector.
    _name_to_freq_vector(name, normalize)
        Convert a name to its frequency vector.
    _compute_distances(X)
        Compute distance matrix between X and class profiles.
    """

    def __init__(
        self,
        n: int = 3,
        top_k: int = 300,
        distance: str = 'rank',
        pad_char: str = '^',
        end_char: str = '$'
    ) -> None:
        self.n = n
        self.top_k = top_k
        self.distance = distance
        self.pad_char = pad_char
        self.end_char = end_char
        self.min_match_rate_ngrams = 1

    def fit(self, X: list[str], y: list[str]) -> "NGramClassifier":
        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        classes = self.encoder_.classes_

        self.class_profiles_ = {}
        ngram_set = set()
        self.class_words_ = {label: [] for label in classes}

        for class_label in classes:
            class_words = [w for w, lbl in zip(X, y) if lbl == class_label]
            counter = Counter()
            for word in class_words:
                counter.update(self._extract(word))
            top_ngrams = [ng for ng, _ in counter.most_common(self.top_k)]
            self.class_profiles_[class_label] = top_ngrams
            ngram_set.update(top_ngrams)

        self.vocabulary_ = sorted(ngram_set)
        self.vocab_index_ = {ng: i for i, ng in enumerate(self.vocabulary_)}
        vocab_size = len(self.vocabulary_)

        if self.distance == 'rank':
            self.class_rank_vectors_ = {}
            for cls, ngrams_ in self.class_profiles_.items():
                vec = np.full(vocab_size, self.top_k, dtype=int)
                for i, ng in enumerate(ngrams_):
                    idx = self.vocab_index_.get(ng)
                    if idx is not None:
                        vec[idx] = i
                self.class_rank_vectors_[cls] = vec

        elif self.distance == 'cosine_tfidf':
            self.vectorizer_ = TfidfVectorizer(
                analyzer='char',
                ngram_range=(self.n, self.n),
                vocabulary=self.vocabulary_
            )
            formatted = [f"{self.pad_char}{w.lower()}{self.end_char}" for w in X]
            tfidf_matrix = self.vectorizer_.fit_transform(formatted)
            self.class_tfidf_vectors_ = {}
            for cls in classes:
                idxs = [i for i, lbl in enumerate(y) if lbl == cls]
                cls_vecs = tfidf_matrix[idxs]
                self.class_tfidf_vectors_[cls] = np.asarray(cls_vecs.mean(axis=0)).ravel()

        elif self.distance == 'match_rate':
            self.class_ngram_sets_ = {
                cls: set(self.class_profiles_[cls]) for cls in classes
            }

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return self

    def predict(self, X: list[str]) -> np.ndarray:
        dists = self._compute_distances(X)
        return self.encoder_.inverse_transform(np.argmin(dists, axis=1))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        dists = self._compute_distances(X)
        inv = 1.0 / (dists + 1e-9)
        return inv / inv.sum(axis=1, keepdims=True)

    def _extract(self, word: str) -> list[str]:
        padded = f"{self.pad_char}{word.lower()}{self.end_char}"
        return ["".join(gram) for gram in ngrams(padded, self.n)]

    def _name_to_rank_vector(self, name: str) -> np.ndarray:
        counter = Counter(self._extract(name))
        top_ngrams = [ng for ng, _ in counter.most_common(self.top_k)]
        vec = np.full(len(self.vocabulary_), self.top_k, dtype=int)
        for i, ng in enumerate(top_ngrams):
            idx = self.vocab_index_.get(ng)
            if idx is not None:
                vec[idx] = i
        return vec

    def _name_to_freq_vector(self, name: str, normalize: bool = True) -> np.ndarray:
        counter = Counter(self._extract(name))
        vec = np.zeros(len(self.vocabulary_), dtype=float)
        for ng, cnt in counter.items():
            idx = self.vocab_index_.get(ng)
            if idx is not None:
                vec[idx] = cnt
        if normalize and vec.sum() > 0:
            vec /= vec.sum()
        return vec

    def _compute_distances(self, X: list[str]) -> np.ndarray:
        if self.distance == 'rank':
            dmat = []
            for name in X:
                nv = self._name_to_rank_vector(name)
                dmat.append([
                    np.sum(np.abs(nv - self.class_rank_vectors_[cls]))
                    for cls in self.encoder_.classes_
                ])
            return np.array(dmat)

        if self.distance == 'cosine_tfidf':
            formatted = [f"{self.pad_char}{name.lower()}{self.end_char}" for name in X]
            name_vecs = self.vectorizer_.transform(formatted)
            class_mat = np.stack(
                [self.class_tfidf_vectors_[cls] for cls in self.encoder_.classes_]
            )
            return cosine_distances(name_vecs, class_mat)

        if self.distance == 'match_rate':
            dmat = []
            for name in X:
                ngs = set(self._extract(name))
                expected = max(1, len(name) - self.n + 1)
                denom = max(expected, len(ngs), self.min_match_rate_ngrams)
                row = []
                for cls in self.encoder_.classes_:
                    match = len(ngs & self.class_ngram_sets_[cls])
                    row.append(1.0 - (match / denom))
                dmat.append(row)
            return np.array(dmat)

        raise ValueError(f"Unknown distance: {self.distance}")


class LIGAClassifier(BaseEstimator, ClassifierMixin):
    """
    LIGA graph-based n-gram language classifier.

    Parameters
    ----------
    n : int, default=3
        Length of character n-grams.
    use_log : bool, default=False
        If True, apply log1p to counts.
    use_median : bool, default=False
        If True, aggregate using median instead of sum.

    Attributes
    ----------
    V : set[str]
        Set of observed n-grams (vertices).
    E : set[tuple[str, str]]
        Set of observed n-gram transitions (edges).
    Wv : dict[str, dict[str, int]]
        Vertex counts per language.
    We : dict[tuple[str, str], dict[str, int]]
        Edge counts per language.
    lang_list_ : list[str]
        Ordered list of languages.

    Methods
    -------
    fit(X, y)
        Build frequency graph from training data.
    predict(X)
        Predict language labels for X.
    predict_proba(X)
        Predict language probabilities for X.
    _extract_ngrams(text)
        Extract character n-grams.
    _transform(value)
        Apply optional log transform.
    _compute_totals()
        Compute total counts per language.
    """

    def __init__(
        self,
        n: int = 3,
        use_log: bool = False,
        use_median: bool = False
    ) -> None:
        self.n = n
        self.use_log = use_log
        self.use_median = use_median
        self.V: set[str] = set()
        self.E: set[tuple[str, str]] = set()
        self.Wv: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.We: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.languages: set[str] = set()

    def fit(self, X: list[str], y: list[str]) -> "LIGAClassifier":
        if len(X) != len(y):
            raise AssertionError("Mismatched input and labels")
        for text, lang in zip(X, y):
            self.languages.add(lang)
            ngs = self._extract_ngrams(text)
            prev = None
            for ng in ngs:
                if ng not in self.V:
                    self.V.add(ng)
                self.Wv[ng][lang] += 1
                if prev is not None:
                    edge = (prev, ng)
                    self.E.add(edge)
                    self.We[edge][lang] += 1
                prev = ng
        self.lang_list_ = sorted(self.languages)
        return self

    def predict(self, X: list[str]) -> list[str]:
        probs = self.predict_proba(X)
        return [self.lang_list_[int(np.argmax(p))] for p in probs]

    def predict_proba(self, X: list[str]) -> list[list[float]]:
        n_langs = len(self.lang_list_)
        node_totals, edge_totals = self._compute_totals()
        results: list[list[float]] = []
        for text in X:
            ngs = self._extract_ngrams(text)
            node_scores = []
            edge_scores = []
            # Node
            for ng in ngs:
                if ng in self.Wv:
                    scores = np.array([
                        (self._transform(self.Wv[ng].get(lang, 0)) / node_totals[i]
                         if node_totals[i] > 0 else 0.0)
                        for i, lang in enumerate(self.lang_list_)
                    ])
                    node_scores.append(scores)
            # Edge
            for i in range(len(ngs) - 1):
                edge = (ngs[i], ngs[i+1])
                if edge in self.We:
                    scores = np.array([
                        (self._transform(self.We[edge].get(lang, 0)) / edge_totals[i]
                         if edge_totals[i] > 0 else 0.0)
                        for i, lang in enumerate(self.lang_list_)
                    ])
                    edge_scores.append(scores)
            if node_scores:
                arr = np.vstack(node_scores).T
                node_stat = np.median(arr, axis=1) if self.use_median else np.sum(arr, axis=1)
            else:
                node_stat = np.zeros(n_langs)
            if edge_scores:
                arr = np.vstack(edge_scores).T
                edge_stat = np.median(arr, axis=1) if self.use_median else np.sum(arr, axis=1)
            else:
                edge_stat = np.zeros(n_langs)
            comb = node_stat + edge_stat
            total = comb.sum()
            probs = (comb / total) if total > 0 else np.ones(n_langs) / n_langs
            results.append(probs.tolist())
        return results

    def _extract_ngrams(self, text: str) -> list[str]:
        clean = re.sub(r"\s+", " ", text.lower().strip())
        pad = "." * (self.n - 1) + clean + "." * (self.n - 1)
        return ["".join(g) for g in ngrams(pad, self.n)]

    def _transform(self, value: int) -> float:
        return float(np.log1p(value)) if self.use_log else float(value)

    def _compute_totals(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.lang_list_)
        node_tot = np.zeros(n)
        edge_tot = np.zeros(n)
        for i, lang in enumerate(self.lang_list_):
            node_tot[i] = sum(self._transform(self.Wv[v].get(lang, 0)) for v in self.V)
            edge_tot[i] = sum(self._transform(self.We[e].get(lang, 0)) for e in self.E)
        return node_tot, edge_tot


class NgramEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble combining multiple n-gram classifiers and LIGA classifiers with a LightGBM meta-model.

    Parameters
    ----------
    n_values : list[int]
        n-gram lengths for NGramClassifier.
    top_k_values : list[int]
        Top-K values corresponding to n_values.
    distances : list[str]
        Distance metrics for NGramClassifier.
    liga_n_values : list[int]
        n-gram lengths for LIGAClassifier.
    liga_optimizations : list[str]
        Optimization modes: 'plain', 'log', 'median', 'log+median'.
    model_params : dict
        Additional parameters for LGBMClassifier.
    feature_selection : bool
        Whether to apply feature selection before meta-model.

    Attributes
    ----------
    classifiers : dict[str, BaseEstimator]
        Trained individual classifiers.
    selector : SelectFromModel or None
        Feature selector if enabled.
    model : LGBMClassifier
        Trained meta-model.
    feature_names_ : list[str]
        Names of features fed to the meta-model.

    Methods
    -------
    fit(X, y)
        Fit all sub-classifiers, extract features, and train meta-model.
    predict(X)
        Predict labels using the ensemble.
    predict_proba(X)
        Predict probabilities using the ensemble.
    _parse_liga_opt(opt)
        Map optimization string to flags.
    _extract_features(names)
        Build feature matrix from sub-classifier outputs.
    _generate_feature_names()
        Generate list of feature column names.
    """

    def __init__(
        self,
        n_values: list[int] = [1, 2, 3, 4, 5],
        top_k_values: list[int] = [100, 300, 1250, 1250, 1250],
        distances: list[str] = ['rank', 'cosine_tfidf', 'match_rate'],
        liga_n_values: list[int] = [3],
        liga_optimizations: list[str] = ['plain', 'log', 'median', 'log+median'],
        model_params: dict = None,
        feature_selection: bool = False
    ) -> None:
        self.n_values = n_values
        self.top_k_values = top_k_values
        self.distances = distances
        self.liga_n_values = liga_n_values
        self.liga_optimizations = liga_optimizations
        self.model_params = model_params or {}
        self.feature_selection = feature_selection
        self.classifiers: dict[str, BaseEstimator] = {}
        self.selector = None
        self.model = None
        self.feature_names_: list[str] = []

    def fit(self, X: list[str], y: list[str]) -> "NgramEnsembleClassifier":
        X_list = list(X)
        y_arr = np.array(y)
        self.classifiers = {}
        self.label_order_ = sorted(np.unique(y_arr))

        # NGramClassifier instances
        for dist in self.distances:
            for n, k in zip(self.n_values, self.top_k_values):
                key = f"ngram_{dist}_n{n}"
                clf = NGramClassifier(n=n, top_k=k, distance=dist).fit(X_list, y)
                self.classifiers[key] = clf

        # LIGAClassifier instances
        for n in self.liga_n_values:
            for opt in self.liga_optimizations:
                log_flag, med_flag = self._parse_liga_opt(opt)
                key = f"liga_n{n}_{opt}"
                clf = LIGAClassifier(n=n, use_log=log_flag, use_median=med_flag).fit(X_list, y)
                self.classifiers[key] = clf

        # Feature extraction
        X_feat = self._extract_features(X_list)

        # Optional feature selection
        if self.feature_selection:
            tmp = LGBMClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
            tmp.fit(X_feat, y_arr)
            self.selector = SelectFromModel(tmp, threshold='mean', prefit=True)
            X_feat = self.selector.transform(X_feat)

        # Meta-model
        self.model = LGBMClassifier(
            class_weight='balanced', n_jobs=-1, random_state=42, **self.model_params
        )
        weights = compute_sample_weight(class_weight='balanced', y=y_arr)
        self.model.fit(X_feat, y_arr, sample_weight=weights)

        self.feature_names_ = self._generate_feature_names()
        return self

    def predict(self, X: list[str]) -> np.ndarray:
        X_feat = self._extract_features(X)
        if self.feature_selection and self.selector is not None:
            X_feat = self.selector.transform(X_feat)
        return self.model.predict(X_feat)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        X_feat = self._extract_features(X)
        if self.feature_selection and self.selector is not None:
            X_feat = self.selector.transform(X_feat)
        return self.model.predict_proba(X_feat)

    def _parse_liga_opt(self, opt: str) -> tuple[bool, bool]:
        return {
            'plain': (False, False),
            'log': (True, False),
            'median': (False, True),
            'log+median': (True, True)
        }[opt]

    def _extract_features(self, names: list[str]) -> np.ndarray:
        mats = []
        for key, clf in self.classifiers.items():
            proba = clf.predict_proba(names)
            mats.append(proba)
        # name lengths
        lengths = np.array([[len(n)] for n in names], dtype=int)
        mats.append(lengths)
        return np.hstack(mats)

    def _generate_feature_names(self) -> list[str]:
        names: list[str] = []
        for key in self.classifiers:
            for cls in self.label_order_:
                names.append(f"{key}_proba_{cls}")
        names.append('length')
        return names

    @property
    def feature_importances_(self) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        if self.feature_selection and self.selector is not None:
            mask = self.selector.get_support()
            names = np.array(self.feature_names_)[mask]
        else:
            names = self.feature_names_
        return pd.Series(self.model.feature_importances_, index=names)
