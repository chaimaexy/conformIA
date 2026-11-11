"""BM25 en mémoire, avec racinisation française.

Pourquoi pas la recherche plein texte de Postgres ? `ts_rank` n'a pas d'IDF : un mot présent
partout (« les », « part ») pèse autant qu'un mot rare et discriminant (« annexe », « 113 »).
Une question en langage naturel donne alors soit 0 résultat (tsquery en ET), soit presque tout
le corpus mal classé (en OU). BM25 pondère chaque terme par sa rareté (IDF) et sature la
fréquence (k1) en tenant compte de la longueur du chunk (b). Le corpus (~1 200 chunks par
stratégie) tient en mémoire : l'index se construit en moins d'une seconde.

score(q, d) = somme sur t de q : IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / moy))
IDF(t)      = ln(1 + (N - n(t) + 0,5) / (n(t) + 0,5))
"""

import math
import re
from collections import Counter
from functools import lru_cache

import snowballstemmer

# Mots-outils français (liste courte : on garde les mots juridiques comme « sauf », « avant »)
_STOPWORDS_TEXT = """
    a à afin ai aie ainsi alors au aucun aucune aussi autre autres aux avec avez avoir avons
    c ce ceci cela celle celles celui cet cette ces chaque ci comme comment d dans de des dont
    du elle elles en entre est et été être eu eux fait faire il ils j je l la le les leur leurs
    lui m ma mais me même mes moi mon n ne ni nos notre nous on ont ou où par pas peu peut
    plus pour pourquoi qu quand que quel quelle quelles quels qui quoi s sa se ses si son sont
    sous sur t ta te tes toi ton tous tout toute toutes tu un une vos votre vous y
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())
TOKEN = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _stemmer():
    return snowballstemmer.stemmer("french")


def tokenize(text: str) -> list[str]:
    """Minuscules, élisions (« l'annexe » -> « annexe »), mots-outils retirés, racines."""
    words = [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS]
    return _stemmer().stemWords(words)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        n_docs = len(documents)
        df = Counter(term for tf in self.tfs for term in tf)
        self.idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        # index inversé : terme -> documents qui le contiennent (évite de parcourir tout le corpus)
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for term in tf:
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.tfs[i][term]
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_length
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
