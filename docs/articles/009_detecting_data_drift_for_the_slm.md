# Detecting data drift for the SLM

> [!note] Status
> Draft. Theoretical only, no build. Next chapter of the same article as 008.

In the strategy above, a small model (SLM) does the heavy work: it reads every new document
and writes a description of it. It is cheap because I train it once and then just run it.

But the SLM only knows the documents it was trained on. A company's data changes over time:
new tools, new templates, new topics, new people with their own way of writing. When new
documents stop looking like the old ones, the SLM can quietly start making mistakes, and
search gets worse without anyone noticing.

So I need a way to check that the SLM still works on the new data, and to know when it's
time to train it again.

## Detecting it with labels

In production I already track how satisfied users are with the answers: thumbs up and down,
follow-up questions, "that's wrong" replies. If the SLM starts failing, this is where it
shows up first, so the first check is simply whether satisfaction changes over time.

A drop alone doesn't say why. So I also track which documents were ingested and when: how
many, from which tools, channels and teams. If satisfaction falls right after a wave of new
documents, or only for questions whose answers come from a new source, that points to drift
in the data rather than, say, harder questions.

User feedback and the signals below only say that something might be wrong. To confirm it,
I take a small sample of the new documents, label it with the LLM as before, and score the
SLM on it. If the score is clearly below its score on the calibration set, the drift is real
and the SLM needs new training data.

## Detecting it without labels

User feedback comes late: someone has to ask a question and get a bad answer first. Some
signals can be read straight from the new documents and the SLM's output, with no labels
and no users:

- **Input statistics:** document length, language, source mix, share of images or tables.
  A sudden change means the SLM is seeing something new.
- **Embedding shift:** for every new document, find its nearest neighbours in a golden set
  of embeddings (the SLM's training documents) and measure the distance. A document whose
  closest match is far away is something the SLM hasn't seen. This is the
  nearest-neighbour method from out-of-distribution detection
  ([Sun et al., 2022](https://proceedings.mlr.press/v162/sun22d.html)); normalizing the
  embeddings first is key for it to work. It also gives a filter: documents above a distance
  threshold go straight to the fresh-label check.
- **SLM confidence:** low token probabilities in the output (from the logits) mean the model
  is unsure. A step further is a small classifier trained on the calibration set that looks
  at the SLM's output and predicts whether it is right or wrong.
- **Broken output:** a growing share of empty descriptions, missing fields or malformed JSON.
- **Retrieval scores:** track the similarity and reranker scores of the top results for live
  questions. If they slowly fall, or new documents make it into the top results less often
  than old ones did when they were new, search is matching the new data worse.

"Change" here shouldn't be judged by eye. Statistical tests compare a window of new data
with a reference window (the training set, or last month) and say whether the difference
is bigger than chance:

- the **Kolmogorov–Smirnov (KS) test** for numbers: document length, confidence, retrieval
  scores;
- the **chi-square test** for categories: source, language, channel;
- the **population stability index (PSI)**, a common single number for "how much has this
  distribution moved";
- for embeddings, the KS test on the nearest-neighbour distances: distances of new
  documents against distances of held-out golden documents.

A simpler option for embeddings is to average each set into one vector and track the
distance between the two averages
([Dhinakaran, 2022](https://medium.com/data-science/measuring-embedding-drift-aa9b7ddb84ae)).
It works when the data is one kind of document, but a company corpus mixes many topics and
styles: a new topic that is a few percent of the traffic barely moves the average, while a
harmless change in the topic mix does. The nearest-neighbour distance doesn't have that
problem.

For later, there are heavier methods: a domain classifier trained to tell old embeddings
from new, which [Evidently](https://evidentlyai.com/blog/embedding-drift-detection)
recommends as a default, and the reduce-then-test pipelines compared in
[Failing Loudly](https://arxiv.org/abs/1810.11953) (Rabanser et al., 2019). I'd start
with the nearest-neighbour distance and move to these only if it misses drift that
matters.

## When to act

An alert means something is going off, so every alert gets a reaction. What changes is how
urgent it is. To decide that, I define an SLA for the pipeline: a target for the signals
that matter (for example, user satisfaction and the fresh-label score), and an error budget:
how much the pipeline may fall below that target in a month before it counts as a failure.

When drift eats the budget slowly, I still investigate it, and the fix goes into the next
planned retraining. When more than 10–20% of the budget is already used and the current
rate means it will run out before the end of the period, I look at it immediately: confirm
the drift with fresh labels, find the documents behind it, and retrain the SLM on them.

The new training data mixes the new documents with the old ones. A model trained only on
the new documents can get worse on the old ones, which are still most of the corpus.

This is what keeps the cost argument of the strategy above true. The SLM is only cheap as
long as it keeps working; without drift checks, the safe choice would sooner or later be to
send everything back to the LLM.

