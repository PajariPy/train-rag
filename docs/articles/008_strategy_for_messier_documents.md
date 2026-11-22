# Strategy for messier documents

> [!note] Status
> Draft. Descriptive only, no build.

EnterpriseRAG-Bench is clean: every document is text, every Slack message is a full
paragraph, and the answer is always written down somewhere. Real company data is messier.
This article covers what would have to change in the pipeline to handle it.

One strategy works for every kind of mess, so I describe it first. After it come short notes
on the techniques specific to images, tables and Slack threads.

## The common strategy

### Turn the mess into meaningful text

Search works on text, so every document needs meaningful text about it. An OCR'd table, a
raw screenshot or a chain of "+1" messages matches a question poorly, both semantically and
by keywords, unless the user happens to type the exact name of the table or image. So
extracting the raw content is not enough: I also need to capture what the document is
about. Once that description exists as text, it can be chunked and indexed like any other
document.

### From LLM labels to an SLM

Describing every document means running a model over all the company's data: millions of
documents and billions of tokens. With an LLM that costs a lot; a small model fine-tuned for
this one task does the same work for a fraction of the price. So the LLM labels only a
sample, and the SLM it trains processes everything else:

1. **LLM descriptions:** a strong LLM writes detailed descriptions (and extracts fields)
   for a small sample of documents.
2. **Manual review:** I read them, catch hallucinations and fix the prompt. The reviewed
   sample becomes the calibration set.
3. **Training data:** the LLM labels a larger sample, which I split into training and test
   sets for the SLM.
4. **SLM training:** I train an SLM that takes the raw document (text, table or image) as
   input, then score it on the test set and on the same calibration set the LLM was checked
   against.

I have already taken this route for Slack: Qwen3.5-2B, trained on DeepSeek labels,
extracts fields that search as well as DeepSeek's own (articles 005–007).

### Why invest in the SLM

Extracted data is much easier to work with than raw data. Once every document has a
description and fields, each later step gets cheaper:

- **Search:** descriptions are embedded and matched like any other text, and fields work as
  filters.
- **Answering:** the LLM reads short, clean descriptions and fields instead of raw images,
  whole spreadsheets or long threads: fewer input tokens, fewer vision calls and faster
  responses.
- **Cost:** the SLM runs once per document, at ingestion, on a cheap GPU; the LLM runs on
  every question. Every piece of work moved from question time to ingestion time saves LLM
  cost and latency on every answer that follows.

Two more benefits come from owning the model:

- **New data stays cheap:** a company produces new documents every day. With the SLM, each
  one takes a fraction of a second of GPU time to process, so a growing corpus or a sudden
  burst of new data barely changes the bill. With an LLM, the cost would grow with every new
  token.
- **No dependence on a provider:** the SLM runs on any GPU I rent or own. A price change, a
  deprecated model or an API outage at the LLM provider doesn't stop documents from being
  processed.

The price is the upfront work of labelling, review and training. It pays off once the corpus
or the question volume is large, and it keeps paying: the same labelled data can train the
next SLM when the data changes.

### Hallucinations

The main weakness of this pipeline is hallucinated content. As long as the descriptions are
only used for search and never shown to the user, a few hallucinations are acceptable: what
matters is that retrieval accuracy stays high.

The answer is where hallucinations start to hurt. If the agent answers a question about a
pricing table from the SLM's description, and the description got a number wrong, the user
sees the wrong number. So once the right document is found, there are two options:

- **Trust the description:** push the SLM's hallucination rate down far enough that its
  descriptions are safe to answer from. This keeps the cheap path all the way to the answer.
- **Go back to the source:** use the description only to find the document, then let a
  calibrated LLM agent read the raw image, table or thread and pull the answer from it. This
  is more accurate, but it brings back the LLM cost, though only for the few documents that
  were retrieved.

Hallucinations can be measured with RAGAS faithfulness or DeepEval's hallucination metric.
To reduce them, the SLM can be asked to cite its numbers, which are then checked against a
deterministic source such as OCR, and uncertainty can be estimated from the logits. Which of
these techniques are needed depends on the data and on how often the model hallucinates.

## Techniques for specific data

### Images and screenshots

In a company, an image can be almost anything: a screenshot of a presentation, photos from
the team's bowling night, a picture of a table, a screenshot of a chat, a scanned PDF with a
contract. Plain OCR can't cover that variety, and an image can't be chunked at arbitrary
points. The SLM here is a vision model, and its description must say what the image shows,
not just transcribe the text in it. OCR still has a role: it is the deterministic source the
description's numbers are checked against.

### Tables and spreadsheets

A table split into chunks loses its headers, so a row of numbers no longer says what they
mean. The description covers what the table is about, and every chunk keeps the table's
title and column headers. Exact lookups and aggregations ("total spend in Q3") are better
answered by querying the table itself than by search, so the table is also kept in a form
the agent can query.

That form doesn't come for free. Many real tables are information blocks laid out as a grid
rather than proper tables: merged cells, several header rows, subtotals mixed in with the
data, notes in the margins. Before they can be queried, they have to be normalized into
clean rows and columns. The normalization follows the same pattern as the descriptions: an
SLM or a dedicated table-parsing tool does the work, checked against an evaluation dataset
of normalized tables and a round of manual review.

### Fragmented Slack threads

Real threads are full of "+1", "lgtm" and "same as last time", and the answer is often spread
over several short messages, sometimes across different threads or days.

The goal is to normalize this into the clean form the benchmark already has: one thread, one
topic, the facts written in full. Then the same pipeline used for EnterpriseRAG-Bench applies
unchanged. To get there, each message becomes a node in a small directed graph, with an edge
wherever it depends on an earlier message: a reply, a "+1" to a proposal, a "same as last
time" pointing back to an older thread. A message that depends on nothing starts a new
piece.

Finding those edges needs proper metadata on every message: time, participants and channel.
With it, many links can be found by rules: an explicit reply, the same people in the same
channel within a few minutes, a ticket id mentioned in both. The links rules can't decide,
like what "the thing Maya fixed" refers to, go to an LLM-based linker that reads the
candidate messages and picks the one meant.

Following the edges groups the messages into self-contained pieces, and each piece is
merged into a single block of text. Filler like "+1" adds no facts, so it disappears in the
merge, but short decisive messages ("approved", "rolled back to 1.21.4") carry the outcome
and are kept.
