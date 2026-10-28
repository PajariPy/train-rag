# Before any embeddings: reading half a million company documents

Status: draft.

I'm building an agentic RAG system, and I want to test it on something that looks like a real
company's knowledge rather than a pile of Wikipedia pages. I started with HERB, but it turned out
to have too little data, so I switched to
[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench). It
simulates one company, Redwood Inference, through 511,962 internal documents from nine tools:
Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira and Confluence. It also comes
with 500 questions, each with a gold answer and the documents that hold the evidence.

It's tempting to go straight to chunking and embedding. I didn't. Before choosing how to index
anything, I wanted to know what is actually inside these documents. This post covers that first
step: exploring the data, and then parsing it into structured fields without using an LLM.

## Reading the data

I put the whole corpus into Postgres first, with each document's source, size and origin. That
way every later step can query it, instead of me rereading parquet files.

Then I went through every one of the nine document types and read samples side by side. I was
looking for three things:

1. **Key fields:** which parts of a document actually carry its meaning.
2. **Metadata:** who, when and where. That's useful for filtering, and for linking documents
   to each other.
3. **Free summaries:** text the document already contains that can stand in for an LLM summary.
   Every summary I don't have to generate is money I don't spend.

Here's what I found:

| Source | Documents | Key fields | Metadata | Free summary |
|---|---:|---|---|---|
| Slack | 285,605 | thread messages as `speaker: text` lines | channel, thread starter, participants (first names, sometimes with a role) | — |
| Gmail | 121,390 | the emails of one thread | per email: from, to, cc, date, subject, attachment names | — |
| Linear | 35,308 | description | participants and dates from `YYYY-MM-DD - Name (Role):` comments | — |
| Google Drive | 25,108 | free text | — | — |
| HubSpot | 15,017 | notes, next step, blockers, timeline | company or deal name (the title) | — |
| Fireflies | 10,173 | summary, topics, transcript | attendees with roles, date, start time, duration | the Fireflies summary |
| GitHub | 8,052 | description, review discussion | participants from the review | the description |
| Jira | 6,120 | issue summary, description, impact, environment | participants with roles, dates from comments | `Issue summary` |
| Confluence | 5,189 | free text | — | — |

Three sources already come with something I can embed for free. That's a nice start.

### Things that surprised me

- **Everything is already flattened into text.** I expected to deal with exports from each tool.
  Instead, whoever built the dataset had already turned every record into plain text. Linear,
  Jira and GitHub became `key:` sections, Gmail became a list of email strings, and Slack became
  `speaker: text` lines.
- **No images, no tables.** I went looking for both and found neither, so no OCR and no table
  extraction. Attachments exist only as file names mentioned in emails; the files themselves
  aren't in the dataset.
- **One Gmail document is a whole thread,** a median of 5 emails, not a single email. Replies
  often drop their headers, so `To`, `Date` and `Subject` show up on only about three-quarters
  of the emails.
- **One source, many formats.** Fireflies writes its meeting header in at least five different
  layouts. GitHub keeps the review discussion under nine different keys. Gmail threads come in
  three storage formats. A parser that only handles the most common layout misses a lot.
- **Some metadata just isn't there.** I planned to take attendees and a date from every
  Fireflies meeting. But many meetings have neither, except inside the transcript, where people
  appear only as speakers.

### The same id, twice

Every document has an id, `doc_id`, and I assumed it was unique. It isn't: four ids appear twice.
My first thought was to merge the duplicates, so I looked at each pair:

| Shared id | Documents | Same document? |
|---|---|---|
| `dsid_6df5…3340` | two Jira tickets, "Cleanup stale test instances…" | **Yes, two versions.** The longer one is an update: the cost is revised from ~$1.2k to ~$0.9k, and the grace window changes from 72 hours to 5 business days. |
| `dsid_8a0c…f4df5` | Confluence account brief + HubSpot record for Signal Peak Logistics | No: same customer, different tools |
| `dsid_f292…6490` | Confluence sprint plan + Drive doc on SDK docs triage | No: same project, different tools |
| `dsid_feb1…f01f0f` | Confluence incident review + Jira ticket on an API gateway 5xx surge | No: same incident, different tools |

Three of the pairs are different documents that cover the same topic, so merging them would
throw away a source. The Jira pair really is one ticket in two versions, and that one looked like
an easy merge.

Then I found the question that uses it. Question `qst_0413` asks: *"Who approves terminating
STALE-ORPHAN test instances after the 72h grace window: infra manager or cost-ops?"* It's a
"conflicting information" question, and it cites that ticket twice, on purpose. To answer it,
you have to notice that the two versions disagree. If I'd merged them, I would have quietly
deleted the exact conflict the benchmark is testing.

So I kept every document. In my database a document is identified by where it came from (source
file and row), not by `doc_id`. Each parsed document links back to its row.

## Parsing without an LLM

With a picture of the data, I wrote parsers. They're deterministic: regular expressions and
section splitting, no LLM. A run over all half a million documents takes about a minute and
costs nothing.

- **One parser per source.** Each parser is a class with one method, `parse(document)`, that
  returns a typed model for that source: `SlackThread`, `EmailThread`, `JiraTicket` and so on.
  A registry maps each source to its parser, so I can swap one implementation for another later
  without touching the rest of the pipeline.
- **Shared pieces.** Comments (author, role, date, text) and people (name, role) are parsed the
  same way for Linear, Jira and GitHub.
- **Parsers never crash.** A parser returns whatever it found and lists the expected fields it
  couldn't find. One odd document shouldn't stop a run over 500K.
- **Google Drive and Confluence are skipped.** They have no structure that holds across
  documents, so they'll go straight to LLM summaries later.

I tested each parser against real documents from the dataset, one test document for each
format variant I found.

## Flag, review, repeat

A document gets **flagged** when an expected field is missing. I don't throw flagged documents
away. I go back and read them, to find out whether the data is really missing or my parser just
doesn't know that format yet.

My first full run flagged three-quarters of all Fireflies meetings. That number was a good
reason to read the flagged documents, and they showed that my assumptions were too narrow:

| What I found in the flagged documents | What I changed |
|---|---|
| The meeting header sometimes sits inside the transcript | search the whole document |
| Headers separated by `|`, or labelled `Date/Time` | accept both |
| Attendees split by company: `Attendees (Redwood):`, `Attendees (Acme):` | read every attendee list |
| Dates written out: `March 3, 2026` | parse written dates |
| Line breaks stored as the literal characters `\n` | normalise them first |

After a few rounds, Fireflies went from 74% flagged to 23%. The same loop caught bugs in other
sources too: Slack messages from `Name (Role):` speakers were being dropped, and words like
"Goal" and "Blocking" were being mistaken for comment authors in tickets.

At that point I assumed the rest was genuinely missing data. Then I read another round of flagged
documents and found more formats I hadn't seen yet:

| What I found | Where |
|---|---|
| Prose with no section key, often under capitalized labels like `Summary:` or `## Context` | Linear, Jira, GitHub |
| Other section names: `background:`, `summary_long:`, `pr_body:` | Linear, Jira, GitHub |
| The whole document stored as a Python list of strings | Slack, Fireflies |
| One `message_N_body:` section per email | Gmail |
| `Header:` instead of `Meeting header:`, and `Attendees` without a colon | Fireflies |
| Lowercase full names (`maria gonzalez:`) and lines starting with a space | Slack |

So I changed how parsers handle formats. When a field can be written in several ways, the parser
now has one small method per format and tries them in order until one finds the field. Supporting
a new format means writing one method and adding it to the list. Each parsed document also records
which method found each field, so I can count how often every format occurs.

Where things stand now:

| Source | Flagged | Main reason |
|---|---:|---|
| HubSpot | 0% | — |
| Linear | 0.1% | no description in any format |
| Gmail | 0.2% | emails with no headers at all |
| Slack | 0% | — (threads without a channel are labelled group chats, see below) |
| Jira | 6.0% | no issue summary label of any kind |
| GitHub | 0.02% | no description (a missing review discussion is normal, so it isn't flagged) |
| Fireflies | 15.7% | no meeting date (attendees fall back to the transcript speakers) |

This time I checked what's left: the remaining flags are data that's really missing, not formats
my parsers don't know.

The goal isn't zero. Once the remaining flags are data that's genuinely missing, the question
changes from "how do I parse this?" to "what do I do with these documents?": use a fallback
field, run an LLM over them, or set them aside.

## When I can't read them all: an LLM judge

Slack was the biggest pile. Its parser takes the channel from the thread's title, and my
first version accepted any title that wasn't a file path. Thousands of titles turned out to be
numbers (`2112233445`), whole sentences, or one-off slugs (`compact-cta-toast-ordering`). The
workspace really has 34 channels, each used as the title of at least 7 threads, so I made
the parser accept only those. That left **9,043 threads with no channel**.

I read about 50 of them. They didn't look broken: they looked like small group chats, set up
to resolve one issue quickly and then dropped. But 50 isn't 9,043, and the channel could
still be hiding in the text: a header, a bot notice, someone saying "keeping this in
#incidents". I couldn't read them all, so I asked an LLM to.

### First, test the judge

Before trusting a judge on data I can't check, I gave it data I can. I took 50 real threads
whose channel I knew, replaced their titles with file names like the channel-less ones, and
labelled them myself:

- **25 with their own channel hidden** in five different ways: a header line, an export
  banner, a `#channel` remark in the middle of a message, a plain-words mention ("posting in
  the product channel"), and a bot notice at the end.
- **25 without it.** 15 never mention a channel. 10 mention a *different* one: "cross-posting
  to #eng-ml", "let's move the follow-up to #incidents", "fyi I pinned this in #product".

The judge is Gemini Flash. It returns a yes or no, the channel name, and a quote from the
thread as evidence, so every answer can be checked.

The first run scored 80% recall. Reading the misses taught me more about my test than about
the judge: all five came from two sentences I'd written, "archived to the incidents digest"
and "pinned this in #vendors". Both point to *another* channel, so the judge was right to
ignore them. That forced me to say the rule out loud: **only the channel this conversation
took place in counts; a mention of any other channel means no channel.** I fixed the labels,
added those sentences as distractors on the "without" side, and wrote the rule into the
prompt. The judge then got all 50 right, including the 10 threads that mention other
channels.

Building the test set also exposed the title bug above: some of my "channel names" for
the distractors came out as `#2154329999`.

### Then, sample instead of running everything

Running the judge on all 9,043 threads would have cost about $16. My budget was $2. But I
didn't need a verdict for every thread, only the answer to one question: *what share of them
reveal their channel?*

To estimate a share within ±3% at 95% confidence, a random sample of **955** threads is
enough, even in the worst case, where the true share is 50%. I drew it in proportion to the
kind of title (file name, number, sentence, slug), so each kind would get its own estimate,
and capped the run at $2.

| | |
|---|---|
| Sample | 956 of 9,043 threads |
| Reveal their own channel | **4 (0.4%)**, 95% interval 0.2%–1.1% |
| Across all 9,043 | about 38 threads (somewhere between 15 and 97) |
| Cost | $1.91 |

Reading the four quotes, one was the judge's own mistake: "keep sending failures to #eng-ml"
points somewhere else. So the real number is lower still.

The answer is clear enough to act on. These threads don't hide a channel; they don't have
one. Spending $16 to recover maybe 30 channels isn't worth it, so I stopped treating a missing
channel as something to review. Every Slack thread without a channel is now labelled a
**group chat**: a short-lived conversation set up to resolve one issue. That's 9,055 threads,
3% of Slack.

What I'd keep from this: test the judge on data I can check before trusting it on data I
can't; let its mistakes show me where my own rule was vague; and when I only need a number,
sample instead of running everything.

## What's next

- Store the parsed results in Postgres next to the raw documents.
- Generate cheap LLM summaries for the sources that don't have a free one (Slack, Gmail, Linear,
  Google Drive, HubSpot, Confluence), with a cost estimate before each run.
- Decide what flagged documents should use instead, starting with Jira tickets that have no
  `Issue summary`.
