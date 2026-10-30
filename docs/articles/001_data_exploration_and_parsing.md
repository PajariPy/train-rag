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

## The last flags: open text, and two more LLM helpers

After the Slack judge, every source still had a small pile of flagged documents. I went
through them one source at a time, and each time asked the same question: is this a format
I don't know yet, or a document that just doesn't have this field?

### A few more formats

Some of the remaining flags were still formats:

| What I found | Where |
|---|---|
| A `Bcc:` header, and emails separated by `-----Message 2-----` lines | Gmail |
| `msg_N_body:` sections with only a sender: a group or company-wide message, so I keep the sender and expect nothing else | Gmail |
| The description under `summary:` or `detailed_description:` | GitHub |
| The description under any `*_summary:` key: `root_cause_summary`, `impact_summary`, `postmortem_summary`… | Linear |

### Open text

The rest had no structure to find. A Gmail "thread" that's just a list of message bodies
with no headers, a Jira ticket with no issue summary, a Linear issue that's one paragraph
with no keys. I could keep writing formats for these, but there's nothing left to extract.
The useful part is the text itself.

So I added a fallback for every parser: when a document matches none of its source's
formats, the parser returns **open text** instead of a half-filled model. Open text is the
whole document, with line breaks normalised, and nothing else. It's never flagged, because
nothing is expected from it.

It also gets the same length rule I first used for Slack: open text under 300 characters is
marked **not useful**. Most of it is "status: open", "tbd" or a two-word reply, and isn't
worth embedding. The same goes for a whole Gmail thread under 300 characters.

Having one class for this matters later. Every unstructured document, from any source, will
go through the same preprocessing, instead of each source getting its own special case.

| Source | Open text | Of which not useful |
|---|---:|---:|
| Jira | 370 | 1 |
| Fireflies | 131 | 3 |
| Gmail | 30 | 8 |
| Slack | 24 | 23 |
| Linear | 13 | 0 |
| GitHub | 1 | 0 |

For Jira this replaced the open question from my last round: a ticket without an
`Issue summary` is open text. A few lines of the ticket might be orphaned outside any
section, but without a summary there's nothing to attach them to.

### Meeting dates: another LLM judge

Fireflies was different. About 1,500 meetings had no date in their header, and here the
date matters: it's what puts a meeting in order with the emails and tickets around it. But
a meeting without a date header still says a lot about its date: "send the pricing
sheet by 2026-03-06", "after the latency regression on the 12th", a file named
`qbr-aug2026.pdf`. So I built a second judge, the same way as the Slack one.

**First, test it.** I took meetings whose date I knew and removed it: every header line with
a date and every line that repeats it. Then I made 70 test cases:

- **25 with a clue hidden** in one of five ways: someone saying today's date, a dated event
  "2 days ago", a deadline "one week from today", a "week of" banner, or a file name stamped
  with today's date.
- **25 without one:** 15 with no date at all, and 10 with a date that says nothing about
  when the meeting was held, like an event years ago or a contract expiring years ahead.
- **20 real meetings** that keep their own dates in the text (mostly action-item due dates),
  to see how close the judge gets on clues as they actually occur.

The judge returns a date, how precise it is (day, week or month), and a quote as evidence.

The first run had two false positives. Both were my fault: my date stripping missed
compact dates like `20260610` and forms like `july12`, so the "no date" meetings still had
one. After fixing the stripping, the judge got all 50 synthetic cases right. On the 20 real
meetings, every date was within 3 days, with a median of 1 day off. That makes sense:
"deliverables by Friday" says the meeting was in the days before that Friday, not on which
day. So most answers come back with week precision.

**Then, run it on everything.** This time I needed a date for every meeting, not a share,
so sampling wouldn't help. I used Gemini's batch API instead: half the price, with results
within 24 hours.

| | Meetings |
|---|---:|
| Judged | 1,499 |
| Date found | 1,137 (901 to the week, 234 to the day, 2 to the month) |
| No date in the text | 362 |
| Cost | $5.52 (I estimated $3.50; the model thought for longer than I expected) |

126 of the dates come from evidence with no year in it ("by Tue 9/14"), so their year is a
guess, and I store that alongside the date. A few outliers (2021, 2024) are worth a look
before relying on them.

**Inferred is not parsed.** The judge's answer doesn't go into `held_on`. It goes into a
separate field, `held_on_inferred`, stored in its own table with the method that produced
it. The parser uses it only when the header has no date. Retrieval can then decide how far
to trust a week-precision guess, and a rerun with a better judge replaces only its own rows.

Meetings where the judge found nothing keep an empty date. There's nothing left to recover,
so they aren't flagged any more either.

### Attendees: a small extractor

16 meetings had no attendee list, and their transcripts used speaker labels my parser
didn't recognise: `Speaker 1 (Jasmine):`, `[S00:00] Ari (Redwood AE):`. Writing formats for
16 documents wasn't worth it, so a small LLM call lists the people who speak in each one,
with a role when the text gives one. It found speakers in all 16, for a few cents. Like the
dates, they're stored separately as `attendees_inferred`.

Meetings with no Fireflies summary became open text, like everything else without its key
field.

### Zero flags isn't zero bugs

A flag only says a field is *missing*. It says nothing about a field that's there but holds
the wrong text. So, as a last check, I measured every parsed field in characters and looked
at the largest values. A summary of 20,000 characters isn't a summary.

The outliers found six bugs that no flag could have shown:

| Field | What was wrong | Largest before → after |
|---|---|---:|
| Fireflies summary and topics | without a `transcript:` key, they ran on into the transcript | 20,779 → 1,205 |
| Fireflies duration | `~46 minutes. Attendees: …` on one line | 205 → 66 |
| Gmail recipients | lists separated by `;` were kept as one string | 3,462 → 774 |
| Slack messages | ` dave_oncall:` or `maria gonzalez:` lines were merged into the message above | first message, p99: 575 → 437 |
| Jira issue summary and impact | escaped line breaks, and values running past headings | 3,161 → 1,372 |

Slack was the tricky one. A looser speaker pattern would split those messages, but it also
matches `name:` and `max_tokens:` lines in code blocks. Thread starters are almost always
people, so I took the first names of everyone who starts threads (453 of them) and let a
looser line start a message only when its speaker has one of those names, or is a bot.
Code keys stay inside their messages.

What's still large is real: incident threads written on one line, emails that quote the
whole thread below them, and Linear issues that are full product specs. Those are a
chunking problem, not a parsing one.

### Where things stand

| Source | Documents | Parsed | Open text | Flagged |
|---|---:|---:|---:|---:|
| Slack | 285,605 | 285,581 | 24 | 0 |
| Gmail | 121,390 | 121,360 | 30 | 0 |
| Linear | 35,308 | 35,295 | 13 | 0 |
| HubSpot | 15,017 | 15,017 | 0 | 0 |
| Fireflies | 10,173 | 10,042 | 131 | 0 |
| GitHub | 8,052 | 8,051 | 1 | 0 |
| Jira | 6,120 | 5,750 | 370 | 0 |

Every document of the seven parsed sources is now either a typed model or open text, and
nothing is waiting for review. In Fireflies, 1,119 meetings use an inferred date and 342
have none; 16 use inferred attendees.

Earlier I wrote that the goal wasn't zero. It still isn't: the zero comes from deciding what
each leftover document *is*, not from forcing every document into a model.

## What's next

- Store the parsed results in Postgres next to the raw documents.
- Generate cheap LLM summaries for the sources that don't have a free one (Slack, Gmail, Linear,
  Google Drive, HubSpot, Confluence), with a cost estimate before each run.
- Decide the shared preprocessing for open text.
