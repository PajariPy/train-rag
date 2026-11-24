Before the first search, the agent turns the conversation into one standalone search
query. Today the pipeline searches with the last message as it is. ^desc

## Why
In a real chat the last message often depends on earlier turns ("and what about the EU
one?", "who fixed it?"). Searching with it alone misses the subject. Our dev sets have only
single, self-contained questions, so this gap isn't measured yet.

## How
1. A query-preparation step (Gemini) gets the chat history and the last message and writes
   the search query: subject resolved, names and context from earlier turns kept, no answer
   guessed. Today's rewrites for BM25 then run on that query.
2. Test set: multi-turn variants of dev questions (the question split over 2–3 turns, or a
   follow-up that needs the earlier turn). Compare last-message search vs prepared query.
3. Also the place to merge what `ask_user` replies add (channel, team, time) into filters.

Paid: ask first (Gemini, small).

## Decision (23 Nov): its own evaluation
The final evaluation ([[Run the final evaluation]]) tests single, already-composed questions
only. Composing the query from the chat, and accuracy inside a conversation, are measured
here, with their own multi-turn test set and budget: composed query vs last message,
accuracy across the turns, cost per turn. Research map: problem "Final evaluation", step 1.
