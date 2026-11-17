# gemma: a local decision model for sensitive data

Built by team Rahmy at Junction 2026. It won one of the three Best Use of
Gemma prizes from Google DeepMind.

Some data must not go to a public API: patient messages, health records,
legal or HR cases. This project shows how to get a good decision model for
such data anyway:

1. **Write fake scenarios.** Gemini writes realistic, made-up messages from
   structured tuples (the method of *Evaluating and Improving AI Agents*,
   section 1.4).
2. **Label them with the real teacher.** The Gemini Decisions API
   (`gemini-decision-1`) answers typed questions about each fake message,
   with a probability for each option. The data is fake, so this is safe.
3. **Teach a local student** on a Modal GPU (distillation, section 5.5.7).
4. **Run the student on the real data, on your own machine.** The real data
   never leaves it.

The example task is triage of patient messages at a clinic.

## Run it

```sh
runit all        # scenarios → labels → benchmark on Modal → local decider
runit demo       # the demo page at http://127.0.0.1:8250/demo
runit --help     # every step on its own
```

You need `GEMINI_API_KEY` in `.env` and a Modal login (`.venv/bin/modal setup`).

`runit` is a small launcher that reads `.runit`. If you do not have it, use
the same commands directly:

```sh
uv sync --python 3.12                     # make .venv
.venv/bin/python -m lab --help            # every step (python -m lab bench, …)
.venv/bin/uvicorn lab.serve:app --host 127.0.0.1 --port 8250   # the local decider
```

The demo page (lab/demo.html, lab/demo.py) shows the three local models on
fake test data: Gemini's triage slip over the local copy, a message to sort,
an inbox search and a voicemail to write down. It calls only the local server.

## The questions (lab/task.py)

One question of each type of the Decisions API:

| Name | Type | Asks |
|---|---|---|
| `triage` | choice | emergency, same_day, routine or admin |
| `route` | choice | nurse, doctor, pharmacy, front_desk or billing |
| `callback_today` | noul (yes/no) | must a clinician phone back today? |
| `severity` | score (5 levels) | none to critical |

## Three decision students (lab/students/)

| | `lora` | `embed` | `embed` with EmbeddingGemma 2 |
|---|---|---|---|
| Base model | Gemma 3 1B (`unsloth/gemma-3-1b-it`) | EmbeddingGemma (`unsloth/embeddinggemma-300m`) | EmbeddingGemma 2, text side only (`unsloth/embeddinggemma-2`) |
| How it answers | One forward pass. It reads the logits of the option tokens (nishtahir's decision model). | It makes one vector of the message. A logistic regression for each question gives the probabilities. | The same as `embed` |
| What learns | A LoRA adapter on every linear layer, with a KL loss against Gemini's probabilities; the best epoch on val; then one temperature for each question | A LoRA adapter on EmbeddingGemma and the regressions together, then the regressions again on the tuned vectors (`--tune-epochs 0`: only the regressions) | The same as `embed` |
| Question types | noul, choice, score (up to 26 options) | noul, choice, score | noul, choice, score |
| New question with no training | Yes (plain Gemma, less good) | No | No |

All use the same interface (`fit`, `predict`, `save`, `load`), so one can
replace another. `runit bench` trains them and compares them with plain
Gemma and with the prior (Gemini's average answer) on test messages that
they did not see. It reports the coverage and challenge sets separately.
`runit table FOLDER…` puts saved students in one table.

## The reranker (lab/rerank.py)

The same distillation for search: Gemini writes 200 fake staff searches and
grades every message against every search (one noul question for each
search, 50 in one Decisions call). Gemini's P(match) is the graded label.

- **cross**: Gemma 3 1B + LoRA reads the search and the message together.
- **bi**: EmbeddingGemma 1 or 2 + LoRA makes a vector for the search and
  one for the message. The message vectors can be stored, so a search is
  fast.
- **two stages**: the bi-encoder keeps its top 30, the cross-encoder orders them.

`runit rerank bench` reports NDCG@10, Spearman and recall@10 against Gemini,
on test searches over test messages.

## The transcriber (lab/transcriber.py, lab/voice.py, lab/lm.py)

EmbeddingGemma 2 can hear (it has a 300M audio encoder), but it does not
write text. The eg2 way (eg2.haku.sh) makes it a transcriber:

1. The audio encoder stays frozen. It gives one vector for each 40 ms.
   With `--mix`, the reader also gets the 12 inner layers and learns one
   weight for each layer.
2. A small reader (14M–20M numbers) moves each 40 ms slice into
   EmbeddingGemma 2's text space, or says "blank".
3. The text side makes one vector for each word of a list of about 200,000
   English words (the LibriSpeech lexicon plus the train and val words).
   The closest word wins (CTC), so it can write a word that it never heard
   in training.
4. A word trigram LM (LibriSpeech train transcripts plus the train
   voicemail scripts, 7 MB) and a beam search choose between close words.
   Its weights are chosen on val.
5. A tiny GRU adds capitals and punctuation.

Long audio goes in 30 s windows that overlap by 4 s. Training data:
LibriSpeech (open speech, CC BY 4.0) and fake voicemails: Gemini writes the
spoken scripts, `gemini-3.8-flash-tts` speaks them with one of 30 voices,
and a phone-line effect is added. `gemini-3.5-transcribe` on the same test
clips is the bar. `runit voice table` shows every run.

| Run | Test voicemails, phone line (WER) | LibriSpeech test-clean (WER) |
|---|---|---|
| gemini-3.5-transcribe (the bar) | 2.8% | – |
| 100 h, 4-layer reader | 11.6% | 8.1% |
| 460 h, 6-layer reader | 9.3% | 5.9% |
| the same, overlapping windows | 8.1% | – |
| the same, overlapping windows + word LM | 6.9% | – |

Other people who built the same thing: a copy of eg2 by mlboydaisuke gets
6.81% (100 h) and 4.21% (960 h) on test-clean, greedy. norsk-asr shows
that a word LM and a mix of the encoder layers help most.

## The local decider (lab/serve.py)

`runit` starts it at `http://127.0.0.1:8250`. It takes the same request as
the Gemini Decisions API and gives the same answer shape:

```sh
curl -s 127.0.0.1:8250/v1/decisions -H 'content-type: application/json' -d '{
  "input": "my dad is very confused since this morning and his face looks droopy",
  "questions": {"triage": {"type": "choice", "instructions": "How urgent?",
    "criteria": {"emergency": "…", "same_day": "…", "routine": "…", "admin": "…"}}}}'
```

It also has `POST /v1/rerank` (`{query, documents, top_n}`) and
`POST /v1/transcribe` (the audio file is the body). It listens on 127.0.0.1
only and keeps nothing that it reads.

## Files

- `lab/task.py`: the questions and the dimensions of the scenarios
- `lab/scenarios.py`: tuples, messages, realism filter, duplicates, splits
- `lab/teacher.py`: Gemini Decisions labels (soft labels)
- `lab/students/`: the decision students
- `lab/rerank.py`: queries, relevance grades, the rerankers and their scores
- `lab/voice.py`: spoken scripts, speech, the phone line, the Gemini bar
- `lab/transcriber.py`: the ear, the reader, the word vectors, punctuation
- `lab/train.py`: training and scores, used on Modal and locally
- `lab/modal_app.py`: the Modal GPU app
- `data/`: scenarios, labels and benchmark reports (all fake data)
- `adapters/`: trained students (not in git)

## License

MIT. See [LICENSE](LICENSE).
