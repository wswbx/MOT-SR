# Reproducibility

This release adds the author-supplied final equations and documents the
available reproducibility materials. Code settings below describe the
released implementation; manuscript settings and historical run records
are identified separately. The full equation-search experiments have not
been rerun as part of this material update.

## Prompts, configuration and environment

The task specifications and released datasets are:

| Task | Prompt specification | Dataset directory |
| --- | --- | --- |
| Oscillation 1 | [specification_oscillator1_numpy.txt](specs/specification_oscillator1_numpy.txt) | [data/oscillator1](data/oscillator1/) |
| Oscillation 2 | [specification_oscillator2_numpy.txt](specs/specification_oscillator2_numpy.txt) | [data/oscillator2](data/oscillator2/) |
| E. coli Growth | [specification_bactgrow_numpy.txt](specs/specification_bactgrow_numpy.txt) | [data/bactgrow](data/bactgrow/) |
| Stress-Strain | [specification_stressstrain_numpy.txt](specs/specification_stressstrain_numpy.txt) | [data/stressstrain](data/stressstrain/) |

The equation-generator instruction and evolving prompts are assembled in
[mot_sr/sampler.py](mot_sr/sampler.py) and
[mot_sr/buffer.py](mot_sr/buffer.py). The scientist module's active default
meta-prompt and dynamic analysis prompts are in
[mot_sr/scientist_llm.py](mot_sr/scientist_llm.py); the allowed tools are
listed in [mot_sr/scientist_defaults.py](mot_sr/scientist_defaults.py).
Although [main.py](main.py) defines a scientist `meta_prompt`, the current
sampler does not pass that field to `ScientistLLM`, which uses its default.
The manuscript appendix also describes the prompts.

[environment.yml](environment.yml) is the fuller pinned Linux environment
export, specifying Python 3.11.7, PyTorch 2.0.1, Transformers 4.38.1,
NumPy 1.26.4 and SciPy 1.12.0. [requirements.txt](requirements.txt) is a
shorter pinned dependency list with CUDA 11.8 PyTorch wheels. It omits some
packages imported by the implementation, including `absl-py` and `numba`,
which are pinned in the environment export. Neither file establishes the
exact original environment used for every reported run.

For a compatible Linux/CUDA setup, the supplied environment can be created
with:

```bash
conda env create -f environment.yml
conda activate mot_sr
```

### Entry points and task commands

[run_motsr.sh](run_motsr.sh) starts the local model server and invokes
`python main.py` with the default Oscillation 1 arguments. To select a
task explicitly, start the server in one terminal and wait for model
loading and the `/completions` endpoint to be ready:

```bash
python llm_engine/engine_new.py
```

Then use the appropriate command in a second terminal from the repository
root:

```bash
python main.py --problem_name oscillator1 --spec_path specs/specification_oscillator1_numpy.txt --log_path logs/oscillator1/example
python main.py --problem_name oscillator2 --spec_path specs/specification_oscillator2_numpy.txt --log_path logs/oscillator2/example
python main.py --problem_name bactgrow --spec_path specs/specification_bactgrow_numpy.txt --log_path logs/bactgrow/example
python main.py --problem_name stressstrain --spec_path specs/specification_stressstrain_numpy.txt --log_path logs/stressstrain/example
```

These examples are derived from the released command-line interface, not
recovered historical experiment commands or validated end-to-end
reproduction commands. See the implementation caveats below.
`main.py` loads each task's `train.csv` and derives ID/OOD subsets with
`find_best_split`; the supplied test CSVs are not loaded by this entry point.

### Model and search settings

| Setting | Released source |
| --- | --- |
| Local model | `LLM-Research/Meta-Llama-3.1-8B-Instruct`, loaded with 4-bit quantization by default in [engine_new.py](llm_engine/engine_new.py) |
| Local server defaults | `temperature=0.8`, `top_k=30`, `top_p=0.9`, `do_sample=True`, `max_new_tokens=1024`, `num_return_sequences=1` |
| Equation candidates per prompt | `samples_per_prompt=4` in [config.py](mot_sr/config.py) |
| Search budget argument | `main.py` passes `max_sample_nums=2000` to the pipeline; this is a sample-count setting, not a verified mapping to the manuscript's iteration budget |
| Population configuration | `functions_per_prompt=1`, `num_islands=100`, `reset_period=0` in `config.py` |
| Scientist module | Enabled by default in `main.py`; local endpoint `http://127.0.0.1:5000/completions`, model label `deepseek-coder`, `temperature=0.7`, configured `max_tokens=512` |

The accompanying manuscript's "MOT-SR Configuration and Language Model
Details" appendix reports LLaMA-3.1-8B-Instruct on NVIDIA H100 80GB GPUs,
GPT-4o-mini through the OpenAI API, four candidates per iteration,
`top_k=30`, `top_p=0.3` and `temperature=0.6`. It reports 2000 iterations
for the four main benchmarks and 1000 for LSR-Synth-Chemistry. These are
manuscript-reported settings, not values recovered from original request
traces, and they differ from several released code defaults.

Current implementation caveats relevant to these commands:

- The local equation sampler explicitly sends `None` for `temperature`,
  `top_k` and `top_p`. The server uses these supplied values instead of
  falling back to its defaults, so the table does not establish the
  effective decoding settings of a successful run.
- The scientist's local request uses `max_tokens`, while the supplied
  server reads `max_new_tokens`; its configured 512-token limit is not
  applied by this server. The local server loads the backbone specified
  by `--model_path`, independently of the scientist model label.

These existing implementation issues are documented here; this materials
release preserves the search implementation.

## Random seeds

The supplied reproducibility materials list the following seeds for
repeated runs:

```text
42, 43, 44, 45, 46
```

The current entry point has no `--seed` option and does not initialize
NumPy, Python or PyTorch random-number generators with these values.
`--run_id` is parsed but does not set a seed. The API payloads do not send
a `seed` field. The original run-to-seed mapping and which generators
were seeded are not included in the supplied records. The list therefore
does not establish deterministic execution of the current commands.

## GPT-4o-mini API documentation

OpenAI's [GPT-4o-mini model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini)
lists the alias `gpt-4o-mini` and the dated snapshot
`gpt-4o-mini-2024-07-18`. The [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
documents request fields such as the model identifier, decoding settings
and output-token limit.

These links document the API model and interface. The exact model
identifier, API access dates and request settings used for an experiment
are determined from its original run configuration and API records.

In the released equation generator, `--use_api True` selects the OpenAI
Chat Completions transport and `--api_model` supplies the request model
identifier. The default is `gpt-3.5-turbo`; selecting GPT-4o-mini requires
an explicit override. The credential is read from the `API_KEY`
environment variable and must be supplied privately by the person running
the experiment. Enabling equation-generator API mode does not switch the
scientist module away from the local endpoint configured by `main.py`.

The equation-generator request explicitly sends `model`, a single user
message and `max_tokens=512`. It makes four separate requests per prompt.
It does not explicitly send `temperature`, `top_p`, `n`, `stop` or `seed`;
omitted fields use provider behavior rather than a recovered paper setting.
The scientist module also has a separate API transport, which explicitly
sends its model name, `temperature` and `max_tokens`, but that transport
is not selected by the default `main.py` scientist configuration.

Original requested and returned API model identifiers, access dates,
response metadata and per-run decoding overrides are not supplied. The
dated snapshot above is an official model identifier, not a claim that
the experiments requested that snapshot.

## Failure and retry handling

- [sampler.py](mot_sr/sampler.py) retries equation-generator local and API
  request exceptions in unbounded loops, without an explicit retry limit,
  delay or request timeout.
- [engine_new.py](llm_engine/engine_new.py) clears caches and retries after
  CUDA out-of-memory errors, without a retry cap.
- The scientist's local HTTP request has a 120-second timeout. Its API
  transport has no explicit timeout, and scientist initialization failures
  disable that module.
- [evaluator.py](mot_sr/evaluator.py) runs candidate evaluations in child
  processes with a configured 30-second timeout per dataset evaluation;
  timed-out processes are terminated, and execution exceptions are
  reported as failed evaluations.
- [evaluate_on_problems.py](mot_sr/evaluate_on_problems.py) fits ten
  parameters with single-start BFGS, initialized uniformly in `[-1, 1]`,
  using `maxiter=500`, `gtol=1e-10` and `eps=1e-12`. Non-finite fitting
  losses are rejected; the optimizer's `success` flag is not explicitly
  checked. No additional fitting restart policy is implemented there.

## Candidate logs

Raw candidate logs will be added under `logs/`, organized by task and run.
No original candidate logs are included in the supplied archive or this
repository release, so no historical run coverage is claimed.

[mot_sr/profile.py](mot_sr/profile.py) can write
`<log_path>/samples/samples_<sample_order>.json` files containing
`sample_order`, `function` and `score`, along with TensorBoard statistics.
These describe the logger's output format, not released original logs.
When original records are released, their task/run coverage and any
available fitting status and fitted coefficients should be documented.
Credentials and private request headers must be excluded from published
copies.

## Final equations

The four benchmark equation structures from MOT-SR (LLaMA-3.1) are in
`results/final_equations/mot_sr_llama31/`. The final EMRI correction is in
`results/final_equations/emri/equation.py`, with its complete supplied
parameter vector in `results/final_equations/emri/params.json`.

See the [final equation documentation](results/final_equations/README.md)
for the expressions, function interfaces and coefficient information.
The four benchmark files preserve equation structures; their original
fitted coefficient vectors were not supplied and must be recovered or
refitted before numerical benchmark evaluation. The EMRI file includes
all ten supplied coefficients, of which indices 0, 1 and 2 are active.

## EMRI data generation

The EMRI dataset was generated by the authors. The data-generation
scripts are coming soon.

## Automation and scientific interpretation

Equation search, iterative candidate filtering, Pareto-front maintenance
and final equation selection are fully automated. Scientific interpretation
of the EMRI correction was conducted with two domain-expert coauthors.
